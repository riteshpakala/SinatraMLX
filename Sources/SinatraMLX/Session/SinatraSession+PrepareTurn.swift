//
//  SinatraSession+PrepareTurn.swift
//  SinatraMLX
//
//  WHAT: Everything that happens before the first token: label the previous turn from
//        this user message, featurise and encode the retrieved partitions, blend learned
//        and prior weights, and build the injection.
//

import Foundation

extension SinatraSession {

    /// Plan one generation. `dryRun` computes exactly what a real turn would without
    /// recording anything (paired comparisons, previews).
    public func prepareTurn(_ input: TurnInput, mode requestedMode: BiasMode? = nil, dryRun: Bool = false) throws -> InjectionPlan {
        let started = Date()
        let now = input.now
        let owner = input.owner
        let mode = requestedMode ?? configuration.biasMode
        let turnId = UUID()

        // 1. The feedback channel: this message labels the previous turn.
        var ledger: OwnerLedger
        var labelled: LabelDiagnostic?
        if dryRun {
            ledger = store.ledger(owner, now: now)
            sweep(&ledger, now: now)
            if let message = input.userMessage {
                labelled = labelLatestPending(&ledger, reply: message, conversationId: input.conversationId, now: now)
            }
        } else {
            labelled = store.update(owner, now: now) { ledger -> LabelDiagnostic? in
                sweep(&ledger, now: now)
                guard let message = input.userMessage else { return nil }
                return labelLatestPending(&ledger, reply: message, conversationId: input.conversationId, now: now)
            }
            ledger = store.ledger(owner, now: now)
        }

        // 2. The retrieved context — the only text the side model reads.
        var seen = Set<String>()
        let partitions = Array(input.retrieved.filter { seen.insert($0.id).inserted }.prefix(configuration.maxPartitionsPerTurn))
        let rows = partitions.map {
            TokenBatch.clip($0.tokenIds ?? tokenizer.encode($0.text), maxTokens: configuration.maxTokensPerPartition)
        }
        let terms = rows.map { filter.contentCounts($0, hot: ledger.hotTokens, configuration: configuration) }

        // 3. Features.
        let index = LedgerIndex(ledger: ledger, now: now, configuration: configuration)
        let indicators = IndicatorSeries.features(
            points: ledger.series, periods: ledger.periods, minimumCount: configuration.minimumSeriesForIndicators)
        let scores = Self.normalisedScores(partitions.map(\.score), isDistance: configuration.scoreIsDistance)
        let periodics = TimeFeatures.periodics(now, timeZone: configuration.timeZone)
        let last = ledger.lastSignals
        let pending = Float(min(ledger.pendingTurns, 5)) / 5
        let queryGap: Float = ledger.lastUserMessageAt.map {
            TimeFeatures.lnNorm(seconds: (input.userMessage?.at ?? now).timeIntervalSince($0), horizon: configuration.sessionWindow)
        } ?? 1

        var statics: [[Float]] = []
        var ageDays: [Double] = []
        var bands: [RelevancyBand] = []
        var bandWeights: [Float] = []
        var priors: [(reward: Float, count: Int)] = []
        for (rank, partition) in partitions.enumerated() {
            let reference = partition.createdAt ?? partition.indexedAt ?? ledger.firstSeenDocuments[partition.documentId] ?? now
            let age = TimeFeatures.days(now.timeIntervalSince(reference))
            let band = TimeFeatures.band(ageDays: age, configuration: configuration)
            let bandWeight = TimeFeatures.bandWeight(band, configuration: configuration)
            let stats = index.partition(partition.id)
            let documentPrior = index.document(partition.documentId).prior
            let firstSeen = ledger.firstSeenPartitions[partition.id]
            var values: [Float] = []
            values.reserveCapacity(FeatureVector.staticCount)
            values.append(TimeFeatures.lnNorm(days: age))
            values.append(partition.createdAt != nil ? 1 : 0)
            values.append(firstSeen.map { TimeFeatures.lnNorm(days: TimeFeatures.days(now.timeIntervalSince($0))) } ?? 0)
            values.append(firstSeen == nil ? 1 : 0)
            values.append(stats.lastRetrieved.map {
                TimeFeatures.lnNorm(days: TimeFeatures.days(now.timeIntervalSince($0)), horizon: configuration.retentionDays)
            } ?? 1)
            values.append(min(1, Float(stats.retrievals30) / 10))
            values.append(min(1, Float(stats.retrievals7) / 5))
            values.append(stats.prior.reward)
            values.append(min(1, Float(stats.prior.count) / 5))
            values.append(documentPrior.reward)
            values.append(partitions.count > 1 ? 1 - Float(rank) / Float(partitions.count - 1) : 1)
            values.append(scores[rank])
            values.append(min(1, Float(rows[rank].count) / 512))
            values.append(last?.pace ?? 0.5)
            values.append(last?.length ?? 0.5)
            values.append(last?.continuation ?? 1)
            values.append(pending)
            values.append(contentsOf: periodics)
            values.append(queryGap)
            statics.append(values.map { $0.finite().rounded4 })
            ageDays.append(age)
            bands.append(band)
            bandWeights.append(bandWeight)
            priors.append(stats.prior)
        }

        // 4. Encode the context (never the prompt).
        let encodeStarted = Date()
        var pooled: [[Float]]?
        if let encoder, !rows.isEmpty {
            do {
                pooled = try encoder.encode(TokenBatch(rows: rows))
            } catch {
                log?.log(.warning, "context encoding failed: \(error)")
            }
        }
        let contextValid = pooled != nil
        let contexts: [[Float]] = pooled.map { vectors in
            let projection = contextProjection(inputDimension: encoder?.hiddenSize ?? vectors.first?.count ?? 1)
            return vectors.map { projection.project($0) }
        } ?? Array(repeating: Array(repeating: 0, count: configuration.contextDim), count: partitions.count)
        let encodeMillis = Date().timeIntervalSince(encodeStarted) * 1000

        // 5. Weights: learned where reliable, ledger priors otherwise.
        let model = weightModel(for: owner)
        let features = statics.map { FeatureVector.assemble(static: $0, indicators: indicators, contextValid: contextValid) }
        let gate: Float = model.isTrained ? ledger.training.reliability : 0
        var prediction: WeightPrediction?
        if gate > 0, !partitions.isEmpty {
            do {
                prediction = try model.predict(features: features, context: contexts)
            } catch {
                log?.log(.warning, "weight model prediction failed: \(error)")
            }
        }
        let effectiveGate: Float = prediction == nil ? 0 : gate
        var weights: [Float] = []
        var priorWeights: [Float] = []
        for p in partitions.indices {
            let prior = priors[p]
            let confidence = min(1, Float(prior.count) / 5)
            let advantage = tanh(configuration.advantageScale * (prior.reward - index.meanReward))
            let priorWeight = configuration.coldStartPrior ? bandWeights[p] * advantage * confidence : 0
            let netWeight = prediction?.weights[p] ?? 0
            priorWeights.append(priorWeight)
            weights.append(((1 - effectiveGate) * priorWeight + effectiveGate * netWeight).clamped(-1, 1))
        }
        if let override = input.weightOverride {
            for (p, value) in override.prefix(weights.count).enumerated() { weights[p] = value.clamped(-1, 1) }
        }

        // 6. The injection.
        let buildStarted = Date()
        let partitionIds = partitions.map(\.id)
        var built: (SparseBias, ImpactMask)?
        switch mode {
        case .off:
            built = nil
        case .lexical:
            built = lexical(partitionIds: partitionIds, terms: terms, weights: weights, bandWeights: bandWeights)
        case .dense:
            if let pooled, let logits = try? encoder?.outputLogits(pooled) {
                built = DenseBiasCombiner.build(
                    headLogits: logits, partitionIds: partitionIds, weights: weights, bandWeights: bandWeights,
                    vocabularySize: vocabularySize, configuration: configuration,
                    isContent: { self.filter.isContent($0) })
            } else {
                log?.log(.info, "dense mode unavailable for this encoder; using lexical")
                built = lexical(partitionIds: partitionIds, terms: terms, weights: weights, bandWeights: bandWeights)
            }
        }
        let buildMillis = Date().timeIntervalSince(buildStarted) * 1000
        let bias = built?.0
        var mask = built?.1
        if let ids = mask?.tokenIds {
            mask?.tokenTexts = ids.map { tokenizer.decode([Int($0)]) }
        }

        // 7. Record the turn; it waits for its reply.
        if !dryRun {
            store.update(owner, now: now) { ledger in
                ledger.turns.append(TurnRecord(
                    id: turnId, at: now, userMessageAt: input.userMessage?.at,
                    conversationId: input.conversationId, modelKey: modelKey, mode: mode,
                    partitionIds: partitionIds,
                    injection: InjectionAggregates(
                        biasMaxAbs: bias?.maxAbs ?? 0, biasNonZero: bias?.nonZero ?? 0, biasL1: bias?.l1 ?? 0,
                        gate: effectiveGate, weightedPartitions: weights.filter { abs($0) >= configuration.minimumWeight }.count)))
                for (p, partition) in partitions.enumerated() {
                    ledger.events.append(RetrievalEvent(
                        turnId: turnId, at: now, partitionId: partition.id, documentId: partition.documentId,
                        rank: p, staticFeatures: statics[p],
                        context: contextValid ? contexts[p] : nil,
                        contextModelKey: contextValid ? modelKey : nil,
                        content: terms[p].keys.sorted(),
                        priorWeight: priorWeights[p].rounded4,
                        netWeight: prediction.map { $0.weights[p].rounded4 },
                        appliedWeight: weights[p].rounded4, bandWeight: bandWeights[p], label: nil))
                    if ledger.firstSeenPartitions[partition.id] == nil { ledger.firstSeenPartitions[partition.id] = now }
                    if ledger.firstSeenDocuments[partition.documentId] == nil {
                        ledger.firstSeenDocuments[partition.documentId] = partition.indexedAt ?? now
                    }
                    ledger.hotTokens.observe(Set(terms[p].keys))
                }
                ledger.lastUserMessageAt = input.userMessage?.at ?? now
                ledger.lastBiasMagnitude = bias?.maxAbs ?? 0
            }
        }

        // 8. Diagnostics.
        let diagnostics = TurnDiagnostics(
            turnId: turnId, mode: mode,
            coldStart: ledger.labelledEvents < configuration.minimumLabelledEvents,
            labelledEvents: ledger.labelledEvents, labelledTurns: ledger.labelledTurns,
            observedTurns: ledger.turns.count + (dryRun ? 0 : 1),
            partitions: partitions.indices.map { p in
                PartitionDiagnostic(
                    id: partitions[p].id, documentId: partitions[p].documentId, rank: p,
                    docAgeDays: (ageDays[p] * 100).rounded() / 100, band: bands[p].rawValue,
                    bandWeight: bandWeights[p], priorWeight: priorWeights[p],
                    netWeight: prediction?.weights[p], appliedWeight: weights[p],
                    contentTokens: terms[p].count, labelledBefore: priors[p].count)
            },
            weightedPartitions: weights.filter { abs($0) >= configuration.minimumWeight }.count,
            biasNonZero: bias?.nonZero ?? 0, biasMaxAbs: bias?.maxAbs ?? 0, biasL1: bias?.l1 ?? 0,
            gate: effectiveGate,
            forecastReward: prediction.map { $0.forecast.isEmpty ? 0 : $0.forecast.reduce(0, +) / Float($0.forecast.count) },
            periods: ledger.periods,
            topBiased: (bias?.top(12) ?? []).map {
                BiasedToken(id: $0.id, text: tokenizer.decode([$0.id]), bias: $0.bias)
            },
            labelledPrevious: labelled,
            encodeMillis: encodeMillis, buildMillis: buildMillis, dryRun: dryRun,
            features: features)
        log?.log(.debug, "prepareTurn \(owner) \(partitions.count) partitions, bias \(bias?.nonZero ?? 0) tokens in \(Int(Date().timeIntervalSince(started) * 1000)) ms")

        var weightMap: [String: Float] = [:]
        for (p, id) in partitionIds.enumerated() { weightMap[id] = weights[p] }
        return InjectionPlan(
            turnId: turnId, owner: owner, mode: mode, bias: bias, mask: mask,
            perPartitionWeights: weightMap, gate: effectiveGate, diagnostics: diagnostics, dryRun: dryRun)
    }

    func lexical(partitionIds: [String], terms: [[Int: Int]], weights: [Float], bandWeights: [Float]) -> (SparseBias, ImpactMask)? {
        LexicalBiasBuilder.build(
            .init(partitionIds: partitionIds, terms: terms, weights: weights, bandWeights: bandWeights),
            vocabularySize: vocabularySize, configuration: configuration,
            isExcluded: { self.filter.isControl($0) })
    }

    /// Min–max to [0, 1] within the turn, 1 = best. Equal scores → 0.5.
    static func normalisedScores(_ scores: [Float], isDistance: Bool) -> [Float] {
        guard let lo = scores.min(), let hi = scores.max(), hi > lo else {
            return Array(repeating: 0.5, count: scores.count)
        }
        return scores.map { value in
            let unit = (value - lo) / (hi - lo)
            return isDistance ? 1 - unit : unit
        }
    }
}

/// Per-partition and per-document statistics over the band, built once per turn.
struct LedgerIndex {
    struct Stats {
        var retrievals30 = 0
        var retrievals7 = 0
        var lastRetrieved: Date?
        var prior: (reward: Float, count: Int) = (0.5, 0)
    }

    private var partitions: [String: Stats] = [:]
    private var documents: [String: Stats] = [:]
    /// The owner's mean event reward over the band: the baseline advantages are measured against.
    private(set) var meanReward: Float = 0.5

    init(ledger: OwnerLedger, now: Date, configuration: SinatraConfiguration) {
        let horizon30 = now.addingTimeInterval(-configuration.retentionDays * 86_400)
        let horizon7 = now.addingTimeInterval(-configuration.freshBandDays * 86_400)
        var rewardSums: [String: (Float, Int)] = [:]
        var documentSums: [String: (Float, Int)] = [:]
        for event in ledger.events where event.at >= horizon30 {
            var stats = partitions[event.partitionId] ?? Stats()
            stats.retrievals30 += 1
            if event.at >= horizon7 { stats.retrievals7 += 1 }
            if stats.lastRetrieved.map({ event.at > $0 }) ?? true { stats.lastRetrieved = event.at }
            partitions[event.partitionId] = stats
            if let label = event.label {
                let sum = rewardSums[event.partitionId] ?? (0, 0)
                rewardSums[event.partitionId] = (sum.0 + label.reward, sum.1 + 1)
                let doc = documentSums[event.documentId] ?? (0, 0)
                documentSums[event.documentId] = (doc.0 + label.reward, doc.1 + 1)
            }
        }
        let totals = rewardSums.values.reduce((Float(0), 0)) { ($0.0 + $1.0, $0.1 + $1.1) }
        if totals.1 > 0 { meanReward = totals.0 / Float(totals.1) }
        for (id, sum) in rewardSums where sum.1 > 0 {
            partitions[id, default: Stats()].prior = (sum.0 / Float(sum.1), sum.1)
        }
        for (id, sum) in documentSums where sum.1 > 0 {
            documents[id, default: Stats()].prior = (sum.0 / Float(sum.1), sum.1)
        }
    }

    func partition(_ id: String) -> Stats { partitions[id] ?? Stats() }
    func document(_ id: String) -> Stats { documents[id] ?? Stats() }
}
