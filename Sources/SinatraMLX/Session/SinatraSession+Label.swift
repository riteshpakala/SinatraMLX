//
//  SinatraSession+Label.swift
//  SinatraMLX
//
//  WHAT: The feedback channel. A pending turn is labelled by the next user message
//        (or by the no-reply deadline), and the 30-day band is enforced.
//

import Foundation

extension SinatraSession {

    /// Enforce the band and caps; label overdue turns "no reply"; drop turns whose
    /// generation never finished.
    func sweep(_ ledger: inout OwnerLedger, now: Date) {
        var abandoned = Set<UUID>()
        for i in ledger.turns.indices where ledger.turns[i].isPending {
            if let finished = ledger.turns[i].assistantFinishedAt {
                if now.timeIntervalSince(finished) > configuration.replyDeadline {
                    label(&ledger, turnIndex: i, reply: nil, labelAt: finished.addingTimeInterval(configuration.replyDeadline))
                }
            } else if now.timeIntervalSince(ledger.turns[i].at) > 3600 {
                abandoned.insert(ledger.turns[i].id)
            }
        }
        if !abandoned.isEmpty {
            ledger.turns.removeAll { abandoned.contains($0.id) }
            ledger.events.removeAll { abandoned.contains($0.turnId) }
        }

        let horizon = now.addingTimeInterval(-configuration.retentionDays * 86_400)
        ledger.turns.removeAll { $0.at < horizon }
        ledger.events.removeAll { $0.at < horizon }
        ledger.series.removeAll { $0.at < horizon }

        if ledger.turns.count > configuration.maxTurnsPerOwner {
            let dropped = Set(ledger.turns.prefix(ledger.turns.count - configuration.maxTurnsPerOwner).map(\.id))
            ledger.turns.removeAll { dropped.contains($0.id) }
            ledger.events.removeAll { dropped.contains($0.turnId) }
        }
        if ledger.events.count > configuration.maxEventsPerOwner {
            ledger.events.removeFirst(ledger.events.count - configuration.maxEventsPerOwner)
        }
        if ledger.series.count > configuration.maxSeriesPoints {
            ledger.series.removeFirst(ledger.series.count - configuration.maxSeriesPoints)
        }
        Self.cap(&ledger.firstSeenPartitions, to: configuration.maxFirstSeenEntries)
        Self.cap(&ledger.firstSeenDocuments, to: configuration.maxFirstSeenEntries)
    }

    static func cap(_ dates: inout [String: Date], to limit: Int) {
        guard dates.count > limit else { return }
        let oldest = dates.sorted { $0.value < $1.value }.prefix(dates.count - limit).map(\.key)
        for key in oldest { dates[key] = nil }
    }

    /// Label the most recent completed pending turn this reply answers.
    func labelLatestPending(
        _ ledger: inout OwnerLedger, reply: UserMessage, conversationId: String?, now: Date
    ) -> LabelDiagnostic? {
        guard let index = ledger.turns.indices.last(where: { i in
            let turn = ledger.turns[i]
            guard turn.isPending, let finished = turn.assistantFinishedAt, finished <= reply.at else { return false }
            if let conversationId, let other = turn.conversationId, other != conversationId { return false }
            return true
        }) else { return nil }
        return label(&ledger, turnIndex: index, reply: reply, labelAt: reply.at)
    }

    /// Compute the implicit signals for one turn and label its events.
    @discardableResult
    func label(_ ledger: inout OwnerLedger, turnIndex i: Int, reply: UserMessage?, labelAt: Date) -> LabelDiagnostic {
        let turn = ledger.turns[i]
        let eventIndices = ledger.eventIndices(turn: turn.id)
        let partitionSets = eventIndices.map { Set(ledger.events[$0].content ?? []) }
        let idf = ImplicitReward.idf(partitions: partitionSets)

        let replySet: Set<Int> = reply.map {
            filter.contentSet(tokenizer.encode($0.text), hot: ledger.hotTokens, configuration: configuration)
        } ?? []
        let echoes = partitionSets.map {
            ImplicitReward.echo(reply: replySet, partition: $0, idf: idf, partitionCount: partitionSets.count)
        }
        let assistant = Set(turn.assistantContent ?? [])
        let attributions = partitionSets.map { ImplicitReward.attribution(assistant: assistant, partition: $0) }

        let latency = reply.map { $0.at.timeIntervalSince(turn.assistantFinishedAt ?? turn.at) }
        let signals = ImplicitReward.signals(
            latency: latency, assistantWords: turn.assistantWords ?? 0,
            replyWords: reply.map { ImplicitReward.wordCount($0.text) } ?? 0,
            sameConversation: true, echoMax: echoes.max() ?? 0, configuration: configuration)
        let usage = ImplicitReward.usage(echo: echoes, attribution: attributions)
        let kindWeight = configuration.kindWeights.weight(for: signals.kind)

        for (k, e) in eventIndices.enumerated() {
            let reward = ImplicitReward.eventReward(turnReward: signals.reward, usage: usage[k])
            ledger.events[e].label = EventLabel(
                reward: reward, target: 2 * reward - 1, kindWeight: kindWeight,
                echo: echoes[k], attribution: attributions[k], turnReward: signals.reward,
                kind: signals.kind, labelledAt: labelAt)
            ledger.events[e].content = nil
        }
        ledger.turns[i].signals = signals
        ledger.turns[i].labelledAt = labelAt
        ledger.turns[i].assistantContent = nil
        ledger.training.labelsSinceTrain += eventIndices.count

        let point = SeriesPoint(at: reply?.at ?? labelAt, value: Double(signals.reward), volume: signals.replyWords)
        if let last = ledger.series.last, last.at > point.at {
            let insertAt = ledger.series.firstIndex { $0.at > point.at } ?? ledger.series.count
            ledger.series.insert(point, at: insertAt)
        } else {
            ledger.series.append(point)
        }
        return LabelDiagnostic(turnId: turn.id, signals: signals, labelledEvents: eventIndices.count)
    }

    // MARK: Assistant side

    /// Record the assistant's side of a turn. Returns true when training is due.
    @discardableResult
    public func generationDidFinish(
        owner: OwnerID, turnId: UUID, assistantText: String, startedAt: Date, finishedAt: Date,
        sampledTokenIds: [Int]? = nil, trace: InjectionTrace? = nil
    ) -> Bool {
        let tokens = sampledTokenIds ?? tokenizer.encode(assistantText)
        store.update(owner, now: finishedAt) { ledger in
            guard let i = ledger.turnIndex(turnId) else { return }
            ledger.turns[i].assistantStartedAt = startedAt
            ledger.turns[i].assistantFinishedAt = finishedAt
            ledger.turns[i].assistantWords = ImplicitReward.wordCount(assistantText)
            ledger.turns[i].assistantChars = assistantText.count
            let content = filter.contentSet(tokens, hot: ledger.hotTokens, configuration: configuration)
            ledger.turns[i].assistantContent = Array(content.sorted().prefix(configuration.maxAssistantContentTokens))
            if let trace {
                ledger.turns[i].trace = TraceAggregates(trace)
                ledger.lastTraceId = trace.traceId
            }
        }
        if let trace {
            do {
                try store.saveTrace(trace, owner: owner)
            } catch {
                log?.log(.warning, "saving trace \(trace.traceId) failed: \(error)")
            }
        }
        return trainingDue(owner: owner)
    }

    /// Keep a trace without recording its turn (dry runs and paired comparisons), so it
    /// can still be read back by id.
    public func storeTrace(_ trace: InjectionTrace, owner: OwnerID) {
        do {
            try store.saveTrace(trace, owner: owner)
        } catch {
            log?.log(.warning, "saving trace \(trace.traceId) failed: \(error)")
        }
    }

    /// The generation was cancelled: forget the turn rather than label a partial answer.
    public func generationAbandoned(owner: OwnerID, turnId: UUID) {
        store.update(owner) { ledger in
            ledger.turns.removeAll { $0.id == turnId }
            ledger.events.removeAll { $0.turnId == turnId }
        }
    }

    /// Explicit labelling for callers that keep their own timeline (replay, tests).
    public func observe(owner: OwnerID, turn observation: TurnObservation) throws -> ObservationReceipt {
        let now = observation.reply?.at ?? observation.assistantFinishedAt ?? Date()
        return try store.update(owner, now: now) { ledger in
            guard let i = ledger.turnIndex(observation.turnId) else {
                throw SinatraError.unknownTurn(observation.turnId)
            }
            if ledger.turns[i].assistantFinishedAt == nil {
                let text = observation.assistantText ?? ""
                let finished = observation.assistantFinishedAt ?? observation.reply?.at ?? now
                ledger.turns[i].assistantStartedAt = observation.assistantStartedAt ?? finished
                ledger.turns[i].assistantFinishedAt = finished
                ledger.turns[i].assistantWords = ImplicitReward.wordCount(text)
                ledger.turns[i].assistantChars = text.count
                let content = filter.contentSet(tokenizer.encode(text), hot: ledger.hotTokens, configuration: configuration)
                ledger.turns[i].assistantContent = Array(content.sorted().prefix(configuration.maxAssistantContentTokens))
            }
            let diagnostic = label(&ledger, turnIndex: i, reply: observation.reply, labelAt: now)
            var perPartition: [String: Float] = [:]
            for e in ledger.eventIndices(turn: observation.turnId) {
                if let reward = ledger.events[e].label?.reward { perPartition[ledger.events[e].partitionId] = reward }
            }
            return ObservationReceipt(turnId: observation.turnId, signals: diagnostic.signals, perPartition: perPartition)
        }
    }

    func trainingDue(owner: OwnerID) -> Bool {
        let ledger = store.ledger(owner)
        return ledger.labelledEvents >= configuration.minimumLabelledEvents
            && ledger.training.labelsSinceTrain >= configuration.trainEveryLabelledEvents
    }

    public func isTrainingDue(owner: OwnerID) -> Bool { trainingDue(owner: owner) }
}
