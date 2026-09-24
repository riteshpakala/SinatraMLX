//
//  SinatraHarness.swift
//  SinatraMLX
//
//  WHAT: The on-device LLM harness. One resident MLX model, one generation at a time, and
//        the Sinatra side model wired between the transformer and the decoder:
//
//          prompt + context ─▶ chat template ─▶ transformer ─▶ logits ─┐
//          retrieved context only ─▶ embedding table ─▶ SinatraNet ─▶ bias ─┴▶ + ─▶ sampler ─▶ tokens
//          user behaviour over time (implicit feedback) ─▶ labels ─▶ training ─┘
//
//  PIN:  The gate serialises everything that touches MLX: prefill and decode, context
//        encoding, trace evaluation, weight-model training, model swaps. Training runs
//        after a generation, bounded, and yields to any generation waiting for the gate.
//

import Foundation
import FrigateBridge
import MLX
import MLXLLM
import MLXLMCommon

public actor SinatraHarness {
    public nonisolated let storeDirectory: URL
    public let configuration: SinatraConfiguration

    let downloader: any Downloader
    let tokenizerLoader: any TokenizerLoader
    let weightModelFactory: WeightModelFactory
    let log: SinatraLog?

    private var context: ModelContext?
    private var session: SinatraSession?
    private var tokenizer: (any SinatraTokenizing)?
    private var encoder: LanguageModelContextEncoder?
    public private(set) var state: HarnessLoadState = .cold
    public private(set) var modelKey: String?
    private var loading: (id: String, task: Task<Void, Error>)?

    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private let waiting = WaitingCounter()
    private var trainingTask: Task<Void, Never>?
    public private(set) var lastTrainingReport: TrainingReport?

    public init(
        storeDirectory: URL,
        configuration: SinatraConfiguration = .init(),
        downloader: any Downloader = HubDownloader(),
        tokenizerLoader: any TokenizerLoader = HubTokenizerLoader(),
        weightModelFactory: WeightModelFactory? = nil,
        log: SinatraLog? = nil
    ) {
        self.storeDirectory = storeDirectory
        self.configuration = configuration
        self.downloader = downloader
        self.tokenizerLoader = tokenizerLoader
        self.weightModelFactory = weightModelFactory ?? MLXWeightModel.factory(configuration: configuration)
        self.log = log
    }

    // MARK: - Residency

    /// Load (downloading on first use) a Hugging Face model id. A different resident model
    /// is evicted under the gate, never out from under a running generation.
    public func load(modelID: String, progress: (@Sendable (Double) -> Void)? = nil) async throws {
        if modelKey == modelID, context != nil { return }
        if let loading, loading.id == modelID {
            try await loading.task.value
            return
        }
        let task = Task { try await self.performLoad(modelKey: modelID, directory: nil, progress: progress) }
        loading = (modelID, task)
        defer { if loading?.id == modelID { loading = nil } }
        try await task.value
    }

    /// Load a model directory already on disk; `modelKey` names it in the store.
    public func load(directory: URL, modelKey: String) async throws {
        if self.modelKey == modelKey, context != nil { return }
        try await performLoad(modelKey: modelKey, directory: directory, progress: nil)
    }

    private func performLoad(modelKey: String, directory: URL?, progress: (@Sendable (Double) -> Void)?) async throws {
        if context != nil {
            await acquire()
            await session?.flush()
            context = nil
            session = nil
            tokenizer = nil
            encoder = nil
            self.modelKey = nil
            release()
        }
        state = .loading(0)
        log?.log(.info, "loading \(modelKey)")
        do {
            let loaded: ModelContext
            if let directory {
                loaded = try await MLXLMCommon.loadModel(from: directory, using: tokenizerLoader)
            } else {
                loaded = try await MLXLMCommon.loadModel(
                    from: downloader, using: tokenizerLoader,
                    configuration: ModelConfiguration(id: modelKey),
                    progressHandler: { fraction in
                        progress?(fraction.fractionCompleted)
                        Task { await self.noteProgress(fraction.fractionCompleted) }
                    })
            }
            install(loaded, modelKey: modelKey)
            log?.log(.info, "ready \(modelKey)")
        } catch {
            state = .failed(String(describing: error))
            throw error
        }
    }

    private func install(_ loaded: ModelContext, modelKey: String) {
        let tokenizer = MLXLMTokenizer(loaded.tokenizer, configuration: loaded.configuration)
        var encoder: LanguageModelContextEncoder?
        do {
            encoder = try LanguageModelContextEncoder(model: loaded.model, modelKey: modelKey)
        } catch {
            log?.log(.warning, "no context encoder for \(modelKey): \(error); the injection falls back to priors on hashed context")
        }
        let vocabularySize = encoder?.vocabularySize ?? 0
        session = SinatraSession(
            configuration: configuration, storeDirectory: storeDirectory, modelKey: modelKey,
            tokenizer: tokenizer, encoder: encoder ?? HashingContextEncoder(), vocabularySize: vocabularySize,
            weightModelFactory: weightModelFactory, log: log)
        context = loaded
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.modelKey = modelKey
        state = .ready(modelKey)
    }

    private func noteProgress(_ fraction: Double) {
        if case .loading = state { state = .loading(fraction) }
    }

    public func unload() async {
        await acquire()
        await session?.flush()
        context = nil
        session = nil
        tokenizer = nil
        encoder = nil
        modelKey = nil
        state = .cold
        release()
    }

    public var isLoaded: Bool { context != nil }
    public var vocabularySize: Int? { encoder?.vocabularySize }
    public var tokenizerAdapter: (any SinatraTokenizing)? { tokenizer }

    // MARK: - Generation

    /// One generation. With a `turn`, the side model plans the injection from the retrieved
    /// context, the decode runs with it, and the turn is recorded for its reply to label.
    public func generate(_ request: GenerateRequest) async throws -> GenerationHandle {
        waiting.increment()
        await acquire()
        waiting.decrement()
        guard let context, let session else {
            release()
            throw SinatraError.notLoaded
        }
        do {
            let input = try await context.processor.prepare(input: request.input)
            var plan: InjectionPlan?
            if let turn = request.turn {
                plan = try await session.prepareTurn(turn, mode: request.mode, dryRun: !request.record)
            }
            let requested = request.trace == .automatic ? configuration.traceLevel : request.trace
            let run = try InjectedGeneration.start(
                input: input, parameters: request.parameters, context: context, plan: plan,
                trace: requested, topK: configuration.traceTopK, tools: request.tools)
            return stream(run: run, plan: plan, request: request, tokenizer: tokenizer)
        } catch {
            release()
            throw error
        }
    }

    private func stream(run: InjectedGeneration.Run, plan: InjectionPlan?, request: GenerateRequest, tokenizer: (any SinatraTokenizing)?) -> GenerationHandle {
        let startedAt = Date()
        let (outer, continuation) = AsyncStream<Generation>.makeStream()
        let completion = Task.detached { () -> TurnCompletion in
            var text = ""
            var info: GenerateCompletionInfo?
            for await item in run.stream {
                switch item {
                case .chunk(let chunk): text += chunk
                case .info(let completed): info = completed
                case .toolCall: break
                }
                continuation.yield(item)
            }
            continuation.finish()
            await run.task.value
            let cancelled = Task.isCancelled || info == nil || info?.stopReason == .cancelled
            let steps = run.drainTrace(info: info, tokenizer: tokenizer, mask: plan?.mask)
            return await self.finish(
                run: run, plan: plan, request: request, text: text, info: info, steps: steps,
                startedAt: startedAt, finishedAt: Date(), cancelled: cancelled)
        }
        continuation.onTermination = { termination in
            if case .cancelled = termination { completion.cancel() }
        }
        return GenerationHandle(stream: outer, plan: plan, completion: completion)
    }

    private func finish(
        run: InjectedGeneration.Run, plan: InjectionPlan?, request: GenerateRequest, text: String,
        info: GenerateCompletionInfo?, steps: [StepTrace], startedAt: Date, finishedAt: Date, cancelled: Bool
    ) async -> TurnCompletion {
        var trace: InjectionTrace?
        if run.traceLevel != .off {
            let mask = run.injected ? plan?.mask : nil
            trace = InjectionTrace(
                traceId: plan?.turnId ?? UUID(), owner: request.turn?.owner.rawValue, level: run.traceLevel,
                mode: run.injected ? (plan?.mode ?? .off) : .off, seed: run.seed,
                temperature: request.parameters.temperature, topP: request.parameters.topP,
                createdAt: finishedAt, promptTokens: info?.promptTokenCount ?? 0,
                generatedTokens: info?.generationTokenCount ?? 0,
                stopReason: info.map { String(describing: $0.stopReason) },
                mask: mask, steps: steps, summary: TraceSummary.compute(steps: steps, mask: mask))
        }
        let owner = request.turn?.owner
        var due = false
        if let trace, let owner, let session, !request.record || plan?.dryRun == true {
            await session.storeTrace(trace, owner: owner)
        }
        if let plan, let owner, let session, request.record, !plan.dryRun {
            if cancelled {
                await session.generationAbandoned(owner: owner, turnId: plan.turnId)
            } else {
                due = await session.generationDidFinish(
                    owner: owner, turnId: plan.turnId, assistantText: text,
                    startedAt: request.timeline?.startedAt ?? startedAt,
                    finishedAt: request.timeline?.finishedAt ?? finishedAt,
                    sampledTokenIds: steps.isEmpty ? nil : steps.map(\.sampled),
                    trace: trace)
            }
        }
        release()

        var summary: OwnerSummary?
        if let owner, let session {
            summary = await session.summary(owner: owner)
            if request.record {
                Task.detached { await session.persist(owner: owner) }
            }
        }
        if due, let owner { scheduleTraining(owner) }
        return TurnCompletion(
            text: text, info: info, trace: trace, summary: summary, trainingScheduled: due, cancelled: cancelled)
    }

    /// What the side model would do for `turn`, without recording anything.
    public func preview(_ turn: TurnInput, mode: BiasMode? = nil) async throws -> InjectionPlan {
        guard let session else { throw SinatraError.notLoaded }
        await acquire()
        defer { release() }
        return try await session.prepareTurn(turn, mode: mode, dryRun: true)
    }

    // MARK: - Training

    private func scheduleTraining(_ owner: OwnerID) {
        guard trainingTask == nil, let session else { return }
        let waiting = self.waiting
        trainingTask = Task {
            await self.acquire()
            var report: TrainingReport?
            do {
                report = try await session.train(owner: owner, shouldAbort: { waiting.value > 0 })
            } catch {
                self.log?.log(.error, "training \(owner) failed: \(error)")
            }
            self.release()
            await session.persist(owner: owner)
            self.trainingFinished(report)
        }
    }

    private func trainingFinished(_ report: TrainingReport?) {
        trainingTask = nil
        if let report { lastTrainingReport = report }
    }

    /// Train now (under the gate). `force` lowers the minimum to a handful of labels.
    public func train(owner: OwnerID, force: Bool = false, budget: TimeInterval? = nil) async throws -> TrainingReport {
        guard let session else { throw SinatraError.notLoaded }
        await acquire()
        defer { release() }
        let waiting = self.waiting
        let report = try await session.train(owner: owner, budget: budget, force: force, shouldAbort: { waiting.value > 0 })
        await session.persist(owner: owner)
        lastTrainingReport = report
        return report
    }

    /// Wait for any scheduled training to finish (tests, CLI replay).
    public func awaitTraining() async {
        await trainingTask?.value
    }

    // MARK: - Status

    public func summary(owner: OwnerID) async -> OwnerSummary? {
        await session?.summary(owner: owner)
    }

    public func trace(owner: OwnerID, turnId: UUID) async -> InjectionTrace? {
        await session?.trace(owner: owner, turnId: turnId)
    }

    public func latestTrace(owner: OwnerID) async -> InjectionTrace? {
        await session?.latestTrace(owner: owner)
    }

    public func entropyReport(owner: OwnerID) async -> EntropyReport? {
        await session?.entropyReport(owner: owner)
    }

    public func forget(owner: OwnerID) async throws {
        try await session?.forget(owner: owner)
    }

    public func flush() async {
        await trainingTask?.value
        await session?.flush()
    }

    public var currentSession: SinatraSession? { session }

    var currentContext: ModelContext? { context }

    /// Pooled embeddings of arbitrary texts through the loaded model's table (CLI `encode`).
    public func encode(texts: [String]) async throws -> [[Float]] {
        guard let encoder, let tokenizer else { throw SinatraError.notLoaded }
        await acquire()
        defer { release() }
        let rows = texts.map { TokenBatch.clip(tokenizer.encode($0), maxTokens: configuration.maxTokensPerPartition) }
        return try encoder.encode(TokenBatch(rows: rows))
    }

    /// Run `body` with the loaded context under the gate.
    func withContext<R>(_ body: (ModelContext) throws -> R) async throws -> R {
        guard let context else { throw SinatraError.notLoaded }
        await acquire()
        defer { release() }
        return try body(context)
    }

    // MARK: - The gate

    private func acquire() async {
        while busy {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                waiters.append(continuation)
            }
        }
        busy = true
    }

    private func release() {
        busy = false
        if !waiters.isEmpty { waiters.removeFirst().resume() }
    }
}

/// Generations waiting for the gate; training stops at its next step when this is > 0.
final class WaitingCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func increment() { lock.lock(); count += 1; lock.unlock() }
    func decrement() { lock.lock(); count -= 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}
