//
//  InjectedGeneration.swift
//  SinatraMLX
//
//  WHAT: Generation with the injection layer in the decode loop. Builds the token
//        iterator with [injection, penalties] (or the tracing wrappers) and hands it to
//        MLXLMCommon's own generate loop, so detokenisation, stop strings and tool-call
//        parsing are exactly what `MLXLMCommon.generate` does.
//  PIN:  With no injection and no trace this IS `MLXLMCommon.generate` (same iterator
//        init, KV quantisation kept). The direct iterator init used otherwise has no KV
//        quantisation, so the cache is created here from the parameters (keeps maxKVSize).
//

import Foundation
import MLX
import MLXLMCommon

public enum InjectedGeneration {

    /// A running generation: its stream, its task, and its trace once it ends.
    public final class Run: @unchecked Sendable {
        public let stream: AsyncStream<Generation>
        public let task: Task<Void, Never>
        public let traceLevel: TraceLevel
        public let seed: UInt64?
        public let injected: Bool
        let buffer: TraceBuffer?

        init(stream: AsyncStream<Generation>, task: Task<Void, Never>, traceLevel: TraceLevel, seed: UInt64?, injected: Bool, buffer: TraceBuffer?) {
            self.stream = stream
            self.task = task
            self.traceLevel = traceLevel
            self.seed = seed
            self.injected = injected
            self.buffer = buffer
        }

        /// Evaluate the recorded steps, trimmed to the tokens the loop consumed (the
        /// iterator computes one step ahead; a stop keeps its deciding step).
        public func drainTrace(info: GenerateCompletionInfo?, tokenizer: (any SinatraTokenizing)?, mask: ImpactMask?) -> [StepTrace] {
            guard let buffer else { return [] }
            var limit: Int?
            if let info {
                switch info.stopReason {
                case .stop: limit = info.generationTokenCount + 1
                default: limit = info.generationTokenCount
                }
            }
            let raw = buffer.drain(limit: limit)
            func text(_ id: Int) -> String? { tokenizer?.decode([id]) }
            return raw.enumerated().map { index, step in
                StepTrace(
                    index: index, sampled: step.sampled, counterfactual: step.counterfactual,
                    sampledText: text(step.sampled),
                    counterfactualText: step.counterfactual == step.sampled ? nil : text(step.counterfactual),
                    entropyPre: step.entropyPre, entropyPost: step.entropyPost, kl: step.kl, js: step.js,
                    logprobPre: step.logprobPre, logprobPost: step.logprobPost,
                    rankPre: step.rankPre, rankPost: step.rankPost,
                    argmaxPre: step.argmaxPre, argmaxPost: step.argmaxPost,
                    massIntoMask: step.massIntoMask, inMask: mask?.position(of: step.sampled) != nil,
                    topPre: step.topPre?.map { TokenLogprob(id: $0.0, text: text($0.0), logprob: $0.1) },
                    topPost: step.topPost?.map { TokenLogprob(id: $0.0, text: text($0.0), logprob: $0.1) },
                    movement: step.movement)
            }
        }
    }

    /// Start a generation with `plan`'s injection. `trace` is resolved against whether the
    /// plan actually injects (`.automatic` → `.summary` only when it does).
    public static func start(
        input: LMInput, parameters: GenerateParameters, context: ModelContext, plan: InjectionPlan?,
        trace: TraceLevel = .off, topK: Int = 8, cache: [KVCache]? = nil,
        tools: [[String: any Sendable]]? = nil, wiredMemoryTicket: WiredMemoryTicket? = nil
    ) throws -> Run {
        let injects = plan?.injects ?? false
        let level = trace.resolved(injecting: injects)
        var parameters = parameters
        if level != .off && parameters.seed == nil {
            // The counterfactual needs an identically seeded shadow sampler.
            parameters.seed = UInt64.random(in: 1...UInt64(Int64.max))
        }
        let injection: InjectionProcessor? = injects ? plan?.bias.map { InjectionProcessor(bias: $0) } : nil
        let promptTokens = input.text.tokens.size

        if injection == nil && level == .off {
            let iterator = try TokenIterator(input: input, model: context.model, cache: cache, parameters: parameters)
            let (stream, task) = generateTask(
                promptTokenCount: promptTokens, modelConfiguration: context.configuration,
                tokenizer: context.tokenizer, iterator: iterator, wiredMemoryTicket: wiredMemoryTicket, tools: tools)
            return Run(stream: stream, task: task, traceLevel: .off, seed: parameters.seed, injected: false, buffer: nil)
        }

        let cache = cache ?? context.model.newCache(parameters: parameters)
        let penalty = parameters.processor()
        let processor: any LogitProcessor
        let sampler: any LogitSampler
        var buffer: TraceBuffer?
        if level == .off {
            processor = CompositeLogitProcessor([injection, penalty].compactMap { $0 })
            sampler = parameters.sampler()
        } else {
            var maskIndices: MLXArray?
            if injects, let bias = plan?.bias, !bias.isEmpty {
                maskIndices = MLXArray(bias.indices, [bias.nonZero])
            }
            let traceBuffer = TraceBuffer(level: level, mask: maskIndices, topK: topK)
            buffer = traceBuffer
            processor = TracingLogitProcessor(injection: injection, penalty: penalty, buffer: traceBuffer)
            sampler = TracingSampler(real: parameters.sampler(), shadow: parameters.sampler(), buffer: traceBuffer)
        }
        let iterator = try TokenIterator(
            input: input, model: context.model, cache: cache, processor: processor, sampler: sampler,
            prefillStepSize: parameters.prefillStepSize, maxTokens: parameters.maxTokens)
        let (stream, task) = generateTask(
            promptTokenCount: promptTokens, modelConfiguration: context.configuration,
            tokenizer: context.tokenizer, iterator: iterator, wiredMemoryTicket: wiredMemoryTicket, tools: tools)
        return Run(stream: stream, task: task, traceLevel: level, seed: parameters.seed, injected: injection != nil, buffer: buffer)
    }

    /// Convenience: just the stream.
    public static func generate(
        input: LMInput, parameters: GenerateParameters, context: ModelContext, plan: InjectionPlan?,
        tools: [[String: any Sendable]]? = nil
    ) throws -> AsyncStream<Generation> {
        try start(input: input, parameters: parameters, context: context, plan: plan, trace: .off, tools: tools).stream
    }
}
