import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import SinatraMLX

/// Gated: needs `./scripts/metallib.sh debug` and FRIGATE_MLX_TESTS=1.
@Suite("MLX: injection, trace, weight model, encoder", .enabled(if: mlxAvailable))
struct MLXTests {

    @Test func injectionAddsTheBiasAndKeepsTheDtype() {
        let processor = InjectionProcessor(bias: SparseBias(vocabularySize: 8, entries: [3: 2.0, 5: -1.0]))
        let logits = MLXArray((0..<8).map { Float($0) * 0.1 }, [1, 8])
        let values = processor.process(logits: logits).asArray(Float.self)
        #expect(abs(values[3] - 2.3) < 1e-5)
        #expect(abs(values[5] - (-0.5)) < 1e-5)
        #expect(abs(values[0]) < 1e-6)
        #expect(processor.process(logits: logits.asType(.bfloat16)).dtype == .bfloat16)
        let wider = MLXArray.zeros([1, 9])
        #expect(processor.process(logits: wider).asArray(Float.self) == [Float](repeating: 0, count: 9))
    }

    @Test func traceMathIsZeroWithoutAnInjection() {
        let logits = MLXArray([1.0, 2.0, 0.5, -1.0] as [Float], [1, 4])
        let y = argMax(logits, axis: -1)
        let step = TraceMath.step(
            base: logits, injected: logits, sampled: y, counterfactual: y,
            mask: MLXArray([Int32(1)], [1]), topK: 2, full: true)
        eval(step.arrays)
        #expect(abs(step.kl.item(Float.self)) < 1e-6)
        #expect(abs(step.entropyPre.item(Float.self) - step.entropyPost.item(Float.self)) < 1e-6)
        #expect(abs(step.massIntoMask.item(Float.self)) < 1e-6)
        // Entropy against a double-precision reference.
        let z: [Double] = [1.0, 2.0, 0.5, -1.0]
        let lse = log(z.map(exp).reduce(0, +))
        let reference = -z.map { exp($0 - lse) * ($0 - lse) }.reduce(0, +)
        #expect(abs(Double(step.entropyPre.item(Float.self)) - reference) < 1e-5)
    }

    @Test func traceMathMeasuresABoost() {
        let base = MLXArray([1.0, 2.0, 0.5, -1.0] as [Float], [1, 4])
        let injected = base + MLXArray([0, 0, 3, 0] as [Float], [1, 4])
        let step = TraceMath.step(
            base: base, injected: injected, sampled: argMax(injected, axis: -1),
            counterfactual: argMax(base, axis: -1), mask: MLXArray([Int32(2)], [1]), topK: 2, full: true)
        eval(step.arrays)
        #expect(step.kl.item(Float.self) > 0.1)
        #expect(step.massIntoMask.item(Float.self) > 0.3)
        #expect(step.rankPre.item(Float.self) == 2)  // token 2 trailed 2.0 and 1.0 before the boost
        #expect(step.rankPost.item(Float.self) == 0)
        #expect(step.argmaxPre.item(Int32.self) == 1)
        #expect(step.argmaxPost.item(Int32.self) == 2)
        #expect((step.movement?.asArray(Float.self).first ?? 0) > 0.3)
    }

    @Test func identicallySeededSamplersAgreeOnEqualLogits() {
        let parameters = GenerateParameters(temperature: 0.7, seed: 99)
        let a = parameters.sampler()
        let b = parameters.sampler()
        let logits = MLXArray((0..<50).map { Float(sin(Double($0))) }, [1, 50])
        for _ in 0..<20 {
            #expect(a.sample(logits: logits).item(Int32.self) == b.sample(logits: logits).item(Int32.self))
        }
    }

    @Test func traceBufferRecordsTheCounterfactual() {
        let buffer = TraceBuffer(level: .summary, mask: MLXArray([Int32(2)], [1]), topK: 2)
        let processor = TracingLogitProcessor(
            injection: InjectionProcessor(bias: SparseBias(vocabularySize: 4, entries: [2: 3])),
            penalty: nil, buffer: buffer)
        let sampler = TracingSampler(real: ArgMaxSampler(), shadow: ArgMaxSampler(), buffer: buffer)
        let logits = MLXArray([1.0, 2.0, 0.5, -1.0] as [Float], [1, 4])
        let token = sampler.sample(logits: processor.process(logits: logits))
        #expect(token.item(Int32.self) == 2)
        let steps = buffer.drain(limit: nil)
        #expect(steps.count == 1)
        #expect(steps[0].sampled == 2)
        #expect(steps[0].counterfactual == 1)
        #expect(steps[0].kl > 0)
    }

    @Test func weightModelStartsAtZeroLearnsAndRoundTrips() throws {
        let schema = ModelSchema(featureCount: 4, contextDim: 3, projection: "test", featureNames: ["a", "b", "c", "d"])
        let model = MLXWeightModel(schema: schema, learningRate: 1e-2, weightDecay: 0, forecastWeight: 0.1)
        let features = (0..<64).map { i -> [Float] in [Float(i % 2), 0.5, Float(i % 3) / 3, 1] }
        let context = features.map { _ in [Float](repeating: 0.1, count: 3) }
        #expect(try model.predict(features: features, context: context).weights.allSatisfy { $0 == 0 })

        let targets = features.map { $0[0] > 0.5 ? Float(0.6) : Float(-0.6) }
        let batch = TrainingBatch(
            features: features, context: context, targets: targets,
            sampleWeights: Array(repeating: 1, count: 64), turnRewards: Array(repeating: 0.5, count: 64))
        let report = try model.train(batch, budget: 10, maxSteps: 200, shouldAbort: { false })
        #expect(report.steps > 0)
        #expect(model.isTrained)
        let predictions = try model.predict(features: features, context: context).weights
        #expect(predictions[1] > predictions[0])

        let url = temporaryStore().appendingPathComponent("model.safetensors")
        try model.save(to: url, metadata: [:])
        let reloaded = MLXWeightModel(schema: schema)
        _ = try reloaded.load(from: url)
        #expect(reloaded.isTrained)
        for (a, b) in zip(predictions, try reloaded.predict(features: features, context: context).weights) {
            #expect(abs(a - b) < 1e-5)
        }
        let mismatched = MLXWeightModel(schema: ModelSchema(featureCount: 5, contextDim: 3, projection: "test", featureNames: []))
        #expect(throws: SinatraError.self) { _ = try mismatched.load(from: url) }
    }

    @Test func trainingStopsWhenAGenerationIsWaiting() throws {
        let schema = ModelSchema(featureCount: 2, contextDim: 2, projection: "test", featureNames: ["a", "b"])
        let model = MLXWeightModel(schema: schema)
        let batch = TrainingBatch(
            features: Array(repeating: [1, 0], count: 30), context: Array(repeating: [0, 0], count: 30),
            targets: Array(repeating: 0.5, count: 30), sampleWeights: Array(repeating: 1, count: 30),
            turnRewards: Array(repeating: 0.5, count: 30))
        let report = try model.train(batch, budget: 10, maxSteps: 500, shouldAbort: { true })
        #expect(report.steps == 0)
        #expect(report.stoppedBy == "aborted")
    }

    @Test func encoderFindsTheEmbeddingTableByKeyPath() throws {
        let model = TinyLanguageModel(vocabulary: 12, hidden: 8)
        let encoder = try LanguageModelContextEncoder(model: model, modelKey: "tiny")
        #expect(encoder.hiddenSize == 8)
        #expect(encoder.vocabularySize == 12)
        #expect(encoder.embeddingPath == "model.embed_tokens")
        #expect(encoder.headPath == "lm_head")
        let pooled = try encoder.encode(TokenBatch(rows: [[1, 2, 3], [4]]))
        #expect(pooled.count == 2 && pooled[0].count == 8)
        let row = model.model.embedTokens.weight[4].asArray(Float.self)
        for (a, b) in zip(pooled[1], row) { #expect(abs(a - b) < 1e-5) }
        let logits = try #require(try encoder.outputLogits(pooled))
        #expect(logits.count == 2 && logits[0].count == 12)
        #expect(throws: SinatraError.self) { _ = try LanguageModelContextEncoder(model: Linear(2, 2), modelKey: "none") }
    }
}

final class TinyInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    init(vocabulary: Int, hidden: Int) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: vocabulary, dimensions: hidden)
        super.init()
    }
}

final class TinyLanguageModel: Module {
    @ModuleInfo(key: "model") var model: TinyInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(vocabulary: Int, hidden: Int) {
        self._model.wrappedValue = TinyInner(vocabulary: vocabulary, hidden: hidden)
        self._lmHead.wrappedValue = Linear(hidden, vocabulary, bias: false)
        super.init()
    }
}
