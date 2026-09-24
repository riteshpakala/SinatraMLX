import Foundation
import Testing
@testable import SinatraMLX

@Suite("Implicit reward")
struct RewardTests {
    let configuration = SinatraConfiguration()

    @Test func promptEngagedReply() {
        // 100 assistant words read in 30 s; replied after 60 s with 60 words; half echo.
        let s = ImplicitReward.signals(
            latency: 60, assistantWords: 100, replyWords: 60, sameConversation: true, echoMax: 0.5,
            configuration: configuration)
        #expect(s.kind == .replied)
        #expect(s.continuation == 1 && s.pace == 1 && abs(s.length - 1) < 1e-6)
        #expect(abs(s.reward - 0.85) < 1e-5)
    }

    @Test func skimmedReplyHasLowPace() {
        let s = ImplicitReward.signals(
            latency: 3, assistantWords: 100, replyWords: 2, sameConversation: true, echoMax: 0,
            configuration: configuration)
        #expect(abs(s.pace - 0.1) < 1e-6)
        #expect(s.reward < 0.5)
    }

    @Test func lateAndMissingReplies() {
        let late = ImplicitReward.signals(
            latency: 7200, assistantWords: 10, replyWords: 5, sameConversation: true, echoMax: 0,
            configuration: configuration)
        #expect(late.kind == .late && late.continuation == 0.5)
        let none = ImplicitReward.signals(
            latency: nil, assistantWords: 10, replyWords: 0, sameConversation: true, echoMax: 0,
            configuration: configuration)
        #expect(none.kind == .noReply && none.reward == configuration.noReplyReward)
        let otherConversation = ImplicitReward.signals(
            latency: 60, assistantWords: 10, replyWords: 5, sameConversation: false, echoMax: 0,
            configuration: configuration)
        #expect(otherConversation.kind == .late)
    }

    @Test func echoIsIDFWeighted() {
        let idf = ImplicitReward.idf(partitions: [[2, 3, 4], [3]])
        let echo = ImplicitReward.echo(reply: [1, 2, 3], partition: [2, 3, 4], idf: idf, partitionCount: 2)
        let expected: Float = (log(1.5) + 1 + 1) / ((log(3) + 1) + (log(1.5) + 1) + 1)
        #expect(abs(echo - expected) < 1e-5)
        #expect(ImplicitReward.echo(reply: [], partition: [1], idf: idf, partitionCount: 2) == 0)
    }

    @Test func attributionAndUsage() {
        #expect(abs(ImplicitReward.attribution(assistant: [2], partition: [2, 3, 4]) - 1.0 / 3) < 1e-6)
        #expect(ImplicitReward.rank01([0.1, 0.5, 0.5, 0.9]) == [0, 0.5, 0.5, 1])
        #expect(ImplicitReward.rank01([0.3, 0.3]) == [0.5, 0.5])
        #expect(ImplicitReward.usage(echo: [0, 0], attribution: [0, 0]) == [0.5, 0.5])
        #expect(abs(ImplicitReward.eventReward(turnReward: 0.8, usage: 1) - 0.8) < 1e-6)
        #expect(abs(ImplicitReward.eventReward(turnReward: 0.8, usage: 0) - 0.4) < 1e-6)
    }

    @Test func sampleWeightsDecayInsideTheBand() {
        #expect(ImplicitReward.sampleWeight(kindWeight: 1, ageDays: 3, configuration: configuration) == 1)
        #expect(ImplicitReward.sampleWeight(kindWeight: 1, ageDays: 10, configuration: configuration) == 0.7)
        #expect(ImplicitReward.sampleWeight(kindWeight: 1, ageDays: 31, configuration: configuration) == 0)
    }

    @Test func wordCount() {
        #expect(ImplicitReward.wordCount("Hello, world — it's 2026!") == 5)
        #expect(ImplicitReward.wordCount("") == 0)
    }
}

@Suite("Time features and relevancy bands")
struct TimeFeatureTests {
    let configuration = SinatraConfiguration()

    @Test func bandEdges() {
        #expect(TimeFeatures.band(ageDays: 7, configuration: configuration) == .fresh)
        #expect(TimeFeatures.band(ageDays: 7.01, configuration: configuration) == .mid)
        #expect(TimeFeatures.band(ageDays: 30, configuration: configuration) == .mid)
        #expect(TimeFeatures.band(ageDays: 30.01, configuration: configuration) == .stale)
        #expect(TimeFeatures.bandWeight(.stale, configuration: configuration) == 0.4)
    }

    @Test func logNormalisation() {
        #expect(TimeFeatures.lnNorm(days: 0) == 0)
        #expect(abs(TimeFeatures.lnNorm(days: 365) - 1) < 1e-6)
        #expect(TimeFeatures.lnNorm(days: 10_000) == 1)
        #expect(TimeFeatures.lnNorm(seconds: 1800, horizon: 1800) == 1)
    }

    @Test func periodicsAreUnitCircles() {
        let values = TimeFeatures.periodics(Date(timeIntervalSince1970: 1_758_000_000), timeZone: TimeZone(identifier: "UTC")!)
        #expect(values.count == 4)
        #expect(abs(values[0] * values[0] + values[1] * values[1] - 1) < 1e-5)
        #expect(abs(values[2] * values[2] + values[3] * values[3] - 1) < 1e-5)
    }
}

@Suite("Content-token filter")
struct TokenFilterTests {
    @Test func controlAndStopwordsAreNotContent() {
        let filter = TokenFilter(tokenizer: StubTokenizer())
        for id in [0, 1, 2, 5, 6, 7] { #expect(filter.isControl(id)) }
        #expect(filter.isContent(9))
        #expect(filter.isContent(12))
        #expect(!filter.isContent(8))   // "the"
        #expect(!filter.isContent(10))  // ","
        #expect(!filter.isContent(11))  // "a"
        #expect(!filter.isContent(6))
    }

    @Test func hotTokensNeedAGuaranteedShare() {
        var hot = HotTokens(capacity: 4)
        for _ in 0..<30 { hot.observe([1, 2]) }
        for i in 0..<30 { hot.observe([100 + i]) }
        #expect(hot.isHot(1, ratio: 0.5, minimumPartitions: 20))
        #expect(!hot.isHot(129, ratio: 0.5, minimumPartitions: 20))
        #expect(hot.tracked <= 4)
        #expect(!HotTokens(capacity: 4).isHot(1, ratio: 0.5, minimumPartitions: 20))
    }

    @Test func clipKeepsHeadAndTail() {
        #expect(TokenBatch.clip(Array(0..<10), maxTokens: 4) == [0, 1, 8, 9])
        #expect(TokenBatch.clip([1, 2], maxTokens: 4) == [1, 2])
        let padded = TokenBatch(rows: [[1, 2, 3], [4]]).padded()
        #expect(padded.columns == 3 && padded.ids == [1, 2, 3, 4, 0, 0] && padded.mask == [1, 1, 1, 1, 0, 0])
    }
}

@Suite("Sparse bias and impact mask")
struct SparseBiasTests {
    @Test func sparseBiasBasics() {
        let bias = SparseBias(vocabularySize: 10, entries: [5: 1.0, 2: -0.5, 9: 0.25, 12: 3])
        #expect(bias.indices == [2, 5, 9])
        #expect(bias.value(of: 5) == 1 && bias.value(of: 3) == 0)
        #expect(bias.denseVector()[2] == -0.5)
        #expect(bias.maxAbs == 1 && bias.l1 == 1.75)
        #expect(bias.top(1).first?.id == 5)
    }

    @Test func clampingKeepsAttributionConsistent() throws {
        let (bias, mask) = try #require(BiasAssembly.assemble(
            perPartition: [[3: 1.5], [3: 1.0, 4: -0.2]], partitionIds: ["a", "b"], vocabularySize: 10, cap: 2))
        #expect(bias.value(of: 3) == 2)
        let shares = mask.attribution(of: 3)
        #expect(abs((shares["a"] ?? 0) - 1.2) < 1e-5 && abs((shares["b"] ?? 0) - 0.8) < 1e-5)
        #expect(abs(mask.attribution(of: 4)["b"]! - (-0.2)) < 1e-6)
    }

    @Test func lexicalBiasFollowsWeightsAndBands() throws {
        var configuration = SinatraConfiguration()
        configuration.alpha = 1
        let input = LexicalBiasBuilder.Input(
            partitionIds: ["a"], terms: [[1: 4, 2: 1]], weights: [1], bandWeights: [1])
        let (bias, _) = try #require(LexicalBiasBuilder.build(input, vocabularySize: 10, configuration: configuration, isExcluded: { _ in false }))
        #expect(abs(bias.value(of: 1) - 1) < 1e-6 && abs(bias.value(of: 2) - 0.5) < 1e-6)

        let negative = LexicalBiasBuilder.Input(partitionIds: ["a"], terms: [[1: 4]], weights: [-0.5], bandWeights: [0.4])
        let (pushed, _) = try #require(LexicalBiasBuilder.build(negative, vocabularySize: 10, configuration: configuration, isExcluded: { _ in false }))
        #expect(abs(pushed.value(of: 1) - (-0.2)) < 1e-6)

        let idle = LexicalBiasBuilder.Input(partitionIds: ["a"], terms: [[1: 4]], weights: [0], bandWeights: [1])
        #expect(LexicalBiasBuilder.build(idle, vocabularySize: 10, configuration: configuration, isExcluded: { _ in false }) == nil)

        let excluded = LexicalBiasBuilder.build(input, vocabularySize: 10, configuration: configuration, isExcluded: { $0 == 1 })
        #expect(excluded?.0.value(of: 1) == 0)
    }

    @Test func countSketchIsDeterministicAndUnitNorm() {
        let projection = CountSketchProjection(inputDimension: 128, outputDimension: 16)
        let x = (0..<128).map { Float(sin(Double($0))) }
        let y = projection.project(x)
        #expect(y == CountSketchProjection(inputDimension: 128, outputDimension: 16).project(x))
        #expect(abs(y.reduce(0) { $0 + $1 * $1 } - 1) < 1e-3)
        #expect(projection.project([Float](repeating: 0, count: 128)).allSatisfy { $0 == 0 })
    }
}
