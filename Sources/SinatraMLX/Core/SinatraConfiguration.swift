//
//  SinatraConfiguration.swift
//  SinatraMLX
//
//  WHAT: Every tunable in one place. Defaults are the v1 spec.
//

import Foundation

public struct SinatraConfiguration: Sendable {

    // MARK: Relevancy bands

    /// Events, turns and series points older than this are dropped. Training sees only the band.
    public var retentionDays: Double = 30
    /// Documents younger than this (by creation, index or first-seen time) are "fresh".
    public var freshBandDays: Double = 7
    /// Injection multiplier per band: context older than the retention window is damped, never inverted.
    public var bandWeights = BandWeights(fresh: 1.0, mid: 0.7, stale: 0.4)
    /// Sample-weight decay by event age inside the band.
    public var midBandSampleDecay: Float = 0.7

    // MARK: Caps

    public var maxEventsPerOwner = 4000
    public var maxTurnsPerOwner = 1000
    public var maxSeriesPoints = 200
    public var maxPartitionsPerTurn = 16
    public var maxTokensPerPartition = 256
    public var maxFirstSeenEntries = 20_000
    public var maxAssistantContentTokens = 1024

    // MARK: Implicit feedback

    /// A reply within this many seconds continues the session.
    public var sessionWindow: TimeInterval = 30 * 60
    /// A turn with no reply after this long is labelled "no reply".
    public var replyDeadline: TimeInterval = 24 * 60 * 60
    public var noReplyReward: Float = 0.25
    public var rewardWeights = RewardWeights(continuation: 0.35, pace: 0.20, length: 0.15, echo: 0.30)
    /// Seconds per assistant word a reader plausibly needs (200 wpm).
    public var readSecondsPerWord: Double = 0.3
    /// Reply length that saturates ℓ.
    public var replyWordsForFullLength: Double = 60
    public var kindWeights = KindWeights(replied: 1.0, late: 0.5, noReply: 0.25)

    // MARK: Weight model and training

    public var minimumLabelledEvents = 20
    public var trainEveryLabelledEvents = 4
    public var trainingBudget: TimeInterval = 0.4
    public var maxTrainingSteps = 150
    public var learningRate: Float = 2e-3
    public var weightDecay: Float = 1e-2
    public var holdoutFraction: Double = 0.2
    public var forecastLossWeight: Float = 0.25
    /// Reliability ramps from 0 at `lower` labelled events to 1 at `upper`.
    public var reliabilityRamp: ClosedRange<Double> = 20...100
    /// Holdout skill over the mean predictor that earns full reliability.
    public var reliabilitySkillForFull: Float = 0.3
    /// Indicators stay neutral until the reward series has this many points.
    public var minimumSeriesForIndicators = 20
    /// Weights are advantages: tanh(scale · (r_p − the owner's mean reward in the band)).
    /// A partition that merely co-occurs with good turns earns nothing; one that beats this
    /// user's baseline is leaned into, one that falls short is leaned away from.
    public var advantageScale: Float = 3

    // MARK: Injection

    public var biasMode: BiasMode = .lexical
    public var alpha: Float = 1.0
    /// Hard cap on |bias| in nats.
    public var cap: Float = 2.0
    public var alphaDense: Float = 0.5
    public var denseTopK = 4096
    /// |w_p| below this contributes nothing.
    public var minimumWeight: Float = 1e-3
    /// Use ledger priors while the weight model is unreliable (cold start).
    public var coldStartPrior = true
    /// Multiply each partition's term by its band weight.
    public var bandPrior = true
    /// Width of the count-sketch projection of pooled context embeddings.
    public var contextDim = 64
    /// Treat `Partition.score` as a distance (lower = better).
    public var scoreIsDistance = true

    // MARK: Vocabulary

    public var hotTokenCapacity = 512
    /// A token seen in at least this share of the owner's partitions is too common to bias.
    public var hotTokenExclusionRatio: Double = 0.5
    public var hotTokenMinimumPartitions = 20
    public var useStopwords = true

    // MARK: Trace

    public var traceLevel: TraceLevel = .automatic
    public var traceTopK = 8
    /// Full traces kept per owner.
    public var traceHistory = 20

    // MARK: Time

    public var timeZone: TimeZone = .current

    public init() {}

    public struct BandWeights: Sendable, Codable, Equatable {
        public var fresh: Float
        public var mid: Float
        public var stale: Float
        public init(fresh: Float, mid: Float, stale: Float) {
            self.fresh = fresh
            self.mid = mid
            self.stale = stale
        }
    }

    public struct RewardWeights: Sendable, Codable, Equatable {
        public var continuation: Float
        public var pace: Float
        public var length: Float
        public var echo: Float
        public init(continuation: Float, pace: Float, length: Float, echo: Float) {
            self.continuation = continuation
            self.pace = pace
            self.length = length
            self.echo = echo
        }
    }

    public struct KindWeights: Sendable, Codable, Equatable {
        public var replied: Float
        public var late: Float
        public var noReply: Float
        public init(replied: Float, late: Float, noReply: Float) {
            self.replied = replied
            self.late = late
            self.noReply = noReply
        }

        func weight(for kind: ReplyKind) -> Float {
            switch kind {
            case .replied: return replied
            case .late: return late
            case .noReply: return noReply
            }
        }
    }
}
