//
//  SinatraTypes.swift
//  SinatraMLX
//
//  WHAT: The value types that cross the harness boundary: who a turn belongs to,
//        the retrieved context, the user's message, and what came back.
//  PIN:  Only `Partition.text` / `tokenIds` ever reach the side model's encoder.
//        `UserMessage` is never encoded; its timing, length and token overlap only
//        label the PREVIOUS assistant turn — that is the implicit feedback channel.
//

import Foundation

/// Who the personalization belongs to. Sewn passes its lowercased auth user id.
public struct OwnerID: Hashable, Sendable, Codable, CustomStringConvertible, ExpressibleByStringLiteral {
    public let rawValue: String

    public init(_ rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    public var description: String { rawValue }

    public init(from decoder: Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

/// One retrieved chunk of context.
public struct Partition: Sendable, Codable, Equatable, Identifiable {
    public var id: String
    public var documentId: String
    public var text: String
    /// Pre-tokenised content without special tokens. The session tokenises `text` when nil.
    public var tokenIds: [Int]?
    /// Retrieval score. A distance by default (lower = closer), as Thread's PQ search
    /// returns it; see `SinatraConfiguration.scoreIsDistance`.
    public var score: Float
    /// Document creation time, when the caller knows it.
    public var createdAt: Date?
    /// Index time, when the caller knows it. Otherwise the first time this owner retrieved
    /// the document stands in for it.
    public var indexedAt: Date?

    public init(
        id: String, documentId: String, text: String, tokenIds: [Int]? = nil,
        score: Float = 0, createdAt: Date? = nil, indexedAt: Date? = nil
    ) {
        self.id = id
        self.documentId = documentId
        self.text = text
        self.tokenIds = tokenIds
        self.score = score
        self.createdAt = createdAt
        self.indexedAt = indexedAt
    }
}

/// The user's message for the turn being generated.
public struct UserMessage: Sendable, Codable, Equatable {
    public var text: String
    public var at: Date

    public init(text: String, at: Date) {
        self.text = text
        self.at = at
    }
}

/// Everything the side model needs to personalise one generation.
public struct TurnInput: Sendable {
    public var owner: OwnerID
    public var retrieved: [Partition]
    /// This turn's user message. It is the reply that labels the previous assistant turn.
    public var userMessage: UserMessage?
    public var conversationId: String?
    public var now: Date
    /// Debug override of the per-partition weights (CLI `--force-weights`). Aligned with
    /// `retrieved` after de-duplication; missing entries keep the computed weight.
    public var weightOverride: [Float]?

    public init(
        owner: OwnerID, retrieved: [Partition], userMessage: UserMessage? = nil,
        conversationId: String? = nil, now: Date = Date(), weightOverride: [Float]? = nil
    ) {
        self.owner = owner
        self.retrieved = retrieved
        self.userMessage = userMessage
        self.conversationId = conversationId
        self.now = now
        self.weightOverride = weightOverride
    }
}

/// How the injection is built.
public enum BiasMode: String, Sendable, Codable, CaseIterable {
    /// No injection. Turns are still recorded and labelled, so learning continues.
    case off
    /// Sparse token bias over the retrieved context's content tokens (v1 default).
    case lexical
    /// Steering vector through the model's own output head, sparsified (experimental).
    case dense
}

/// How much of the injection's effect on the logits is recorded.
public enum TraceLevel: String, Sendable, Codable, CaseIterable {
    /// `.summary` when an injection is active, otherwise `.off`.
    case automatic
    case off
    /// Per-step entropy, divergence, gain, ranks and the counterfactual token.
    case summary
    /// `.summary` plus top-k before/after and the per-step movement over the impact mask.
    case full

    func resolved(injecting: Bool) -> TraceLevel {
        switch self {
        case .automatic: return injecting ? .summary : .off
        default: return self
        }
    }
}

/// How the reply to an assistant turn arrived.
public enum ReplyKind: String, Sendable, Codable {
    /// Within the session window, same conversation.
    case replied
    /// Within the reply deadline but outside the session window (or another conversation).
    case late
    /// No reply before the deadline.
    case noReply
}

/// Implicit signals that labelled one assistant turn.
public struct TurnSignals: Sendable, Codable, Equatable {
    /// c: did the session continue.
    public var continuation: Float
    /// p: time taken relative to reading time.
    public var pace: Float
    /// ℓ: reply length.
    public var length: Float
    /// max over partitions of the reply's echo of that partition.
    public var echoMax: Float
    /// R: the turn reward in [0, 1].
    public var reward: Float
    public var kind: ReplyKind
    public var replyWords: Int
    /// Δ: seconds from the end of the assistant turn to the reply.
    public var latency: Double?
}

/// The label the reply gave one retrieved partition.
public struct EventLabel: Sendable, Codable, Equatable {
    /// r_p in [0, 1].
    public var reward: Float
    /// 2 r_p − 1, for reference. Training targets are advantages over the band mean,
    /// computed when the dataset is built.
    public var target: Float
    /// Weight from how the reply arrived (age decay is applied at training time).
    public var kindWeight: Float
    public var echo: Float
    public var attribution: Float
    public var turnReward: Float
    public var kind: ReplyKind
    public var labelledAt: Date
}

/// Labelling done for a caller that holds its own timeline (CLI replay, tests).
public struct TurnObservation: Sendable {
    public var turnId: UUID
    public var assistantText: String?
    public var assistantStartedAt: Date?
    public var assistantFinishedAt: Date?
    /// nil labels the turn as "no reply".
    public var reply: UserMessage?

    public init(
        turnId: UUID, assistantText: String? = nil, assistantStartedAt: Date? = nil,
        assistantFinishedAt: Date? = nil, reply: UserMessage?
    ) {
        self.turnId = turnId
        self.assistantText = assistantText
        self.assistantStartedAt = assistantStartedAt
        self.assistantFinishedAt = assistantFinishedAt
        self.reply = reply
    }
}

/// What labelling one turn produced.
public struct ObservationReceipt: Sendable, Codable, Equatable {
    public var turnId: UUID
    public var signals: TurnSignals
    /// partition id → r_p
    public var perPartition: [String: Float]
}

/// A token and its bias, for diagnostics.
public struct BiasedToken: Sendable, Codable, Equatable {
    public var id: Int
    public var text: String?
    public var bias: Float
}

/// Why one partition got the weight it did.
public struct PartitionDiagnostic: Sendable, Codable, Equatable {
    public var id: String
    public var documentId: String
    public var rank: Int
    public var docAgeDays: Double
    public var band: String
    public var bandWeight: Float
    public var priorWeight: Float
    public var netWeight: Float?
    public var appliedWeight: Float
    public var contentTokens: Int
    public var labelledBefore: Int
}

/// Which previous turn this turn's user message labelled.
public struct LabelDiagnostic: Sendable, Codable, Equatable {
    public var turnId: UUID
    public var signals: TurnSignals
    public var labelledEvents: Int
}

/// Everything `prepareTurn` decided, for the caller and for traces.
public struct TurnDiagnostics: Sendable, Codable, Equatable {
    public var turnId: UUID
    public var mode: BiasMode
    /// Fewer labelled events than the training minimum: weights come from priors only.
    public var coldStart: Bool
    public var labelledEvents: Int
    public var labelledTurns: Int
    public var observedTurns: Int
    public var partitions: [PartitionDiagnostic]
    public var weightedPartitions: Int
    public var biasNonZero: Int
    public var biasMaxAbs: Float
    public var biasL1: Float
    /// g: reliability of the learned weights, blended against the priors.
    public var gate: Float
    public var forecastReward: Float?
    public var periods: IndicatorPeriods
    public var topBiased: [BiasedToken]
    public var labelledPrevious: LabelDiagnostic?
    public var encodeMillis: Double
    public var buildMillis: Double
    public var dryRun: Bool
    /// The weight model's input per partition, in `FeatureVector.names` order.
    public var features: [[Float]]
}

/// The side model's decision for one generation. Plain values: safe to hand across actors.
public struct InjectionPlan: Sendable, Codable {
    public let turnId: UUID
    public let owner: OwnerID
    public let mode: BiasMode
    /// nil ⇒ identity (no injection this turn).
    public let bias: SparseBias?
    public let mask: ImpactMask?
    /// partition id → applied weight w_p
    public let perPartitionWeights: [String: Float]
    public let gate: Float
    public let diagnostics: TurnDiagnostics
    public let dryRun: Bool

    public var injects: Bool { bias != nil && mode != .off }
}

/// Where an owner's personalization stands.
public struct OwnerSummary: Sendable, Codable, Equatable {
    public var owner: String
    /// Turns observed in the relevancy band.
    public var observations: Int
    public var labelled: Int
    public var pending: Int
    public var labelledEvents: Int
    public var trainedAt: Date?
    public var trainingCycles: Int
    public var reliability: Float
    public var holdoutMAE: Float?
    public var baselineMAE: Float?
    public var periods: IndicatorPeriods
    public var lastBiasMagnitude: Float?
    public var lastTraceId: UUID?
    public var store: String
}

/// What one training cycle did.
public struct TrainingReport: Sendable, Codable, Equatable {
    public var owner: String
    public var skipped: String?
    public var rows: Int
    public var trainRows: Int
    public var holdoutRows: Int
    public var steps: Int
    public var initialLoss: Float?
    public var finalLoss: Float?
    public var holdoutMAE: Float?
    public var baselineMAE: Float?
    public var reliability: Float
    public var elapsed: TimeInterval
    public var stoppedBy: String?
    public var cycle: Int
    public var harmonyRan: Bool
    public var periodsChanged: Bool
    public var periods: IndicatorPeriods

    static func skipped(_ reason: String, owner: OwnerID, rows: Int, reliability: Float, cycle: Int, periods: IndicatorPeriods) -> TrainingReport {
        TrainingReport(
            owner: owner.rawValue, skipped: reason, rows: rows, trainRows: 0, holdoutRows: 0,
            steps: 0, initialLoss: nil, finalLoss: nil, holdoutMAE: nil, baselineMAE: nil,
            reliability: reliability, elapsed: 0, stoppedBy: nil, cycle: cycle,
            harmonyRan: false, periodsChanged: false, periods: periods)
    }
}

public enum SinatraError: Error, CustomStringConvertible, Sendable {
    case notLoaded
    case embeddingNotFound(String)
    case schemaMismatch(String)
    case unknownTurn(UUID)
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .notLoaded: return "No model is loaded."
        case .embeddingNotFound(let detail): return "No input embedding table found: \(detail)"
        case .schemaMismatch(let detail): return "Saved weight model does not match this schema: \(detail)"
        case .unknownTurn(let id): return "Unknown turn \(id)"
        case .invalidArgument(let detail): return detail
        }
    }
}
