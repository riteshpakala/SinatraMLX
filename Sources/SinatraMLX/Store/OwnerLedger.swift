//
//  OwnerLedger.swift
//  SinatraMLX
//
//  WHAT: Everything the side model remembers about one owner: turns, the partitions each
//        turn retrieved (events), the reward series the indicators run over, first-seen
//        times (the index-time proxy), learned stopwords, IMBHS state and training state.
//  PIN:  Bounded by the 30-day relevancy band plus hard caps. Content-token sets live on a
//        turn only while it is pending; labelling clears them, which keeps the file small.
//

import Foundation

struct OwnerLedger: Codable {
    static let schema = 1

    var schema: Int = OwnerLedger.schema
    var owner: String
    var createdAt: Date
    var updatedAt: Date
    var turns: [TurnRecord] = []
    var events: [RetrievalEvent] = []
    var series: [SeriesPoint] = []
    var firstSeenPartitions: [String: Date] = [:]
    var firstSeenDocuments: [String: Date] = [:]
    var hotTokens: HotTokens
    var periods: IndicatorPeriods = .default
    var harmony: HarmonyMemory
    var training = TrainingState()
    var lastUserMessageAt: Date?
    var lastBiasMagnitude: Float?
    var lastTraceId: UUID?

    init(owner: OwnerID, now: Date, configuration: SinatraConfiguration) {
        self.owner = owner.rawValue
        self.createdAt = now
        self.updatedAt = now
        self.hotTokens = HotTokens(capacity: configuration.hotTokenCapacity)
        self.harmony = HarmonyMemory()
    }

    // MARK: Queries

    var labelledTurns: Int { turns.lazy.filter { $0.signals != nil }.count }
    var pendingTurns: Int { turns.lazy.filter { $0.signals == nil }.count }
    var labelledEvents: Int { events.lazy.filter { $0.label != nil }.count }

    func turnIndex(_ id: UUID) -> Int? { turns.firstIndex { $0.id == id } }

    func eventIndices(turn id: UUID) -> [Int] {
        events.indices.filter { events[$0].turnId == id }
    }

    /// The most recent labelled turn's signals.
    var lastSignals: TurnSignals? {
        turns.last(where: { $0.signals != nil })?.signals
    }
}

struct TurnRecord: Codable {
    var id: UUID
    /// When the turn was prepared (≈ the user's message arrived).
    var at: Date
    var userMessageAt: Date?
    var conversationId: String?
    var modelKey: String
    var mode: BiasMode
    var partitionIds: [String]
    var assistantStartedAt: Date?
    var assistantFinishedAt: Date?
    var assistantWords: Int?
    var assistantChars: Int?
    /// The assistant's content-token set, kept only until the turn is labelled.
    var assistantContent: [Int]?
    var signals: TurnSignals?
    var labelledAt: Date?
    var injection: InjectionAggregates?
    var trace: TraceAggregates?

    var isPending: Bool { signals == nil }
    var isComplete: Bool { assistantFinishedAt != nil }
}

struct InjectionAggregates: Codable, Equatable {
    var biasMaxAbs: Float
    var biasNonZero: Int
    var biasL1: Float
    var gate: Float
    var weightedPartitions: Int
}

struct RetrievalEvent: Codable {
    var turnId: UUID
    /// As-of time for features: the turn's prepare time.
    var at: Date
    var partitionId: String
    var documentId: String
    var rank: Int
    var staticFeatures: [Float]
    /// Count-sketch of the pooled context embedding.
    var context: [Float]?
    var contextModelKey: String?
    /// The partition's content-token set, kept only until the turn is labelled.
    var content: [Int]?
    var priorWeight: Float
    var netWeight: Float?
    var appliedWeight: Float
    var bandWeight: Float
    var label: EventLabel?
}

struct TrainingState: Codable, Equatable {
    var cycles = 0
    var lastTrainedAt: Date?
    var labelsSinceTrain = 0
    var reliability: Float = 0
    var holdoutMAE: Float?
    var baselineMAE: Float?
    var lastReport: TrainingReport?
}
