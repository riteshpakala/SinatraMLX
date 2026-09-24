//
//  EntropyAnalysis.swift
//  SinatraMLX
//
//  WHAT: Entropy against personalization. For every labelled turn that carried a trace,
//        set what the injection did to decoding (entropy shift, KL, gain, divergence)
//        against the implicit reward the user's behaviour gave that turn. Answers "does
//        sharper personalization — lower entropy, more gain — engage this user more?"
//

import Foundation

public struct EntropyReport: Codable, Sendable {
    public struct Row: Codable, Sendable, Equatable {
        public var turnId: UUID
        public var at: Date
        public var mode: BiasMode
        public var reward: Float
        public var kind: ReplyKind
        public var entropyPre: Float
        public var entropyPost: Float
        public var entropyShift: Float
        public var kl: Float
        public var gain: Float
        public var divergenceRate: Float
        public var massIntoMask: Float
    }

    public struct Correlation: Codable, Sendable, Equatable {
        public var metric: String
        /// Pearson r with the turn reward; nil when undefined (n < 3 or no variance).
        public var pearson: Double?
        public var n: Int
    }

    public struct Bin: Codable, Sendable, Equatable {
        public var label: String
        public var count: Int
        public var meanReward: Double?
        public var meanEntropyShift: Double?
    }

    public var owner: String
    public var rows: [Row]
    public var correlations: [Correlation]
    public var bins: [Bin]
    /// The turns the injection changed most (highest divergence rate).
    public var mostChanged: [Row]
}

enum EntropyAnalysis {

    static func report(owner: OwnerID, ledger: OwnerLedger) -> EntropyReport {
        let rows: [EntropyReport.Row] = ledger.turns.compactMap { turn in
            guard let signals = turn.signals, let trace = turn.trace else { return nil }
            return EntropyReport.Row(
                turnId: turn.id, at: turn.at, mode: turn.mode, reward: signals.reward,
                kind: signals.kind, entropyPre: trace.meanEntropyPre, entropyPost: trace.meanEntropyPost,
                entropyShift: trace.entropyShift, kl: trace.kl, gain: trace.gain,
                divergenceRate: trace.divergenceRate, massIntoMask: trace.massIntoMask)
        }
        let reward = rows.map { Double($0.reward) }
        let metrics: [(String, (EntropyReport.Row) -> Float)] = [
            ("entropyShift", \.entropyShift), ("entropyPost", \.entropyPost), ("kl", \.kl),
            ("gain", \.gain), ("divergenceRate", \.divergenceRate), ("massIntoMask", \.massIntoMask),
        ]
        let correlations = metrics.map { name, value in
            EntropyReport.Correlation(
                metric: name, pearson: pearson(rows.map { Double(value($0)) }, reward), n: rows.count)
        }
        func bin(_ label: String, _ members: [EntropyReport.Row]) -> EntropyReport.Bin {
            EntropyReport.Bin(
                label: label, count: members.count,
                meanReward: mean(members.map { Double($0.reward) }),
                meanEntropyShift: mean(members.map { Double($0.entropyShift) }))
        }
        let bins = [
            bin("sharpened (ΔH < 0)", rows.filter { $0.entropyShift < 0 }),
            bin("flattened (ΔH ≥ 0)", rows.filter { $0.entropyShift >= 0 }),
            bin("diverged (rate > 0)", rows.filter { $0.divergenceRate > 0 }),
            bin("unchanged (rate = 0)", rows.filter { $0.divergenceRate == 0 }),
        ]
        let mostChanged = Array(rows.sorted { $0.divergenceRate > $1.divergenceRate }.prefix(5))
        return EntropyReport(
            owner: owner.rawValue, rows: rows, correlations: correlations, bins: bins,
            mostChanged: mostChanged)
    }

    static func mean(_ values: [Double]) -> Double? {
        values.isEmpty ? nil : values.reduce(0, +) / Double(values.count)
    }

    static func pearson(_ x: [Double], _ y: [Double]) -> Double? {
        let n = min(x.count, y.count)
        guard n >= 3 else { return nil }
        let mx = x.prefix(n).reduce(0, +) / Double(n)
        let my = y.prefix(n).reduce(0, +) / Double(n)
        var sxy = 0.0, sxx = 0.0, syy = 0.0
        for i in 0..<n {
            let dx = x[i] - mx
            let dy = y[i] - my
            sxy += dx * dy
            sxx += dx * dx
            syy += dy * dy
        }
        guard sxx > 0, syy > 0 else { return nil }
        return sxy / (sxx * syy).squareRoot()
    }
}
