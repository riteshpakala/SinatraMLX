//
//  ImplicitReward.swift
//  SinatraMLX
//
//  WHAT: The implicit feedback channel. A user's next message labels the previous
//        assistant turn from behaviour alone — no LLM judge:
//          c  continuation  the session went on (30 min) or at least resumed (24 h)
//          p  pace          time taken relative to the answer's reading time
//          ℓ  length        how much the user wrote back
//          e  echo          IDF-weighted share of the reply that came from a partition
//          a  attribution   share of a partition's content the assistant actually used
//        R = 0.35 c + 0.20 p + 0.15 ℓ + 0.30 max(e); a missing reply is R = 0.25.
//        Each partition gets r_p = R · (0.5 + 0.5 u_p), u_p its usage rank in the turn.
//

import Foundation

enum ImplicitReward {

    /// Turn-level signals. `latency` nil means no reply arrived before the deadline.
    static func signals(
        latency: TimeInterval?, assistantWords: Int, replyWords: Int,
        sameConversation: Bool, echoMax: Float, configuration: SinatraConfiguration
    ) -> TurnSignals {
        guard let latency, latency >= 0, latency <= configuration.replyDeadline else {
            return TurnSignals(
                continuation: 0, pace: 0, length: 0, echoMax: 0,
                reward: configuration.noReplyReward, kind: .noReply, replyWords: 0,
                latency: latency)
        }
        let kind: ReplyKind = (latency <= configuration.sessionWindow && sameConversation) ? .replied : .late
        let continuation: Float = kind == .replied ? 1 : 0.5
        let readTime = configuration.readSecondsPerWord * Double(max(assistantWords, 1))
        let pace = Float(latency / readTime).clamped(0, 1)
        let length = Float(log1p(Double(replyWords)) / log1p(configuration.replyWordsForFullLength)).clamped(0, 1)
        let w = configuration.rewardWeights
        let reward = (w.continuation * continuation + w.pace * pace + w.length * length + w.echo * echoMax)
            .clamped(0, 1)
        return TurnSignals(
            continuation: continuation, pace: pace, length: length, echoMax: echoMax,
            reward: reward, kind: kind, replyWords: replyWords, latency: latency)
    }

    /// Smoothed IDF over the turn's partitions: ln((1 + P) / (1 + df)) + 1.
    static func idf(partitions: [Set<Int>]) -> [Int: Float] {
        var df: [Int: Int] = [:]
        for set in partitions {
            for token in set { df[token, default: 0] += 1 }
        }
        let p = Float(partitions.count)
        return df.mapValues { log((1 + p) / (1 + Float($0))) + 1 }
    }

    static func unseenIDF(partitionCount: Int) -> Float {
        log(1 + Float(partitionCount)) + 1
    }

    /// e_p: IDF-weighted share of the reply's content tokens that also occur in `partition`.
    static func echo(reply: Set<Int>, partition: Set<Int>, idf: [Int: Float], partitionCount: Int) -> Float {
        guard !reply.isEmpty else { return 0 }
        let unseen = unseenIDF(partitionCount: partitionCount)
        var total: Float = 0
        var shared: Float = 0
        for token in reply {
            let weight = idf[token] ?? unseen
            total += weight
            if partition.contains(token) { shared += weight }
        }
        return total > 0 ? (shared / total).clamped(0, 1) : 0
    }

    /// a_p: share of the partition's content tokens that appear in the assistant's answer.
    static func attribution(assistant: Set<Int>, partition: Set<Int>) -> Float {
        guard !partition.isEmpty else { return 0 }
        return Float(partition.intersection(assistant).count) / Float(partition.count)
    }

    /// Ranks normalised to [0, 1] with ties averaged; all-equal (or single) → 0.5.
    static func rank01(_ values: [Float]) -> [Float] {
        let n = values.count
        guard n > 1, let lo = values.min(), let hi = values.max(), hi > lo else {
            return Array(repeating: 0.5, count: n)
        }
        let order = values.indices.sorted { values[$0] < values[$1] }
        var ranks = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            var j = i
            while j + 1 < n && values[order[j + 1]] == values[order[i]] { j += 1 }
            let average = Float(i + j) / 2
            for k in i...j { ranks[order[k]] = average }
            i = j + 1
        }
        return ranks.map { $0 / Float(n - 1) }
    }

    /// u_p = 0.6 rank(e_p) + 0.4 rank(a_p); 0.5 when nothing was used at all.
    static func usage(echo: [Float], attribution: [Float]) -> [Float] {
        let n = echo.count
        guard n > 0 else { return [] }
        if echo.allSatisfy({ $0 == 0 }) && attribution.allSatisfy({ $0 == 0 }) {
            return Array(repeating: 0.5, count: n)
        }
        let re = rank01(echo)
        let ra = rank01(attribution)
        return (0..<n).map { 0.6 * re[$0] + 0.4 * ra[$0] }
    }

    static func eventReward(turnReward: Float, usage: Float) -> Float {
        (turnReward * (0.5 + 0.5 * usage)).clamped(0, 1)
    }

    /// Sample weight at training time: how the reply arrived × how old the event is.
    static func sampleWeight(kindWeight: Float, ageDays: Double, configuration: SinatraConfiguration) -> Float {
        if ageDays > configuration.retentionDays { return 0 }
        let decay: Float = ageDays <= configuration.freshBandDays ? 1 : configuration.midBandSampleDecay
        return kindWeight * decay
    }

    static func wordCount(_ text: String) -> Int {
        var count = 0
        var inWord = false
        for scalar in text.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                if !inWord { count += 1; inWord = true }
            } else {
                inWord = false
            }
        }
        return count
    }
}
