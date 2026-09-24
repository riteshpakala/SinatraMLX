//
//  Transcript.swift
//  sinatra-mlx
//
//  WHAT: Transcript files for `replay`, and a deterministic synthetic user whose behaviour
//        (reply speed, length, echo) favours one document — to watch the loop learn it.
//

import Foundation
import SinatraMLX

struct TranscriptTurn: Codable {
    var at: Date
    var conversationId: String?
    /// The user's message; it also labels the previous assistant turn.
    var user: String
    var context: [Partition]
    var assistant: String?
    var assistantAt: Date?
}

enum Transcript {
    static func read(_ url: URL) throws -> [TranscriptTurn] {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([TranscriptTurn].self, from: Data(contentsOf: url))
    }

    static func write(_ turns: [TranscriptTurn], to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(turns).write(to: url)
    }
}

enum Synthesizer {
    static let documents: [(id: String, text: String)] = [
        ("garden", "The raised garden beds get morning sun, so tomatoes and basil thrive there. Compost from the kitchen scraps feeds the soil every spring, and drip irrigation keeps the roots evenly watered through August heat."),
        ("sourdough", "The sourdough starter needs feeding twice a day with equal weights of flour and water. A long cold proof overnight deepens the flavour and gives the crust its blistered, crackling finish."),
        ("cycling", "Interval sessions on the bike build threshold power: four blocks of eight minutes near the limit with short recoveries. Cadence around ninety keeps the legs fresh on long climbs."),
        ("budget", "The monthly budget splits income into fixed costs, savings and discretionary spending. Automating the transfer to savings on payday removes the temptation to skip it."),
        ("piano", "Scales in contrary motion strengthen finger independence at the piano. Slow practice with a metronome, then gradual tempo increases, makes difficult passages reliable."),
        ("travel", "Packing cubes keep a carry-on organised for a two week trip. Rolling clothes saves space, and a single pair of versatile shoes avoids overpacking."),
    ]

    static let preferred = "garden"

    static func transcript(turns count: Int, seed: UInt64, start: Date) -> [TranscriptTurn] {
        var rng = SplitMix64(seed: seed)
        var turns: [TranscriptTurn] = []
        var clock = start
        var conversation = 1
        var previousHadPreferred = false
        for index in 0..<count {
            var chosen = Set<String>()
            if Double.random(in: 0..<1, using: &rng) < 0.6 { chosen.insert(preferred) }
            while chosen.count < 3 { chosen.insert(documents[Int.random(in: 0..<documents.count, using: &rng)].id) }
            let context = chosen.sorted().enumerated().map { rank, id -> Partition in
                let text = documents.first { $0.id == id }!.text
                return Partition(id: "\(id)#0", documentId: "docs/\(id)", text: text, score: Float(rank) * 0.1)
            }
            let user: String
            if index == 0 {
                user = "Tell me what my notes say."
            } else if previousHadPreferred {
                let words = documents.first { $0.id == preferred }!.text.split(separator: " ").shuffled(using: &rng).prefix(12)
                user = "That helps a lot. More about " + words.joined(separator: " ") + " — what should I do next week?"
            } else {
                user = ["ok", "sure", "hm, fine", "next"][Int.random(in: 0..<4, using: &rng)]
            }
            turns.append(TranscriptTurn(
                at: clock, conversationId: "c\(conversation)", user: user, context: context,
                assistant: nil, assistantAt: nil))
            previousHadPreferred = chosen.contains(preferred)
            let answerTime = 6.0
            let latency = previousHadPreferred
                ? Double.random(in: 25...70, using: &rng)
                : Double.random(in: 300...1500, using: &rng)
            clock = clock.addingTimeInterval(answerTime + latency)
            if Double.random(in: 0..<1, using: &rng) < 0.1 {
                clock = clock.addingTimeInterval(Double.random(in: 7200...86_000, using: &rng))
                conversation += 1
            }
        }
        return turns
    }

    /// A stand-in answer: the first sentence of each partition, strongest weight first.
    static func answer(for context: [Partition], weights: [String: Float]) -> String {
        context.sorted { (weights[$0.id] ?? 0) > (weights[$1.id] ?? 0) }
            .compactMap { $0.text.split(separator: ".").first.map { String($0) + "." } }
            .joined(separator: " ")
    }
}
