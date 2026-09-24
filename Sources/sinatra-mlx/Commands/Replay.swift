//
//  Replay.swift
//  sinatra-mlx
//
//  WHAT: Drives a transcript through the feedback loop: every user message labels the
//        previous assistant turn, every turn records its retrieved context, and training
//        runs when due. With --model the assistant turns are generated live through the
//        harness; without it the transcript's (or synthesised) answers stand in.
//

import ArgumentParser
import Foundation
import MLXLMCommon
import SinatraMLX

struct Replay: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Replay a transcript (or a synthetic one) through the implicit-feedback loop and training.")

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var store: StoreOptions
    @OptionGroup var sampling: SamplingOptions
    @OptionGroup var injection: InjectionOptions

    @Option(help: "Transcript JSON: [{at, conversationId?, user, context:[{id, documentId, text, score, createdAt?}], assistant?, assistantAt?}].")
    var transcript: String?

    @Option(help: "Synthesise this many turns instead of reading a transcript.")
    var synthesize: Int?

    @Option(help: "Seed for the synthetic transcript.")
    var synthesisSeed: UInt64 = 7

    @Flag(help: "Train when due after each turn.")
    var train = false

    @Flag(help: "Without --model, still use the MLX weight model (needs the metallib).")
    var mlxWeights = false

    @Option(help: "Write the synthetic transcript here.")
    var writeTranscript: String?

    @Flag(help: "Debug logging.")
    var verbose = false

    func run() async throws {
        var configuration = SinatraConfiguration()
        injection.apply(to: &configuration)
        let log = PrintLog(threshold: verbose ? .debug : .warning)
        let turns: [TranscriptTurn]
        if let synthesize {
            turns = Synthesizer.transcript(turns: synthesize, seed: synthesisSeed, start: Date().addingTimeInterval(-Double(synthesize) * 900))
            if let writeTranscript {
                try Transcript.write(turns, to: URL(fileURLWithPath: (writeTranscript as NSString).expandingTildeInPath))
            }
        } else if let transcript {
            turns = try Transcript.read(URL(fileURLWithPath: (transcript as NSString).expandingTildeInPath))
        } else {
            throw ValidationError("Pass --transcript or --synthesize.")
        }

        if modelOptions.model != nil || modelOptions.modelDir != nil {
            try await replayWithModel(turns, configuration: configuration, log: log)
        } else {
            try await replayWithoutModel(turns, configuration: configuration, log: log)
        }
    }

    private func replayWithoutModel(_ turns: [TranscriptTurn], configuration: SinatraConfiguration, log: SinatraLog) async throws {
        let session = SinatraSession(
            configuration: configuration, storeDirectory: store.storeURL, modelKey: "replay/no-model",
            tokenizer: SimpleTokenizer(), encoder: HashingContextEncoder(), vocabularySize: 1 << 20,
            weightModelFactory: mlxWeights ? MLXWeightModel.factory(configuration: configuration) : PriorWeightModel.factory,
            log: log)
        for (index, turn) in turns.enumerated() {
            let plan = try await session.prepareTurn(
                TurnInput(owner: store.ownerID, retrieved: turn.context, userMessage: UserMessage(text: turn.user, at: turn.at),
                          conversationId: turn.conversationId, now: turn.at))
            let answer = turn.assistant ?? Synthesizer.answer(for: turn.context, weights: plan.perPartitionWeights)
            let finished = turn.assistantAt ?? turn.at.addingTimeInterval(4 + Double(answer.count) / 200)
            let due = await session.generationDidFinish(
                owner: store.ownerID, turnId: plan.turnId, assistantText: answer,
                startedAt: turn.at.addingTimeInterval(1), finishedAt: finished)
            printTurn(index, plan: plan)
            if train && due {
                let report = try await session.train(owner: store.ownerID, now: finished)
                printTraining(report)
            }
        }
        await session.flush()
        Render.summary(await session.summary(owner: store.ownerID))
    }

    private func replayWithModel(_ turns: [TranscriptTurn], configuration: SinatraConfiguration, log: SinatraLog) async throws {
        let harness = SinatraHarness(storeDirectory: store.storeURL, configuration: configuration, log: log)
        try await modelOptions.load(harness)
        var history: [Chat.Message] = []
        for (index, turn) in turns.enumerated() {
            let userText = ContextFile.userMessage(prompt: turn.user, partitions: turn.context)
            let messages = history + [.user(userText)]
            let started = turn.at.addingTimeInterval(1)
            let request = GenerateRequest(
                input: UserInput(chat: messages), parameters: sampling.parameters,
                turn: TurnInput(owner: store.ownerID, retrieved: turn.context, userMessage: UserMessage(text: turn.user, at: turn.at),
                                conversationId: turn.conversationId, now: turn.at),
                mode: injection.mode, trace: .summary, record: true,
                timeline: .init(startedAt: started, finishedAt: turn.assistantAt ?? started.addingTimeInterval(8)))
            let handle = try await harness.generate(request)
            for await _ in handle.stream {}
            let completion = await handle.completion.value
            if let plan = handle.plan { printTurn(index, plan: plan) }
            if let summary = completion.trace?.summary {
                print("      trace: ΔH \(Render.signed(summary.meanEntropyShift)) KL \(Render.f(summary.totalKL)) gain \(Render.signed(summary.totalGain, 2)) divergence \(Render.f(summary.divergenceRate * 100, 1))%")
            }
            history = [.user(turn.user), .assistant(completion.text)]
            if train && completion.trainingScheduled {
                await harness.awaitTraining()
                if let report = await harness.lastTrainingReport { printTraining(report) }
            }
        }
        await harness.flush()
        if let summary = await harness.summary(owner: store.ownerID) { Render.summary(summary) }
    }

    private func printTurn(_ index: Int, plan: InjectionPlan) {
        let d = plan.diagnostics
        var line = "turn \(Render.pad(String(index), 3))"
        if let previous = d.labelledPrevious {
            line += " labelled prev R=\(Render.f(previous.signals.reward, 2)) \(Render.pad(previous.signals.kind.rawValue, 7))"
        } else {
            line += String(repeating: " ", count: 29)
        }
        let weights = d.partitions.map { "\($0.documentId.split(separator: "/").last.map(String.init) ?? $0.id) \(Render.signed($0.appliedWeight, 2))" }
        line += " g=\(Render.f(d.gate, 2)) bias \(Render.pad(String(d.biasNonZero), 4)) | " + weights.joined(separator: "  ")
        print(line)
    }

    private func printTraining(_ report: TrainingReport) {
        if let skipped = report.skipped {
            print("      training skipped: \(skipped)")
            return
        }
        print("      trained cycle \(report.cycle): \(report.steps) steps (\(report.stoppedBy ?? "-")), loss \(report.initialLoss.map { Render.f($0) } ?? "-") → \(report.finalLoss.map { Render.f($0) } ?? "-"), holdout MAE \(report.holdoutMAE.map { Render.f($0) } ?? "-") vs \(report.baselineMAE.map { Render.f($0) } ?? "-"), g=\(Render.f(report.reliability, 2))\(report.harmonyRan ? ", IMBHS ran" : "")\(report.periodsChanged ? " → \(report.periods.logDescription)" : "")")
    }
}
