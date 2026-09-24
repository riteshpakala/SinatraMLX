//
//  Generate.swift
//  sinatra-mlx
//

import ArgumentParser
import Foundation
import SinatraMLX

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Decode with the Sinatra injection beside a baseline decode, and trace its impact on the logits.")

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var store: StoreOptions
    @OptionGroup var sampling: SamplingOptions
    @OptionGroup var injection: InjectionOptions
    @OptionGroup var prompt: PromptOptions

    @Option(help: "Trace level: summary or full (full adds top-k and the movement heatmap).")
    var trace: TraceLevel = .summary

    @Flag(help: "Skip the baseline decode.")
    var noBaseline = false

    @Flag(help: "Record the turn in the owner's ledger, so a later message can label it.")
    var record = false

    @Option(help: "Trace steps to print.")
    var steps = 60

    @Flag(help: "Print JSON instead of tables.")
    var json = false

    @Flag(help: "Debug logging.")
    var verbose = false

    struct Output: Encodable {
        var baseline: String?
        var baselineSummary: TraceSummary?
        var injected: String
        var plan: TurnDiagnostics?
        var trace: InjectionTrace?
    }

    func run() async throws {
        var configuration = SinatraConfiguration()
        injection.apply(to: &configuration)
        let harness = SinatraHarness(
            storeDirectory: store.storeURL, configuration: configuration,
            log: PrintLog(threshold: verbose ? .debug : .warning))
        try await modelOptions.load(harness)

        let partitions = try prompt.partitions()
        let input = prompt.userInput(partitions: partitions)
        let turn = TurnInput(
            owner: store.ownerID, retrieved: partitions,
            userMessage: UserMessage(text: prompt.prompt, at: Date()), weightOverride: injection.weights)

        var baseline: TurnCompletion?
        if !noBaseline {
            let handle = try await harness.generate(GenerateRequest(
                input: input, parameters: sampling.parameters, turn: turn, mode: .off, trace: .summary, record: false))
            for await _ in handle.stream {}
            baseline = await handle.completion.value
            if !json {
                Render.rule("Baseline (no injection)")
                print(baseline?.text ?? "")
            }
        }

        let handle = try await harness.generate(GenerateRequest(
            input: input, parameters: sampling.parameters, turn: turn, mode: injection.mode, trace: trace, record: record))
        if !json {
            if let plan = handle.plan { Render.plan(plan.diagnostics) }
            Render.rule("With Sinatra (\(injection.mode.rawValue))")
        }
        for await item in handle.stream {
            if !json, case .chunk(let chunk) = item {
                print(chunk, terminator: "")
                fflush(stdout)
            }
        }
        let completion = await handle.completion.value
        await harness.flush()

        if json {
            try printJSON(Output(
                baseline: baseline?.text, baselineSummary: baseline?.trace?.summary, injected: completion.text,
                plan: handle.plan?.diagnostics, trace: completion.trace))
            return
        }
        print()
        if let info = completion.info {
            print(String(format: "\n%d tokens, %.1f tok/s (prompt %d tokens, %.2f s)", info.generationTokenCount, info.tokensPerSecond, info.promptTokenCount, info.promptTime))
        }
        if let trace = completion.trace {
            if let baseline = baseline?.trace?.summary {
                print("baseline mean entropy \(Render.f(baseline.meanEntropyPost)) over \(baseline.steps) steps")
            }
            Render.summary(trace.summary)
            Render.steps(trace, limit: steps)
        } else {
            print("(no trace: the plan did not inject — cold start with no labels yet; try --force-weights)")
        }
    }
}
