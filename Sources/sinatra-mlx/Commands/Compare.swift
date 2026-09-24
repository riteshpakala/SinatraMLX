//
//  Compare.swift
//  sinatra-mlx
//

import ArgumentParser
import Foundation
import SinatraMLX

struct Compare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Decode the same prompt with and without Sinatra (same seed) and score each output under both distributions.")

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var store: StoreOptions
    @OptionGroup var sampling: SamplingOptions
    @OptionGroup var injection: InjectionOptions
    @OptionGroup var prompt: PromptOptions

    @Option(help: "Trace level for both sides: summary or full.")
    var trace: TraceLevel = .summary

    @Flag(help: "Skip the teacher-forced cross-likelihood scoring.")
    var noScore = false

    @Flag(help: "Print JSON instead of tables.")
    var json = false

    @Flag(help: "Debug logging.")
    var verbose = false

    func run() async throws {
        var configuration = SinatraConfiguration()
        injection.apply(to: &configuration)
        let harness = SinatraHarness(
            storeDirectory: store.storeURL, configuration: configuration,
            log: PrintLog(threshold: verbose ? .debug : .warning))
        try await modelOptions.load(harness)
        let partitions = try prompt.partitions()
        let turn = TurnInput(
            owner: store.ownerID, retrieved: partitions,
            userMessage: UserMessage(text: prompt.prompt, at: Date()), weightOverride: injection.weights)
        let request = GenerateRequest(
            input: prompt.userInput(partitions: partitions), parameters: sampling.parameters, turn: turn,
            mode: injection.mode, trace: trace, record: false)
        let report = try await harness.compare(request, injectedMode: injection.mode == .off ? .lexical : injection.mode, score: !noScore)
        if json {
            try printJSON(report)
        } else {
            if let plan = report.plan { Render.plan(plan) }
            Render.comparison(report)
            if let summary = report.injected.summary { Render.summary(summary, title: "Injected decode, per-step impact") }
        }
    }
}
