//
//  SinatraCLI.swift
//  sinatra-mlx
//
//  WHAT: The harness from the command line: generate with and without the injection,
//        compare the two decodes, replay transcripts through the feedback loop, and read
//        back traces, entropy analysis and ledgers.
//

import ArgumentParser
import Foundation
import SinatraMLX

@main
struct SinatraCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sinatra-mlx",
        abstract: "On-device LLM harness with a retrieval-feedback injection layer before decoding.",
        subcommands: [
            Generate.self, Compare.self, Replay.self, TraceCommand.self, Analyze.self,
            Inspect.self, Encode.self, Features.self,
        ])
}

extension BiasMode: @retroactive ExpressibleByArgument {}
extension TraceLevel: @retroactive ExpressibleByArgument {}
