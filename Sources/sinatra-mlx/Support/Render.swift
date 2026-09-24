//
//  Render.swift
//  sinatra-mlx
//
//  WHAT: Terminal views of plans, traces, comparisons and analyses.
//

import Foundation
import SinatraMLX

enum Render {

    static func f(_ value: Float, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", value) }
    static func f(_ value: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", value) }
    static func signed(_ value: Float, _ digits: Int = 3) -> String { String(format: "%+.\(digits)f", value) }

    static func token(_ text: String?) -> String {
        guard let text else { return "·" }
        let escaped = text.replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\t", with: "\\t")
        return "\"\(escaped)\""
    }

    static func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? String(text.prefix(width)) : text + String(repeating: " ", count: width - text.count)
    }

    static func rule(_ title: String) {
        print("\n── \(title) " + String(repeating: "─", count: max(0, 72 - title.count)))
    }

    static func plan(_ d: TurnDiagnostics) {
        rule("Sinatra plan (\(d.mode.rawValue)\(d.coldStart ? ", cold start" : ""))")
        print("gate g=\(f(d.gate, 2))  labelled events \(d.labelledEvents)  labelled turns \(d.labelledTurns)  observed \(d.observedTurns)")
        print("windows \(d.periods.logDescription)")
        if let previous = d.labelledPrevious {
            let s = previous.signals
            print("this message labelled the previous turn: R=\(f(s.reward, 2)) (\(s.kind.rawValue)) c=\(f(s.continuation, 2)) p=\(f(s.pace, 2)) ℓ=\(f(s.length, 2)) echo=\(f(s.echoMax, 2)) over \(previous.labelledEvents) partitions")
        }
        print("bias: \(d.biasNonZero) tokens, max |b| \(f(d.biasMaxAbs, 2)), L1 \(f(d.biasL1, 1))  (encode \(f(d.encodeMillis, 1)) ms, build \(f(d.buildMillis, 1)) ms)")
        print("  rank  partition                         band    age(d)  prior    net      applied  labels")
        for p in d.partitions {
            print("  \(pad(String(p.rank), 4))  \(pad(p.id, 32))  \(pad(p.band, 6))  \(pad(f(p.docAgeDays, 1), 6))  \(pad(signed(p.priorWeight), 7))  \(pad(p.netWeight.map { signed($0) } ?? "   -", 7))  \(pad(signed(p.appliedWeight), 7))  \(p.labelledBefore)")
        }
        if !d.topBiased.isEmpty {
            print("  top biased: " + d.topBiased.prefix(10).map { "\(token($0.text)) \(signed($0.bias, 2))" }.joined(separator: "  "))
        }
    }

    static func summary(_ s: TraceSummary, title: String = "Trace summary") {
        rule(title)
        print("steps \(s.steps)   H \(f(s.meanEntropyPre)) → \(f(s.meanEntropyPost))  (ΔH \(signed(s.meanEntropyShift)))")
        print("KL total \(f(s.totalKL)) (mean \(f(s.meanKL, 4)))   JS mean \(f(s.meanJS, 4))   gain Σ \(signed(s.totalGain)) nats")
        print("divergence \(f(s.divergenceRate * 100, 1))% of steps (first at \(s.firstDivergenceStep.map(String.init) ?? "–"))   argmax flips \(s.flippedArgmaxSteps)")
        print("mass into mask \(signed(s.meanMassIntoMask, 4)) per step   sampled tokens inside the mask \(f(s.sampledInMaskShare * 100, 1))%")
        if !s.partitionAttribution.isEmpty {
            let parts = s.partitionAttribution.sorted { abs($0.value) > abs($1.value) }.prefix(6)
            print("attribution: " + parts.map { "\($0.key) \(signed($0.value, 2))" }.joined(separator: "  "))
        }
    }

    static func steps(_ trace: InjectionTrace, limit: Int = 80) {
        rule("Per-step impact (\(trace.level.rawValue), seed \(trace.seed.map(String.init) ?? "–"))")
        print("  step  sampled               counterfactual        H pre→post       KL       gain    rank   mask")
        for step in trace.steps.prefix(limit) {
            let cf = step.diverged ? token(step.counterfactualText) : "="
            let marker = step.inMask ? "●" : " "
            print("  \(pad(String(step.index), 4))  \(pad(token(step.sampledText), 20))  \(pad(cf, 20))  \(f(step.entropyPre, 2))→\(pad(f(step.entropyPost, 2), 6))  \(pad(f(step.kl, 4), 7))  \(pad(signed(step.gain, 2), 6))  \(pad("\(step.rankPre)→\(step.rankPost)", 6)) \(marker)\(signed(step.massIntoMask, 3))")
        }
        if trace.steps.count > limit { print("  … \(trace.steps.count - limit) more steps") }
        if trace.level == .full { topK(trace) ; heatmap(trace) }
    }

    static func topK(_ trace: InjectionTrace, steps: Int = 6) {
        rule("Top-k before → after (first \(steps) diverging steps)")
        for step in trace.steps.filter(\.diverged).prefix(steps) {
            let pre = (step.topPre ?? []).prefix(5).map { "\(token($0.text)) \(f($0.logprob, 2))" }.joined(separator: " ")
            let post = (step.topPost ?? []).prefix(5).map { "\(token($0.text)) \(f($0.logprob, 2))" }.joined(separator: " ")
            print("  step \(step.index):\n    before: \(pre)\n    after:  \(post)")
        }
    }

    /// Steps × the mask tokens whose probability moved most, one glyph per cell.
    static func heatmap(_ trace: InjectionTrace, tokens: Int = 16, steps: Int = 60) {
        guard let mask = trace.mask else { return }
        let rows = trace.steps.prefix(steps).compactMap { step in step.movement.map { (step, $0) } }
        guard !rows.isEmpty else { return }
        var totals = [Float](repeating: 0, count: mask.tokenIds.count)
        for (_, movement) in rows {
            for (i, value) in movement.enumerated() where i < totals.count { totals[i] += abs(value) }
        }
        let columns = totals.indices.sorted { totals[$0] > totals[$1] }.prefix(tokens).filter { totals[$0] > 0 }
        guard !columns.isEmpty else { return }
        rule("Where Sinatra acted: Δp of the mask tokens that moved most")
        print("  legend: '#' +10pp or more  '+' +1pp  '=' −10pp  '-' −1pp  '·' under 1pp")
        for (i, column) in columns.enumerated() {
            let text = mask.tokenTexts?[column]
            print("  col \(pad(String(i), 2)) \(pad(token(text), 16)) bias \(signed(mask.bias[column], 2))  Σ|Δp| \(f(totals[column], 3))")
        }
        let header = columns.indices.map { String($0 % 10) }.joined()
        print("  step \(header)  sampled")
        for (step, movement) in rows {
            let cells = columns.map { column -> Character in
                let value = column < movement.count ? movement[column] : 0
                switch value {
                case 0.1...: return "#"
                case 0.01..<0.1: return "+"
                case ...(-0.1): return "="
                case ...(-0.01): return "-"
                default: return "·"
                }
            }
            print("  \(pad(String(step.index), 4)) \(String(cells))  \(token(step.sampledText))\(step.diverged ? "  (was \(token(step.counterfactualText)))" : "")")
        }
    }

    static func comparison(_ report: ComparisonReport) {
        rule("Baseline (no injection)")
        print(report.baseline.text)
        rule("With Sinatra (\(report.injected.mode.rawValue))")
        print(report.injected.text)
        rule("Comparison (seed \(report.seed), temperature \(f(report.temperature, 2)))")
        print("common prefix \(report.commonPrefixTokens) tokens; first divergence at token \(report.firstDivergenceToken.map(String.init) ?? "– (identical)")")
        let hb = report.baseline.summary?.meanEntropyPost ?? 0
        let hi = report.injected.summary?.meanEntropyPost ?? 0
        print("mean entropy per step: baseline \(f(hb)) vs injected \(f(hi)) (ΔH \(signed(hi - hb)))")
        if let cross = report.crossLikelihood {
            print("log-likelihood of each output under each distribution (nats):")
            print("  baseline output: \(f(cross.baselineUnderBaseline, 2)) plain, \(f(cross.baselineUnderInjected, 2)) injected (\(signed(Float(cross.baselineShiftPerToken), 4))/token)")
            print("  injected output: \(f(cross.injectedUnderBaseline, 2)) plain, \(f(cross.injectedUnderInjected, 2)) injected (personalization \(signed(Float(cross.personalizationPerToken), 4))/token)")
        }
        print("entropy curves (per step, baseline | injected):")
        let n = max(report.baseline.entropy.count, report.injected.entropy.count)
        for i in stride(from: 0, to: min(n, 40), by: 1) {
            let b = i < report.baseline.entropy.count ? f(report.baseline.entropy[i], 2) : "  – "
            let j = i < report.injected.entropy.count ? f(report.injected.entropy[i], 2) : "  – "
            let bar = i < report.injected.entropy.count && i < report.baseline.entropy.count
                ? String(repeating: report.injected.entropy[i] < report.baseline.entropy[i] ? "<" : ">", count: min(20, Int(abs(report.injected.entropy[i] - report.baseline.entropy[i]) * 10)))
                : ""
            print("  \(pad(String(i), 4)) \(b) | \(j)  \(bar)")
        }
    }

    static func entropy(_ report: EntropyReport) {
        rule("Entropy vs personalization for \(report.owner) (\(report.rows.count) labelled, traced turns)")
        for c in report.correlations {
            print("  r(\(pad(c.metric, 15)), reward) = \(c.pearson.map { f($0, 3) } ?? "  n/a")  (n=\(c.n))")
        }
        for bin in report.bins {
            print("  \(pad(bin.label, 22)) n=\(pad(String(bin.count), 4)) mean R \(bin.meanReward.map { f($0, 3) } ?? "–")  mean ΔH \(bin.meanEntropyShift.map { f($0, 3) } ?? "–")")
        }
        if !report.mostChanged.isEmpty {
            print("  turns the injection changed most:")
            for row in report.mostChanged {
                print("    \(row.turnId.uuidString.prefix(8))  divergence \(f(row.divergenceRate * 100, 1))%  ΔH \(signed(row.entropyShift))  gain \(signed(row.gain, 2))  R \(f(row.reward, 2)) (\(row.kind.rawValue))")
            }
        }
    }

    static func summary(_ s: OwnerSummary) {
        rule("Owner \(s.owner)")
        print("observed \(s.observations) turns in the band, \(s.labelled) labelled, \(s.pending) pending, \(s.labelledEvents) labelled events")
        print("training cycles \(s.trainingCycles), last \(s.trainedAt.map { ISO8601DateFormatter().string(from: $0) } ?? "never"), reliability g=\(f(s.reliability, 2))")
        print("holdout MAE \(s.holdoutMAE.map { f($0) } ?? "–") vs mean predictor \(s.baselineMAE.map { f($0) } ?? "–")")
        print("windows \(s.periods.logDescription)")
        print("store \(s.store)")
    }
}
