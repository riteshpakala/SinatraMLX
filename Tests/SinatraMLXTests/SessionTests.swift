import Foundation
import Testing
@testable import SinatraMLX

@Suite("Session: feedback loop, band and store")
struct SessionTests {
    let owner = OwnerID("User@Example.com")

    @Test func aReplyLabelsThePreviousTurn() async throws {
        let store = temporaryStore()
        let session = makeSession(store: store)
        let t0 = Date(timeIntervalSince1970: 1_758_000_000)
        let first = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), userMessage: UserMessage(text: "What do my notes say?", at: t0), now: t0))
        #expect(first.diagnostics.labelledPrevious == nil)
        #expect(first.bias == nil)  // cold start, nothing labelled: identity
        await session.generationDidFinish(
            owner: owner, turnId: first.turnId, assistantText: "The garden beds get morning sun and compost.",
            startedAt: t0.addingTimeInterval(1), finishedAt: t0.addingTimeInterval(5))

        let t1 = t0.addingTimeInterval(65)
        let second = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), userMessage: UserMessage(text: "Tell me more about the garden beds, compost and tomatoes please", at: t1), now: t1))
        let label = try #require(second.diagnostics.labelledPrevious)
        #expect(label.turnId == first.turnId)
        #expect(label.signals.kind == .replied)
        #expect(label.labelledEvents == 3)
        #expect(label.signals.echoMax > 0)

        let summary = await session.summary(owner: owner)
        #expect(summary.labelled == 1 && summary.pending == 1 && summary.labelledEvents == 3)
    }

    @Test func unansweredTurnsBecomeNoReplyAfterTheDeadline() async throws {
        let session = makeSession(store: temporaryStore())
        let t0 = Date(timeIntervalSince1970: 1_758_000_000)
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: t0))
        await session.generationDidFinish(owner: owner, turnId: plan.turnId, assistantText: "Answer.", startedAt: t0, finishedAt: t0.addingTimeInterval(2))
        let later = t0.addingTimeInterval(25 * 3600)
        let next = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), userMessage: UserMessage(text: "hi", at: later), now: later))
        #expect(next.diagnostics.labelledPrevious == nil)
        let summary = await session.summary(owner: owner)
        #expect(summary.labelled == 1)
        let ledger = try JSONDecoder.iso.decode(OwnerLedgerProbe.self, from: await session.ledgerJSON(owner: owner))
        #expect(ledger.series.first?.value == Double(SinatraConfiguration().noReplyReward))
    }

    @Test func theThirtyDayBandIsEnforced() async throws {
        let session = makeSession(store: temporaryStore())
        let t0 = Date(timeIntervalSince1970: 1_758_000_000)
        let old = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: t0))
        await session.generationDidFinish(owner: owner, turnId: old.turnId, assistantText: "x", startedAt: t0, finishedAt: t0)
        let later = t0.addingTimeInterval(31 * 86_400)
        _ = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), now: later))
        let summary = await session.summary(owner: owner)
        #expect(summary.labelledEvents == 0)
        let ledger = try JSONDecoder.iso.decode(OwnerLedgerProbe.self, from: await session.ledgerJSON(owner: owner))
        #expect(ledger.turns.count == 1)
    }

    @Test func dryRunsRecordNothing() async throws {
        let session = makeSession(store: temporaryStore())
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions()), dryRun: true)
        #expect(plan.dryRun)
        #expect(await session.summary(owner: owner).observations == 0)
    }

    @Test func abandonedGenerationsAreForgotten() async throws {
        let session = makeSession(store: temporaryStore())
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions()))
        await session.generationAbandoned(owner: owner, turnId: plan.turnId)
        #expect(await session.summary(owner: owner).observations == 0)
    }

    @Test func ledgersPersistAndCanBeForgotten() async throws {
        let store = temporaryStore()
        let session = makeSession(store: store)
        let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions()))
        await session.generationDidFinish(owner: owner, turnId: plan.turnId, assistantText: "x", startedAt: Date(), finishedAt: Date())
        await session.flush()
        let reopened = makeSession(store: store)
        #expect(await reopened.summary(owner: owner).observations == 1)
        #expect(await reopened.knownOwners() == [owner])
        #expect(FeatureStore.ownerKey(owner).hasPrefix("userexamplecom-"))
        try await reopened.forget(owner: owner)
        #expect(await makeSession(store: store).summary(owner: owner).observations == 0)
    }

    @Test func preferredDocumentEarnsAPositiveWeight() async throws {
        let session = makeSession(store: temporaryStore())
        var clock = Date(timeIntervalSince1970: 1_758_000_000)
        var last: InjectionPlan?
        var reply = "Where do I start?"
        for turn in 0..<24 {
            let plan = try await session.prepareTurn(TurnInput(owner: owner, retrieved: partitions(), userMessage: UserMessage(text: reply, at: clock), now: clock))
            await session.generationDidFinish(
                owner: owner, turnId: plan.turnId,
                assistantText: "Garden beds, sourdough starters and intervals: all three notes apply.",
                startedAt: clock.addingTimeInterval(1), finishedAt: clock.addingTimeInterval(5))
            // This user engages with the garden and ignores the rest.
            reply = turn.isMultiple(of: 2)
                ? "Great, more on the raised garden beds compost tomatoes basil and drip irrigation please"
                : "garden beds tomatoes compost basil sun"
            clock = clock.addingTimeInterval(45)
            last = plan
        }
        let weights = try #require(last?.perPartitionWeights)
        #expect(weights["garden#0"]! > 0.05)
        #expect(weights["garden#0"]! > weights["bread#0"]!)
        #expect(weights["garden#0"]! > weights["bike#0"]!)
        let plan = try #require(last)
        #expect(plan.bias != nil)
        #expect(plan.mask?.partitionIds == ["garden#0", "bread#0", "bike#0"])
    }
}

/// Just enough of the ledger JSON to check it from outside the store.
struct OwnerLedgerProbe: Decodable {
    struct Turn: Decodable { var id: UUID }
    var turns: [Turn]
    var series: [SeriesPoint]
}

extension JSONDecoder {
    static var iso: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return decoder
    }
}
