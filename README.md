# SinatraMLX

An on-device LLM harness for Apple silicon, built on [Frigate](https://github.com/rao-studios/Frigate)'s MLX stack, with an injection layer between the transformer and the decoder.

When a turn is answered with retrieved context, the prompt and the context go into the LLM as usual. Separately, **only the retrieved context** is encoded with the LLM's own embedding table and read by a small time-series side model. That model weighs each retrieved partition from the user's **implicit** history with similar context, and its weights become a bias added to the logits **right before sampling**. What gets decoded is tuned by what was retrieved and by how this user has responded to it before.

```
prompt + context ─▶ chat template ─▶ transformer ─▶ logits ─────────────┐
retrieved context only ─▶ embedding table ─▶ SinatraNet ─▶ bias ─▶  + ─┴▶ sampler ─▶ tokens
user behaviour over time (implicit feedback) ─▶ labels ─▶ training ─┘
```

It is the on-device successor to Sewn's server-side Sinatra. The stock-indicator process and IMBHS (Improved Music-Based Harmony Search) are carried over. The LLM sentiment judge is replaced by behaviour alone.

## The loop

Each generation with retrieved context is one **turn**:

1. **Label the previous turn.** This turn's user message is the reply to the previous assistant turn. Its timing, length and wording give that turn a reward, with no model involved.
2. **Enforce the 30-day band.** Anything older is dropped. Turns left unanswered for 24 hours are labelled "no reply".
3. **Encode the retrieved partitions**: token ids, the LLM's input-embedding rows, mean pooling, and a 64-d count sketch.
4. **Weigh each partition.** The ledger prior and SinatraNet's prediction are blended by a reliability gate.
5. **Build the injection**, a sparse bias over each partition's content tokens.
6. **Decode.** The injection runs before the repetition penalties, and a trace can record its effect step by step.
7. **Record the turn.** It waits for its reply. Training runs afterwards when due, and yields to any generation that is waiting.

### Implicit reward

Δ is the time from the end of the assistant's answer to the next user message.

| Signal | Definition |
|---|---|
| c, continuation | 1 if Δ ≤ 30 min in the same conversation, 0.5 if Δ ≤ 24 h, else no reply |
| p, pace | `clip(Δ / (0.3 s × assistant words), 0, 1)`: share of the answer plausibly read |
| ℓ, length | `clip(ln(1 + reply words) / ln 61, 0, 1)` |
| e_p, echo | IDF-weighted share of the reply's content tokens found in partition p |
| a_p, attribution | share of partition p's content tokens the assistant used |
| **R**, turn reward | `0.35c + 0.20p + 0.15ℓ + 0.30 max(e)`; a missing reply scores 0.25 |
| **r_p**, partition reward | `R × (0.5 + 0.5 u_p)`, where u_p ranks echo (0.6) and attribution (0.4) within the turn |

Weights are **advantages**, not raw rewards: `tanh(3 × (r_p − the owner's mean reward in the band))`. A partition that only co-occurs with good turns earns nothing. One that beats this user's baseline is leaned into, and one that falls short is leaned away from.

### Features (34 per partition)

The model sees 22 static features captured when the turn is prepared, 11 indicators rebuilt as of the turn, and a context-validity flag. Alongside them it gets the 64-d context sketch.

* **Document time.** Age from the creation time, falling back to index time, then to first-seen for this owner, on a log scale. Also a flag for whether a creation time was known.
* **Retrieval history.** First-seen age, never-seen-before, recency of the last retrieval, and frequency over 30 days and over 7 days.
* **Priors.** The partition's mean reward and its confidence, plus the document's mean reward.
* **Within the turn.** Rank, min–max score (distance flipped) and partition size.
* **Last turn's behaviour.** Its p, ℓ and c, and the number of pending turns.
* **When.** Time-of-day and day-of-week as periodics, and the gap since the previous user message.
* **Indicators.** Over the owner's reward series, with reply length as volume: EMA, SMA, MACD, MACD signal and previous signal, average interval change, volume-weighted average, stochastic %K and %D, momentum and velocity. IMBHS tunes the 11 windows every 5 training cycles once 20 have run.

The **30-day relevancy band** is not a model input. It multiplies each partition's injection (fresh 1.0, 7–30 d 0.7, older 0.4) and decays sample weights. As a step-function input, it made every document crossing day 7 an unseen region the net extrapolated into.

### SinatraNet and the gate

The net is `ctx 64→8 ⧺ features 34 → 32 → 16`. It has a zero-initialised tanh weight head (w_p ∈ [−1, 1]) and a sigmoid engagement-forecast head, about 2k parameters. It is trained with AdamW on the CPU device. Training is warm-started each cycle, uses input dropout, and stops early on a chronological validation split that keeps the best weights.

The gate g is the net's skill over the mean predictor on the most recent 20% of turns, ramped from 20 to 100 labelled events. The applied weight is `w_p = (1 − g) × w_prior + g × w_net`. A net that doesn't beat the baseline gets g = 0, and the priors carry the turn.

### The injection

* **`lexical`** (the default). For partition p, `m_p[t] = sqrt(tf) × idf`, max-normalised. The bias is `α Σ_p band_p × w_p × m_p[t]`, clamped to ±2 nats. Special and control tokens, punctuation and stopwords are excluded, and so are the owner's own "hot" tokens, learned with Space-Saving.
* **`dense`** (experimental). Each partition's pooled embedding goes through the model's own `lm_head`. The results are combined by weight, z-scored, and the top 4096 content tokens are kept.
* **`off`**. No injection. Turns are still recorded and labelled.

The prompt is never inspected, since `prompt(_:)` is a no-op. The cost per token is one `[1, V]` add.

### Tracing the injection

The injection is additive and constant within a turn, so its effect on every step can be attributed exactly. With tracing on, a wrapper logit processor keeps both the un-injected logits z and the injected logits z'. Penalties are applied to each separately. A wrapper sampler then draws the real token from z', and draws a **counterfactual** from z with an identically seeded shadow sampler. Per step the trace records:

* the entropy before and after
* KL(p′‖p) and Jensen–Shannon divergence
* the sampled token's log-prob and rank on both sides
* both argmaxes
* the probability mass moved into the **impact mask**, the set of tokens Sinatra biased, each attributed to the partitions that put it there
* with `full`: top-k before and after, and Δp for every mask token

Nothing syncs per step. Arrays are evaluated asynchronously and read once at the end.

`compare` decodes the same prompt with the injection off and on, using the same seed. It then scores each output under both distributions with one teacher-forced pass. The injected output's gain in likelihood is the personalization, in nats per token. `analyze` sets each traced turn's entropy shift, KL, gain and divergence against the reward that turn earned. That shows whether sharper personalization engages this user more.

## Using it

```swift
import SinatraMLX

let harness = SinatraHarness(storeDirectory: storeURL)
try await harness.load(modelID: "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit")

let handle = try await harness.generate(GenerateRequest(
    input: UserInput(chat: [.system(system), .user(promptWithContext)]),
    parameters: GenerateParameters(maxTokens: 400, temperature: 0.4, topP: 0.9),
    turn: TurnInput(
        owner: OwnerID(userId),
        retrieved: partitions,  // [Partition]: id, documentId, text, score, createdAt?
        userMessage: UserMessage(text: userText, at: receivedAt))))
for await event in handle.stream { /* .chunk / .toolCall / .info, as MLXLMCommon.generate */ }
let completion = await handle.completion.value  // trace, owner summary, training scheduled?
```

A `GenerateRequest` without a `turn` is a plain generation: no injection, and nothing recorded. The harness keeps one model resident and runs one generation at a time. Its gate also serialises encoding, trace evaluation, training and model swaps.

The store directory holds `owners/<key>/ledger.json`, `owners/<key>/model-<modelKey hash>.safetensors`, and the last 20 traces in `owners/<key>/traces/<turnId>.json`.

Sewn's `local` provider runs through this harness. [Integrating with an inference server](#integrating-with-an-inference-server) walks through how, and how to do the same in another server.

## Integrating with an inference server

SinatraMLX is built to be the on-device backend of a Swift server that already serves chat from MLX models. Sewn is the reference integration. The same pattern fits any Swift-on-Server stack on Apple silicon, such as Hummingbird, Vapor or a gRPC service, because nothing in SinatraMLX depends on a web framework.

The server keeps its routes, retrieval, prompt assembly, citations and streaming transport. SinatraMLX takes over three things:

* **Model residency.** One model is loaded, and a gate runs one generation at a time.
* **The decode.** The injection runs before sampling, with an optional trace.
* **The per-user feedback loop.** Each turn is recorded, labelled by the next message, and trained on.

### 1. Add the dependency

```swift
// Package.swift
dependencies: [
    // Frigate is the only MLX in the graph. Point it at the same place SinatraMLX does.
    .package(path: "../Frigate"),       // or its Git URL
    .package(path: "../SinatraMLX"),    // or https://github.com/riteshpakala/SinatraMLX.git
],
targets: [
    .executableTarget(
        name: "my-server",
        dependencies: [
            // Apple silicon only: condition the products so other platforms still build.
            .product(name: "SinatraMLX", package: "SinatraMLX", condition: .when(platforms: [.macOS])),
            // UserInput, GenerateParameters and Generation come from here.
            .product(name: "MLXLMCommon", package: "Frigate", condition: .when(platforms: [.macOS])),
        ]),
]
```

* **Declare Frigate yourself.** The API takes MLXLMCommon's `UserInput`, `GenerateParameters` and `Generation`, and SwiftPM only lets a target name products from packages its root declares.
* **Keep one Frigate in the graph.** Every package must point Frigate at the same location: the same path, or the same URL. Never add `ml-explore/mlx-swift`, because Frigate's targets have the same names.
* **Guard the call sites** with `#if canImport(SinatraMLX)`. Give other platforms a stub that refuses honestly; Sewn answers 503.
* **Ship the Metal library.** `swift build` never compiles MLX's shaders. After each build, run Frigate's `scripts/build-metallib.sh release --package <your package>`, and deploy `mlx.metallib` beside the installed binary (pass `--app` for an app bundle). Without it, the first GPU operation fails.

### 2. Keep one harness per process

```swift
import SinatraMLX

let harness = SinatraHarness(
    storeDirectory: dataRoot.appendingPathComponent("sinatra-mlx", isDirectory: true),
    configuration: SinatraConfiguration())   // lexical injection, 30-day band, automatic trace

// At startup, so the first turn doesn't pay for loading:
try await harness.load(modelID: "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit")

// On shutdown:
await harness.flush()
```

* **The harness is an actor.** It owns the resident model, a first-in-first-out gate, and every owner's store. Loading a different model evicts the current one under the gate, never mid-generation.
* **Report its state.** `harness.state` is `cold`, `loading(fraction)`, `ready(model)` or `failed(reason)`. That is what a health or providers endpoint should report.
* **Keep all MLX work on this model inside the harness.** Its gate also covers context encoding, trace evaluation, training and model swaps. Work that bypasses it can overlap a decode.
* **Treat the store as user data.** It sits under your data root and holds per-user files; see the checklist.
* **Models resolve through Frigate's `HubDownloader`.** It reuses a complete snapshot already on disk, in `$HF_HOME/snapshots`, `$HF_HOME` or `~/Documents/huggingface`, before downloading anything.

### 3. Map each chat turn

A chat turn with retrieved context becomes one `GenerateRequest` with a `TurnInput`:

```swift
let handle = try await harness.generate(GenerateRequest(
    input: UserInput(chat: [.system(systemPromptWithContext)] + history + [.user(userText)]),
    parameters: GenerateParameters(
        maxTokens: request.maxTokens, temperature: request.temperature, topP: request.topP,
        repetitionPenalty: request.repetitionPenalty == 1 ? nil : request.repetitionPenalty,
        seed: request.seed),
    turn: TurnInput(
        owner: OwnerID(authenticatedUserID.lowercased()),
        retrieved: hits.map {
            Partition(id: $0.chunkID, documentId: $0.documentID, text: $0.text,
                      score: $0.distance, createdAt: $0.createdAt)
        },
        userMessage: UserMessage(text: userText, at: receivedAt),
        conversationId: conversationID)))
```

| Field | What to pass | Why it matters |
|---|---|---|
| `input` | Your prompt exactly as you would send it to MLX, retrieved context included | SinatraMLX never reads or changes the prompt |
| `retrieved` | The chunks your retriever returned for this turn | The only text the side model encodes |
| `Partition.id` | A stable id per chunk | The user's history (priors, frequency, first-seen) accrues per id |
| `Partition.score` | The retriever's distance, lower = closer | For a similarity score, set `scoreIsDistance = false` in the configuration |
| `Partition.createdAt` | The document's creation time, when known | Otherwise the first time this owner retrieved it places it in the 30-day band |
| `owner` | A stable, normalised user id | Ledgers, weight models and traces are kept per owner |
| `userMessage` | The user's own words, and when they arrived | It labels the previous turn. Never pass the assembled prompt. If clients send no timestamp, use the server's receipt time |
| `conversationId` | Optional thread id | When set, a reply only labels turns from the same conversation |

* **`turn: nil`** makes a plain generation: no injection, and nothing recorded or learned. Use it for every generation that isn't a user-facing chat turn, such as summaries, compaction, titles and tool planning, and never pass a turn for hosted providers.
* **Per-request options** map onto `GenerateRequest`: `mode` (`off`, `lexical`, `dense`), `trace` (`automatic`, `off`, `summary`, `full`), and `record: false` for a dry run that plans and traces without recording the turn.
* **History** must suit the model's chat template. Mistral's rejects anything but alternating user and assistant turns after the system message, so normalise history before building `UserInput`. Sewn's `LocalMessageMapper` does this.

### 4. Stream, then report

```swift
for await event in handle.stream {
    switch event {
    case .chunk(let text):    /* write `text` to your SSE or websocket */ break
    case .toolCall(let call): /* your tool-call path */ break
    case .info:               break
    }
}
let completion = await handle.completion.value
// handle.plan?.diagnostics, completion.trace?.summary, completion.summary → one trailing metadata event
```

* **The stream carries MLXLMCommon's own `Generation` events,** so a writer built for `MLXLMCommon.generate` keeps working.
* **`completion` resolves once the turn is recorded.** It holds the text, the trace (when one was requested or an injection ran), the owner's summary, and whether training was scheduled. Sewn sends this as a trailing `sinatra` object before `[DONE]`.
* **On a client disconnect, cancel the task that consumes the stream.** The harness stops the decode and drops the unfinished turn instead of labelling a partial answer.
* **Training runs by itself** after a turn once enough new labels have arrived. It holds the gate for at most `trainingBudget` (0.4 s by default) and stops early when a generation is waiting.

### 5. Endpoints worth exposing

| Endpoint | Harness call | Sewn's route |
|---|---|---|
| Per-user status | `summary(owner:)` | the local row of `GET /v1/providers` |
| Warm a model | `load(modelID:)` | `POST /v1/providers/local/warm`, optional `{"model": …}` |
| One trace | `trace(owner:turnId:)` | `GET /v1/providers/local/sinatra/traces/{id}` |
| Entropy against reward | `entropyReport(owner:)` | `GET /v1/providers/local/sinatra/analysis` |
| Account deletion | `forget(owner:)` | the admin owner-delete route |

Scope the trace and analysis routes to the authenticated owner: traces contain decoded tokens from that user's context.

### How Sewn does it

Sewn's retrieval (Thread fan-out), prompt assembly, citations and billing are unchanged. The only change on the hot path is that on-device turns carry the retrieved partitions and the turn context into the provider.

| Piece | Where in Sewn |
|---|---|
| Dependency, product conditioned to macOS | `Package.swift` |
| The adapter actor: loading, state, request mapping, diagnostics | `Sources/Providers/Local/LocalInference.swift` |
| Turn context, sampling, the `sinatra` wire types | `Sources/Providers/Local/LocalTurn.swift` |
| Retrieved partitions with their scores, from Thread search | `Sources/Core/Models/Sewn.RetrievedPartition.swift` and `Sources/Core/Commands/Sewn+Search.swift` |
| Handing them over only when `provider` is `local` | `Sources/API/Routes/Handles/handleChatStreamCompletions.swift`, the non-streaming handler, and the realtime grounded pass |
| Status, warm, traces, analysis | `Sources/API/Routes/Providers.swift` |
| Purge on owner delete | `Sources/API/Routes/Admin.swift` |

Environment overrides (`SEWN_SINATRA_MODE`, `SEWN_SINATRA_TRACE`, `SEWN_SINATRA_ALPHA`) set the harness configuration. `sewn-probe` exercises the whole path over HTTP.

### Checklist

* One Frigate in the graph, pointed at the same place by every package.
* `mlx.metallib` beside every binary that runs MLX. That includes test bundles: install it after the build, because the bundle is code-signed and a later copy breaks the seal (see `scripts/test-mlx.sh`).
* Stable owner ids, and the user's own words and arrival time on `userMessage`.
* Retrieved chunks on `retrieved`, never the prompt.
* `turn: nil` for utility generations and for any provider other than the on-device one.
* The store is per-user data: scope reads to the owner, call `forget(owner:)` when an account is deleted, and `flush()` on shutdown.
* One decode at a time per model: concurrent requests queue on the gate.
* Injected decodes build their KV cache without quantisation. That is the iterator path MLXLMCommon offers for custom processors. Plain decodes keep it.

### Lower-level use

A server that already owns model residency and its own serialisation can use the parts directly:

* `SinatraSession` for `prepareTurn`, `generationDidFinish` and `train`, built with `LanguageModelContextEncoder`, `MLXLMTokenizer` and `MLXWeightModel`
* `InjectedGeneration.start(input:parameters:context:plan:trace:)` for the decode

It must then serialise every MLX call itself: the encoding inside `prepareTurn`, the decode, draining the trace, and training. `drainTrace` gives it step traces and their summary, but only the harness assembles the stored trace files. The harness is the supported path.

## CLI

```bash
swift build
./scripts/metallib.sh debug     # MLX's Metal shaders; swift build never compiles them

# With and without the injection, one prompt, traced step by step (--trace full adds the heatmap)
.build/debug/sinatra-mlx generate --model mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit \
  --prompt "In two sentences, how does retrieval feedback work here?" \
  --context Fixtures/corpus/recent.md:2026-09-20 --context Fixtures/corpus/old.md:2026-01-05 \
  --force-weights "1.0,-1.0" --alpha 1.5 --trace full

# Same seed, off vs on, plus cross-likelihood scoring
.build/debug/sinatra-mlx compare --model … --prompt … --context … --force-weights "1.0,-1.0"

# The feedback loop on a synthetic user who favours one document (no model needed)
.build/debug/sinatra-mlx replay --synthesize 160 --train --mlx-weights --store /tmp/sinatra
.build/debug/sinatra-mlx inspect --store /tmp/sinatra
.build/debug/sinatra-mlx analyze --store /tmp/sinatra
.build/debug/sinatra-mlx trace --store /tmp/sinatra          # the latest stored trace
.build/debug/sinatra-mlx features --context notes.md          # the 34 features, dry run
.build/debug/sinatra-mlx encode --model … --context a.md --context b.md
```

`--force-weights` sets the partition weights by hand, so the injection shows on an owner with no history.

## Tests

```bash
swift test                 # pure-Swift suites: indicators, IMBHS, rewards, bands, filter, bias, session loop
./scripts/test-mlx.sh      # everything, with the MLX suites (injection, trace math, weight model, encoder)
```

## Packaging

Frigate is the only MLX in the graph: its targets are literally named `MLX`, `MLXNN`, and so on. Never add `ml-explore/mlx-swift` here, because the names collide. `Package.swift` takes Frigate from the sibling checkout (`../../rao/repositories/Frigate`), which is the same directory Sewn uses. The committed URL form is in the comment beside it.
