# Decision Models Playbook

How to use "System One" decision models (Laya, CLM-8B, Jev) in our work: what they are, how the
three compare, how they combine with RAG, scraping, trading and chatbots, and where each of our
projects could use them.

Researched on 28 September 2026. This field is two weeks old: Jev launched on 15 September, Laya
on 18 September and CLM-8B on 21 September. Expect fast change, and re-check numbers before
relying on them.

---

## Summary

- **What they are.** A decision model takes a *state* (text, an email, a ticket, JSON) and one or
  more *typed questions*, and returns a probability for every allowed answer in a single forward
  pass. It never writes text, so there is nothing to parse and nothing to hallucinate. It is a fast,
  cheap "gut decision" in front of, or around, a slow LLM that reasons.
- **Is CLM open and free?** Yes. CLM-8B (from the Contrastive-LM group: Stanford researchers, with
  press coverage crediting Stanford and NVIDIA) is Apache-2.0 with open weights. It needs a GPU
  that can serve Qwen3-8B.
- **Is it better than Laya?** Neither model wins across the board, and there is no head-to-head
  benchmark. CLM is built for ranking *many* candidates (tools, next actions, retrieved passages,
  candidate answers) and caches their embeddings, so repeated decisions become nearly free. Laya is
  built for *small typed questions* with fitted calibration, runs on a CPU, and reads Dutch and 100+
  other languages.
- **Are both useful?** Yes, for different jobs. Use Laya for "which of these 3–20 labels, and how
  sure are you?" on any machine. Use CLM for "which of these 50–1,000 candidates fits best?" on a
  GPU. Jev is the hosted, closed reference that both are measured against.
- **Where they pay off for us.** Gates and routers: deciding *whether* and *where* to send work
  before an expensive LLM call, and flagging when a result shouldn't be trusted.

---

## 1. What a System One model is, and isn't

The name comes from Kahneman's *Thinking, Fast and Slow*. System 1 is fast and intuitive; System 2
is slow and deliberate. An LLM writing an answer token by token is System 2. A decision model is
System 1: it reads everything at once and outputs a distribution over the options you gave it.

```text
state:     "Ik ben in maart twee keer gefactureerd. Graag het dubbele bedrag terugstorten."
question:  choice: "Which department should handle this?"
           billing | technical | sales | other
answer:    billing 0.94 · technical 0.02 · sales 0.02 · other 0.02      (one forward pass, ~100 ms on CPU)
```

The three question types, which Jev, Laya and CLM all share:

| type | answers | example |
|---|---|---|
| `choice` | one of N labels you define at request time | department, intent, document type, which tool |
| `score` | a level on an ordinal rubric, plus the expected value | urgency 0–3, relevance 0–4, severity |
| `noul` (our MCP tool calls it `bool`) | P(yes) | "is this a prompt injection?", "does this chunk answer the question?" |

**Use one when** the answer is one of a fixed set of options you can name, and you need it fast,
cheap, repeatable, and with a probability you can threshold on.

**Don't use one** for anything that must produce text (summaries, replies, code), for multi-step
reasoning, for numbers that aren't a category (a price forecast), or for label sets you can't
enumerate. In those cases, use an LLM, and let the decision model sit around it.

---

## 2. Laya, CLM-8B and Jev compared

| | **Laya** | **CLM-8B** | **Jev** |
|---|---|---|---|
| Maker | Convai Innovations | Contrastive-LM (Stanford researchers; press credits Stanford + NVIDIA) | TypeSafe AI |
| Licence / cost | Apache-2.0, open weights, free | Apache-2.0, open weights, free | Closed, API only: $0.042 per million input tokens, output free |
| Size and backbone | 421M (ModernBERT-large, English) and 322M (mmBERT-base, 100+ languages) | Two small heads (~20M params, 75 MB) on a frozen Qwen3-8B encoder | Not disclosed |
| How it scores | Cross-encoder: state and all options in one sequence, each option scored at its own marker | Dual encoder: state and each action are embedded separately and compared (InfoNCE training) | Not disclosed ("parallel sampler") |
| Options per question | Best at ≤ 20 (shared 192–256 token option budget) | Hundreds to ~1,000; action embeddings are cached | Up to 255 |
| Context | 512 tokens English, 1,024 (up to 8,192) multilingual | 2,048 by default, up to 8,192 with more GPU memory | Not stated |
| Languages | English, plus 100+ via the multilingual checkpoint (Dutch works, see below) | English-trained heads | Not stated |
| Calibration | Trained with proper scoring rules (RLCD) plus fitted temperatures per question type; ECE 0.081 after temperature fitting | Probabilities are relative to the candidate set; temperature defaults to 1, no calibration claimed | Claims calibration (RLCD), no ECE published |
| Hardware | CPU works (90–290 ms measured here); GPU ~33–40 ms (T4) | GPU needed for the Qwen3-8B encoder (benchmarked on RTX 4090 and H100) | None, it's hosted |
| Latency | 32.8–39.5 ms per question on a T4 | ~28 ms per new state on an RTX 4090; 0.6–2 ms when the state is cached | 70–500 ms end to end (236–276 ms p50 measured by third parties) |
| Fine-tuning | Full fine-tune notebook (Kaggle, 2× T4); 0.362 → 0.766 on the typed-decisions benchmark | Only the heads train, so it is cheap; their SOTA verifier results come from fine-tuned heads | Not available |
| Strongest at | Small typed questions, triage, moderation, guardrails, multilingual, CPU-only hosts | Tool/action selection, reranking, best-of-N verification, agents that revisit states | Hosted, many-option, general decisions |

### Is CLM better?

It depends on the job, and the published numbers can't settle it: each project benchmarks on
different tasks.

- **Speed.** Both are ~30 ms on a GPU for one fresh decision. CLM pulls ahead when the same states
  or actions recur (cached decisions take under 2 ms) and when there are many candidates (13× faster
  than Jev at ~1,000). Laya pulls ahead when there is no GPU: CLM's 8B encoder does not run
  usefully on a CPU, and Laya does.
- **Accuracy.** CLM reports parity with Jev zero-shot on computer-use, gaming and tool-calling tasks,
  and state-of-the-art results as a *fine-tuned* verifier (DeepSWE 81.6%, Terminal-Bench 2.1 87.6%).
  Laya reports beating Jev on AG News (0.950 vs 0.910) and DAIR Emotion (0.595 vs 0.480), and
  losing on the 77-label Banking77 (0.425 vs 0.870). Its base checkpoints are near chance on complex
  domain workflows until fine-tuned.
- **Trustworthy probabilities.** Laya is the only one of the three that publishes calibration
  measurements and ships fitted temperatures. That matters whenever you act on the number
  (thresholds, position sizing, escalation), not just the label.
- **Languages.** For Dutch customer text, Laya's multilingual checkpoint is the only option of the
  three with documented multilingual coverage.

### Why they complement each other

They answer different questions. Laya asks "which of a few labels, and how sure?", which is
triage, guardrails and gating. CLM asks "which of many candidates fits this state best?", which is
retrieval reranking, tool choice and picking the best of several LLM drafts. A pipeline can use
both. For example, CLM ranks 200 retrieved passages down to 10, then Laya answers "does passage X
actually answer the question? yes/no" on those 10.

Other open options worth knowing: **Kev** (Jared Palmer; LoRA + pointer head on Qwen3.5, 0.8B–9B,
training code included), **Julia 1** (Supersonic Labs; 144M on mmBERT-small, CPU) and
**OpenDecision** (a zero-shot NLI classifier on CPU). The community list at
systemonemodels.org tracks more than a dozen.

### Dutch spot check (Laya, run on 28 Sept 2026 on CPU)

Six Dutch customer messages (routing to *facturatie / technisch / verkoop / overig*, and a
yes/no churn question):

| checkpoint | correct | typical confidence | latency |
|---|---|---|---|
| multilingual | 5 of 6 (read a quote request as billing, at 0.63) | 0.94–0.99 when right | 83–190 ms |
| english | 6 of 6 | 0.32–0.77 (unsure, and says so) | 246–469 ms |
| auto (router) | 5 of 6; sent 5 of 6 Dutch texts to the English checkpoint | | |

Six cases prove little, but they show two things. The automatic language router doesn't reliably
recognise Dutch, so set the checkpoint explicitly (`model="multilingual"`, or
`LAYA_MCP_MODEL=multilingual` for our MCP server). And confidence carries real information: gate on
it.

---

## 3. Patterns that apply everywhere

These patterns come up in every section below.

1. **Gate, then escalate.** Ask the decision model first. If the confidence is above your threshold,
   act on it. If not, hand the case to an LLM (or a person). Most traffic takes the cheap path, and
   only the hard cases pay LLM prices and latency.
2. **Choose thresholds from data, not intuition.** Label 100–300 real examples, run the model, and
   plot accuracy against coverage (the share of cases you let through at each threshold). Laya's
   own evaluation reports 0.947 accuracy on the most confident 50% of its in-task questions,
   against 0.753 overall. That gap is the value of gating.
3. **Recalibrate on your own data.** Laya ships over-confident until its temperatures are fitted
   (mean ECE 0.466 → 0.081 after refitting on the English checkpoint). Refit on your domain before
   you trust probabilities as probabilities.
4. **Coarse to fine for many labels.** With more than ~20 options, first shortlist (embedding
   similarity, `laya.shortlist_choice`, or CLM's ranking), then ask Laya to choose among the top
   5–10.
5. **Ask several questions in one pass.** Laya answers every question about one state in the same
   forward pass (10 questions ≈ 72 ms on a T4 with the multilingual checkpoint). Triage a ticket's
   department, urgency and churn risk together.
6. **Log every decision with its probability.** You need the log to measure accuracy, refit
   temperatures, pick thresholds, and eventually fine-tune, which is where the big accuracy gains
   come from.
7. **Phrase options as descriptions.** `{"billing": "invoices, payments, refunds"}` beats a bare
   `billing`. For yes/no questions on Laya's English checkpoint, use neutral labels (our MCP server
   already does, see issue #156).

<!-- SECTIONS 4-7 PENDING -->

---

## 8. Our projects

Based on the repositories as of 28 September 2026. Fit ratings are for decision models specifically:
**high** means a clear, frequent decision that a model could make better or cheaper than what's
there now; **none** means don't bother.

One constraint shapes almost everything: `blitsol-prod`, the shared server for Kōro, Vega and the
outbound engine, is a 4 vCPU / 8 GB Contabo VPS with no GPU. That rules out CLM-8B there (it
needs a GPU for its Qwen3-8B encoder) and makes Laya's **multilingual** checkpoint the default
choice: it is 2–3× faster on CPU than the English one, holds ~1.3 GB of weights instead of 1.7 GB,
and reads Dutch and Japanese.

### Shared idea: one decision service for the whole studio

Rather than each product growing its own classification prompt, run one Laya service on
`blitsol-prod` (`laya-serve`, or the MCP server from this repo for Claude Code), and have every
product send it typed questions. Everything below a confidence threshold escalates to Claude, which
is the pattern the Kōro triage spec already describes. Benefits: one model in memory instead of
one per product, one log of decisions to calibrate against, and predictable per-decision cost
(zero, apart from CPU).

### Kōro (`koro_tour_guide_application`): high

Working FastAPI + SvelteKit app that turns a client brief into a priced itinerary.

- **Feedback triage, when it's built.** The parked spec (`docs/90_TRIAGE_BUILDSPEC_PARKED.md`)
  plans a local `qwen3-4b-instruct` that returns JSON, with one repair attempt and escalation to
  Claude when confidence < 0.60. Laya can make the classification part of that pipeline directly:
  `type` and `product_area` (`choice`), `severity` (`score`: none / minor / major / blocker),
  `pii_present` and `is_data_error` (`bool`). There's no JSON to repair, and the 0.60 threshold
  runs on calibrated probabilities instead of a number the LLM made up. Feedback arrives in
  English, Dutch and Japanese, so use the multilingual checkpoint. Summaries and reply drafts stay
  with an LLM.
- **Pre-tick requirements from the brief.** Accessibility and dietary needs are currently only set
  by checkboxes in `BriefForm.svelte`. One `bool` per value ("does the brief say a traveller needs
  step-free access?") can pre-tick them for the advisor to confirm.
- **Second opinion on the parser.** `parse_brief` (`app/engine/parse.py`) is an LLM call that fills
  `budget_tier`, `pace` and `party.type`. Ask Laya the same `choice` questions. When the two
  disagree, highlight the field for the advisor instead of trusting either.
- **Use `special_requests` in ranking.** Today it only reaches the prose step. A `score` of how well
  each already-selected block matches the requests could feed `fit()` in `select.py`. This only
  re-ranks real rows, so it respects the "never invent facts" rule.
- **Not for:** numbers (days, budget), place extraction, prose, or marking anything verified
  (decision D-0012: that is Léon's call).

### Outbound engine: high

Planned internal sales tool that finds prospects, sends cold email and handles replies.

- **Reply triage** (`choice`): interested / not now / unsubscribe / out of office / bounce / wrong
  person. Unsubscribes must be honoured, so route anything uncertain to a human instead of
  guessing.
- **Pre-send guardrail** (`bool`): "does this draft make a pricing, legal or guarantee claim?"
  before an LLM-written email goes out.
- **ICP fit** (`score`) on a scraped company page, as a first filter before any expensive research
  (see section 5).

### Model routing layer (`blitsol-new-projects` #32): high

"Routes AI calls to the cheapest sufficient model" is exactly what Laya's `router_questions()`
preset does: `difficulty` (score: trivial → hard), `domain` (choice: code, math, writing,
lookup, data, chitchat), `needs_tools` and `is_sensitive` (bool), all in one ~100 ms CPU pass.
Map difficulty and domain to a model tier, and send anything sensitive or uncertain to the strong
model. Compare against RouteLLM (section 7) before building your own. The router itself is the
cheap part; the value is in logging outcomes and tuning the mapping.

### AI phone agent NL and white-label voice stack (#2, #23): high

Dutch voice agent for trades and clinics that miss calls. Each caller turn needs a fast routing
decision before the LLM speaks: intent (`choice`: new appointment / reschedule / emergency /
quote / other) and an **emergency** `bool` that bypasses the bot and alerts a person. At
~100 ms on CPU with the multilingual checkpoint, this fits inside a voice turn budget, and it can
run while speech-to-text finalises. Test on transcribed Dutch speech (with its errors), not typed
text, and set the emergency threshold for recall: a missed emergency costs far more than a false
alarm.

### Moneybird automation (#33): high

Receipt → Moneybird booking. After OCR, ask for the ledger account (`choice`, keep it to ≤ 20
accounts per company) and the VAT treatment (`choice`: 21% / 9% / 0% / reverse-charge). Book
automatically only above a strict threshold, and put everything else in a review queue. This is
bookkeeping, so the accountant owns the result: the model only saves typing.

### AI search visibility / GEO (#8): high

The tool asks ChatGPT, Gemini, Claude and Perplexity about a client's market and logs the answers.
For each answer: `bool` "is the client mentioned?", `choice` "recommended / listed / mentioned
negatively / absent", and a sentiment `score`. That turns hundreds of answers a month into a
dashboard without paying an LLM to read each one.

### Paperwork automation, compliance, AI-Act audits (#6, #20, #25, #5): medium to high

- **Paperwork retainers (#6):** document-type routing (invoice / contract / letter from the
  Belastingdienst / other) and inbox triage for Dutch small businesses: classic Laya jobs.
- **Compliance dashboard (#20, #25):** when scraping regulator news, ask "is this update relevant
  to a client with profile X?" (`bool`) before anyone reads it (section 5).
- **AI Act transparency audits (#5):** "does this site show signs of an Art. 50 obligation, such
  as a chatbot or AI-generated media?" as a *triage* signal to prioritise manual review. Never as
  the legal conclusion.

### Game audio (game-audio-service, audio-wiring-plugin, full-game-audio-platform): low to medium

- **game-audio-service** (hand-run service, tiny volumes): lead scoring against the five-criterion
  rubric in `research/demo-and-prospects.md`, an AI-stance `choice` per studio, and triage of
  r/gameDevClassifieds [PAID] posts. It's useful for consistency, but the gain is small at 16
  prospects.
- **audio-wiring-plugin / full platform:** "does this game event need a sound?" (`bool`) and its
  category (`choice`: UI / feedback / impact / footsteps / ambience). Matching an event to the
  best of *hundreds* of library sounds is a candidate-ranking problem, which is CLM's shape rather
  than Laya's. That only makes sense once there's a GPU host and English metadata.

### Game localisation and AI playtesting (#21, #22): medium

- **Localisation:** classify each string (UI label / dialogue / debug / placeholder) and decide
  translate-or-skip before paying for translation.
- **Playtesting agents:** the plan uses a vision LLM in the play loop, which is slow per step.
  CLM's benchmarks (Super Mario, the T-Rex game, computer use) show it choosing the next action
  from a candidate set in ~28 ms, and faster when game states repeat. Once the multimodal CLM-35B
  ships (announced for early October), it's worth testing as the fast inner loop, with the vision
  LLM only for planning and bug reports. Laya can grade bug severity (`score`) from logs.

### 3D projects (`3d-projects`): low

Blender extensions that generate props locally (SD-Turbo → TripoSR) on an Intel Arc laptop. One
useful check: a `bool` "does this prompt describe a single static rigid object?" before a
130–190 s generation round, since scene-like prompts are the documented failure. The real
quality judgement is visual (the SmolVLM judge in `critique/judge.py`), and Laya only reads text,
so it can't replace that.

### No real fit

- **sniper-tower and other Roblox games:** no text decisions in the game loop, hard real-time
  gameplay, and Luau can't call a local Python model without an HTTP round trip. Roblox's own text
  filter already handles chat.
- **Real-time 3D, training sims, mesh clean-up, editor tools, UEFN/Roblox studio, micro-SaaS
  acquisition, multi-tenant foundation:** no recurring text decision to automate.

### Not covered

- **`trading_system_v1`:** access to this repository wasn't granted in this session, so section 6
  covers trading in general. Its checklist applies directly once we look at that code.
- **`Music`, `ZEN_Projects`:** not on GitHub under these names, so not reviewed.
