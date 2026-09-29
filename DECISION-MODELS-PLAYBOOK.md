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
| Hardware | CPU works (90–290 ms measured here); GPU ~33–40 ms (T4) | GPU for the Qwen3-8B encoder (benchmarked on RTX 4090 and H100); a llama.cpp CPU path works for experiments (~1 s per 30 new tokens, cached texts in ms; see clm-mcp-server) | None, it's hosted |
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

Other open options, according to an unofficial community list at systemonemodels.org that we have
not verified: **Kev** (Jared Palmer; LoRA + pointer head on Qwen3.5, 0.8B–9B, training code
included), **Julia 1** (Supersonic Labs; 144M on mmBERT-small, CPU) and **OpenDecision** (a
zero-shot NLI classifier on CPU). The list tracks more than a dozen.

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

---

## 4. With RAG (retrieval-augmented generation)

A RAG pipeline makes several small decisions around one expensive generation step. Most of them
are decision-model shaped.

```text
question ──▶ [route]  which index / tool / none?            Laya choice
         ──▶ [gate]   does this even need retrieval?        Laya bool
         ──▶ retrieve top 50–1,000 by embedding or BM25
         ──▶ [rank]   order candidates against the question CLM rank
         ──▶ [filter] per passage: relevant? evidence?      Laya bool × 4, on the top 10–12
                      contradicts the premise? injection?
         ──▶ LLM writes the answer from what survived
         ──▶ [check]  is each claim supported by its cited passage?   Laya choice
         ──▶ low confidence anywhere ──▶ stronger model, more search, or "I don't know"
```

- **Route the query.** Adaptive-RAG (arXiv 2403.14403) trains a small classifier on query
  complexity to choose between no retrieval, one retrieval, and iterative retrieval. The same idea
  as a `choice`: `{docs, tickets, sql, web, none}`.
- **Gate retrieval.** Self-RAG (2310.11511) learns *when* to retrieve, and retrieval can hurt on
  questions about well-known facts (2212.10511). A `bool` "does answering this need our documents
  or current data?" skips the search for "rewrite this more formally".
- **Rank, then filter.** Two different jobs:
  - *Ranking* many candidates is CLM's strength: action embeddings are cached, and it is 13× faster
    than Jev at ~1,000 candidates. Its probabilities are relative to the set, though, so it always
    ranks *something* first.
  - *Filtering* needs an absolute answer ("is this passage relevant at all?"), which is a `bool` per
    passage. TypeSafe's RAG-passage cookbook asks four yes/no questions about each of the top 12
    passages and applies them in order: injection > 0.70 → drop; contradicts the question's premise
    > 0.70 → show as conflicting evidence; relevance < 0.45 → drop; answer evidence > 0.55 →
    keep. They call these thresholds "a starting point, not defaults" and stress that "nothing here
    is a security boundary". Their rerank cookbook raised top-10 recall from 38% to 62% by asking
    one yes/no per BM25 result, for $0.06 per 1,200 passages. It can't recover passages the first
    retrieval missed.
  - CRAG (2401.15884) does the same with a small evaluator that labels retrieval
    correct / incorrect / ambiguous, and falls back to web search when it's wrong.
- **Check the answer.** Before showing an answer, check each cited claim against its passage:
  `choice` supports / contradicts / doesn't mention. TypeSafe's citation cookbook first
  string-matches every quote ("a quote that is not in the source is fabricated, and no model is
  needed") and auto-accepts only at ≥ 0.8. Small checkers work well here: MiniCheck (770M)
  matches GPT-4 at fact-checking at 400× lower cost (2404.10774).
- **Cascade.** Run everything cheap first and escalate on low confidence. FrugalGPT (2305.05176)
  matched GPT-4 with up to 98% lower cost. Laya's LangChain integration has this built in:
  `LayaRouter(confidence_threshold=0.80, fallback=...)`.

---

## 5. With scraping and research agents

Scraping produces far more pages than are worth an LLM's attention. Decision models are the
triage layer between the crawler and the expensive extraction step.

- **Page triage before extraction.** Ask several questions about each page in one pass:
  `choice` page type (article / listing / login wall / error / spam), `bool` relevant to the
  research goal, `bool` contains the fields we want to extract. Only pages that pass go to the
  LLM extractor. The precedent: FineWeb-Edu labelled 450k pages with an LLM, then filtered billions
  of pages with a small classifier trained on those labels.
- **Prioritise the crawl frontier.** With the research goal as the state and the candidate links
  (with their anchor text) as actions, CLM can re-rank the frontier on every step cheaply, because
  link embeddings are cached. Craw4LLM (2502.13347) showed that prioritising the frontier reaches
  the same result with 21% of the URLs.
- **Dedup and novelty.** Hash (MinHash) and embed first. Only for near-duplicates, ask a `bool`:
  "does B add facts that A doesn't have?" Don't ask a decision model to *count* facts: TypeSafe's
  own docs say Jev can't reliably count or handle numbers and dates.
- **Know when to stop.** After each search round, ask "is the evidence now sufficient to answer
  with citations?" (`bool`), and stop at your threshold or at the budget. SIM-RAG (2505.02811)
  trains a small "sufficiency critic" for exactly this. CLM can also pick the best of several
  draft answers.
- **Source credibility.** Score each *domain* once and cache it, combined with allow and deny
  lists. Don't re-judge every page.
- **Licences and PII.** Do these deterministically first (robots.txt, licence strings, Presidio for
  PII). If Laya reads text that may contain personal data, redact it in the `on_predict_start` hook
  before tokenisation, as Laya's hook docs recommend.
- **Prompt injection in scraped pages.** A `bool` "does this text try to instruct the AI?" is a
  useful signal (TypeSafe scored a planted post at 0.99), but published guardrail classifiers were
  evaded up to 100% of the time with character tricks (2504.11168). Treat scraped text as untrusted
  whatever the score.

---

## 6. In trading

**The short version: a decision model is a feature, not a strategy.** It can label news, gate
trades and filter alerts consistently and cheaply. There is no published evidence that any System
One model predicts price direction, and there is one careful public test showing it doesn't.

### What has been tried

- **laya-trader** (open source, paper trading only). Every second it turns market indicators into a
  sentence and asks Laya "is the short-term outlook bullish?", then trades on thresholds with
  ATR-based sizing, stops and cooldowns. Its README reports, over 9 months, stocks at about +10–12%
  against buy-and-hold's +12–18%, and crypto at +6% to +26% against −8% to −12%. Over the last
  30 days (to 25 Sept), crypto made −3.7% against +13.1%, and stocks +1.4% against +3.7%. The
  crypto result could as easily come from the stops and risk rules as from the model, and the model
  is reading *numbers rendered as text*, which is exactly where these models are documented to be
  weak.
- **TradeRank's Jev test.** Jev got a masked snapshot (ten price and volume ratios plus ten daily
  returns, no ticker, no date) over 142 trading days. Its direction calls beat a drift baseline on
  66 days and lost on 60 (p = 0.66, so no evidence of skill). Its Brier score of 0.306 was worse
  than a coin flip's 0.25, and its 80% ranges held the outcome 68% of the time. Three backtest runs
  returned +17.5%, −17.4% and −2.6%. On the first live day, nothing cleared the confidence gate and
  it made no trades.
- **jev-signals** turns a headline or filing excerpt into direction, magnitude and horizon, with
  veto checks. Its own README says "it has no measured edge".
- No trading use of CLM has been published.

### What the research says about text signals

- **Sentiment is mostly short-lived.** Pessimistic WSJ columns predict falling prices *followed by
  a reversal* (Tetlock 2007), which suggests noise rather than new information.
- **LLM signals from headlines exist but decay.** They are stronger for small caps and bad news,
  stronger for bigger models (so FinBERT-sized models do worse), and "strategy returns decline as
  LLM adoption rises" (arXiv 2304.07619).
- **News is priced fast.** Positive TV reports were fully priced within about a minute, negative
  ones in about 15 (Busse & Green 2002). A 30–500 ms model plus a news feed will not be first.
  Any edge is in slower drift, in risk filtering, or in text others don't read.
- **Backtests lie in specific ways.**
  - *Lookahead:* anything before a model's training cut-off is contaminated, because the model may
    have read the outcome.
  - *Name bias:* knowing the company name distorts the reading, so mask names in tests
    (2309.17322).
  - *Survivorship:* include delisted names.
  - *Overfitting:* many backtests picked from many variants (see the probability of backtest
    overfitting and the deflated Sharpe ratio).
- **Sentiment classifiers drift** over time (2310.12620), so re-check them regularly.

### Where a decision model does fit

- **Label the text, not the price.** Classify headlines, filings and earnings-call passages into
  event types (guidance cut, M&A, lawsuit, buyback, management change) and central-bank tone
  (hawkish / neutral / dovish). Store the labels as features next to price data, and test them.
- **Meta-labelling.** Let an existing, rules-based strategy generate signals, and train or prompt a
  second model only to decide "take this trade or skip it?" and how big. This is meta-labelling
  (introduced by López de Prado), and calibration matters in it: Meyer, Barziy & Joubert (JFDS
  2023) found that calibrating the second model's probabilities significantly improved position
  sizing.
- **Risk and alert triage.** "Does this news item concern a position we hold?" (`bool`) and
  "how urgent?" (`score`), so only real events wake anyone up.

### Why calibration matters more here than anywhere

Position size should depend on the probability, and an over-confident probability oversizes every
bet. With the Kelly fraction *f\* = p − (1 − p) / b* at even odds (b = 1):

| | true p = 0.55 | model says p = 0.65 |
|---|---|---|
| Kelly bet | 10% of capital | 30% of capital |
| expected log growth per bet, given the true 0.55 | +0.005 | −0.016 (you lose money) |

So a strategy that looks right 55% of the time still loses money when its probabilities are 10
points too high. Refit temperatures on recent, out-of-sample data, and size on the calibrated
number, typically at half Kelly or less.

### A safe sequence

1. Log labels only, with no trading, for a few months. Join them to forward returns and measure
   the information coefficient and Brier score.
2. Keep only features that survive out-of-sample and after costs.
3. Paper trade, then trade small. The model never places orders; code does, within hard risk limits.

This is general research, not advice about `trading_system_v1`, whose code wasn't available in
this session.

---

## 7. In chatbots and voice agents

A chatbot turn has the same shape as a RAG query: several quick decisions around one generation.

- **Intent routing with a way out.** Up to ~20 intents fit one Laya `choice`. Always include
  `other` / out-of-scope: the CLINC150 dataset (1909.02027) exists because intent models without
  one confidently misroute everything unexpected. For 77+ intents or skills, shortlist first:
  TypeSafe's skill-suggestion cookbook cut wrong skill loads from 16.8% to 7.3% over 182 skills,
  and Laya's `predict_shortlist` and CLM's ranking do the same job.
- **Three bands, not one threshold.** TypeSafe's confidence-routing pattern: below 0.6, hand off to
  a person; between 0.6 and 0.85, ask the user to confirm; above 0.85, act. The more a wrong action
  costs (a payment versus a greeting), the higher the top band.
- **Guardrails on input and output.** Laya ships `guard_questions()` (jailbreak, prompt injection,
  sensitive data, harm severity, topic) and `moderation_questions()`, and `LayaGuardrail` can raise,
  filter or annotate inside a LangChain or LangGraph pipeline. Screen the user's message *and* the
  bot's reply. As with scraping, these are signals, not a security boundary.
- **Model routing.** Send easy turns to a small, cheap model and hard ones to a strong model.
  RouteLLM (2406.18665) cut cost by more than 2× at the same quality. Laya's `router_questions()`
  preset (difficulty, domain, needs tools, sensitive) is a ready-made starting point.
- **Keep the state short.** Jev's own docs report that accuracy drops as unrelated text piles up
  in the state. Keep a compact summary of the conversation's slots, and ask one question per slot
  each turn instead of passing the whole transcript.
- **Satisfaction and churn.** Laya's `triage_questions()` covers intent, urgency, frustration and
  churn risk. Its `score` type is the weakest, so for churn prefer a `bool` calibrated against real
  cancellations.
- **Voice latency budget.** Aim for about 800 ms from the caller finishing to the bot speaking,
  roughly 200 ms each for transport, speech-to-text, the LLM's first token, and text-to-speech
  (Kwindla Kramer, Pipecat). A local Laya on CPU (~100 ms with the multilingual checkpoint) fits
  if it runs *in parallel* with the LLM starting on the partial transcript. A hosted API at
  70–500 ms only fits in parallel, never in series.

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

---

## 9. Getting started

The step-by-step to-do list, with a research prompt for each project, is in
[DECISION-MODELS-ROLLOUT.md](DECISION-MODELS-ROLLOUT.md). CLM's MCP server is
[clm-mcp-server](clm-mcp-server/README.md).

1. **Try it in Claude Code now.** This repository's `laya-mcp-server` gives Claude Code the
   `evaluate_laya_decision` tool (see `laya-mcp-server/README.md`). For Dutch text, pass
   `model: "multilingual"`.
2. **Pick one high-fit decision** from section 8. Kōro feedback triage, outbound reply triage and
   the model routing layer are the best first candidates: they happen often, and a wrong answer is
   cheap to catch.
3. **Run it in shadow mode.** Log the model's answer and confidence next to what actually happened
   (the LLM's answer or the human's), without acting on it. Laya's docs call this staged adoption:
   shadow, compare, choose a policy, then promote a bounded slice.
4. **Label 300–2,000 real cases** from the shadow log, refit the temperatures, and choose the
   confidence threshold from the accuracy-versus-coverage curve (section 3).
5. **Turn it on for the confident slice only.** Everything below the threshold keeps the current
   path.
6. **Fine-tune when volume justifies it.** Laya's typed-decisions checkpoint went from 0.362 to
   0.766 accuracy after fine-tuning, and the notebook runs on Kaggle's free GPUs. CLM only needs its
   small heads trained.
7. **Add CLM when there's a GPU** and a real ranking problem: tool choice among many tools, a sound
   library, a crawl frontier, or best-of-N answers.

---

## Sources

Models
- Laya model card and benchmarks: <https://huggingface.co/convaiinnovations/laya>,
  <https://github.com/NandhaKishorM/laya>, docs at <https://nandhakishorm.github.io/laya/>
- CLM-8B model card and code: <https://huggingface.co/Contrastive-LM/CLM-v0.1-8B>,
  <https://github.com/Contrastive-LM/CLM>
- CLM coverage: <https://venturebeat.com/technology/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests>
- Jev: <https://typesafe.ai/blog/introducing-system-one-models-and-jev>, known weaknesses at
  <https://docs.typesafe.ai/model-jaggedness/jev-1.13.md>
- Unofficial community list (unverified): <https://systemonemodels.org/examples/alternatives/>

Patterns (TypeSafe cookbooks apply to any model with the same question types)
- RAG passage classification: <https://docs.typesafe.ai/cookbooks/classifying_rag_passages.md>
- Reranking: <https://docs.typesafe.ai/cookbooks/rerank_typesafe.md>
- Citation checking: <https://docs.typesafe.ai/cookbooks/citation_check.md>
- Confidence routing: <https://docs.typesafe.ai/patterns/confidence-routing.md>
- Guardrails: <https://docs.typesafe.ai/cookbooks/llm_guardrails.md>
- Skill shortlisting: <https://docs.typesafe.ai/cookbooks/skill_suggestion.md>
- Laya LangChain/LangGraph integration: <https://github.com/NandhaKishorM/laya/blob/main/docs/langchain.md>
- Laya staged adoption: <https://nandhakishorm.github.io/laya/staged-adoption/>

Papers (arXiv IDs): Adaptive-RAG 2403.14403 · Self-RAG 2310.11511 · When not to trust retrieval
2212.10511 · CRAG 2401.15884 · MiniCheck 2404.10774 · FrugalGPT 2305.05176 · Craw4LLM 2502.13347 ·
SIM-RAG 2505.02811 · guardrail evasion 2504.11168 · CLINC150 1909.02027 · RouteLLM 2406.18665 ·
temperature scaling 1706.04599 · conformal prediction 2107.07511 · LLM headline returns 2304.07619 ·
company-name bias 2309.17322 · sentiment drift 2310.12620

Trading
- laya-trader: <https://github.com/antonellof/laya-trader>
- TradeRank's Jev test: <https://www.traderank.ai/blog/what-is-jev-typesafe>
- jev-signals: <https://github.com/Rmanjini/jev-signals>
- Tetlock (2007), *Giving Content to Investor Sentiment*, Journal of Finance
- Busse & Green (2002), *Market efficiency in real time*, Journal of Financial Economics
- Meyer, Barziy & Joubert (2023), *Meta-Labeling: Calibration and Position Sizing*, Journal of
  Financial Data Science

Voice latency: Kwindla Kramer's voice-agent latency notes,
<https://gist.github.com/kwindla/f755284ef2b14730e1075c2ac803edcf>
