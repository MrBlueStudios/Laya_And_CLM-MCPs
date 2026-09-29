# Decision Models Rollout

The to-do list for bringing Laya and CLM into our projects. It has four phases:

- **Phase 0:** make both MCP servers available everywhere.
- **Phase 1:** research each project from inside that project with Claude Code.
- **Phase 2:** improve the MCP servers based on what the research found.
- **Phase 3:** build it into each project.

Background for all of it: [DECISION-MODELS-PLAYBOOK.md](DECISION-MODELS-PLAYBOOK.md) (what the
models are and the patterns), [laya-mcp-server](laya-mcp-server/README.md) and
[clm-mcp-server](clm-mcp-server/README.md).

Tick boxes as you go (`- [x]`). Each project's findings go into a `docs/decision-models.md` file
*in that project's repository*; this file only tracks status.

---

## Phase 0: set up once

- [ ] Clone this repo to `~/Projects/Laya_And_CLM-MCPs` on the machine where you run Claude Code
      (as a submodule of the `Projects` repo; see `docs/projects-repo-prompt.md`), then run
      `./register-decision-mcps.sh`. Re-run it whenever the folder moves: the registration
      stores absolute paths. It registers both servers at **user scope**, so they work in
      every project, not only here. Check with `claude mcp list` from any other directory.
- [ ] Download Laya's model once:
      `uv run --project laya-mcp-server --frozen laya-decision-mcp --prefetch` (~0.8 GB).
      Add the multilingual one for Dutch projects:
      `LAYA_MCP_PRELOAD=multilingual uv run --project laya-mcp-server --frozen laya-decision-mcp --prefetch`.
- [ ] Decide where CLM runs. It needs Qwen3-8B embeddings:
  - [ ] **For research:** the CPU backend (option C in the clm-mcp-server README) on a laptop
        with ≥ 16 GB RAM. It's slow on new texts (~1 s per 30 tokens) but fine for dozens of
        samples, and cached texts are instant.
  - [ ] **For production, only if the research shows CLM is worth it:** a GPU host (a rented GPU
        or a desktop with a ≥ 16 GB card) behind an SSH tunnel. `blitsol-prod` has no GPU and
        8 GB of RAM, so it can't host CLM.
- [ ] Set `CLM_BASE_URL` / `CLM_API_KEY` in your shell profile, then check with
      `uv run --project clm-mcp-server --frozen clm-decision-mcp --check`.
- [ ] Open Claude Code in any project and run `/mcp`: both `laya-mcp-server` and `clm-mcp-server`
      should show as connected.

---

## Phase 1: research each project

### How to run a research session

1. `cd` into the project's repository and start `claude`.
2. Paste the prompt below, with the project's line from the table filled in.
3. Review `docs/decision-models.md` in that repo when it's done, then tick the project off here.

````text
Research how the Laya and CLM decision models (MCP tools evaluate_laya_decision,
evaluate_clm_decision, rank_clm_candidates) can help THIS project. Background:
~/Projects/Laya_And_CLM-MCPs/DECISION-MODELS-PLAYBOOK.md (section 8 has a first guess for
this project) and the READMEs of ~/Projects/Laya_And_CLM-MCPs/laya-mcp-server and
~/Projects/Laya_And_CLM-MCPs/clm-mcp-server.

Starting points for this project: <paste the "decisions to test" cell from the rollout table>

1. Read the code and docs, and list every place the project makes (or plans) a decision with a
   fixed set of answers: routing, triage, tagging, yes/no checks, scoring, picking one of many.
   Note where it happens (file, function), how it's made now (rules, an LLM prompt, a person),
   how often, and what a wrong answer costs.
2. For the 2–4 most promising decisions, build a test set of 30–100 realistic inputs with the
   answer you'd expect. Use real data where it exists; never put personal data in the test set.
3. Run the test set through the tools: Laya (the checkpoint that fits the language; multilingual
   for Dutch) and CLM (plain-sentence options; rank_clm_candidates when there are many
   candidates). Try 2–3 phrasings of the question and options. Record accuracy, the confidence of
   right versus wrong answers, and latency.
4. For each decision, pick a design: model, question and option wording, confidence threshold
   (from the accuracy-versus-coverage of your test set), what happens below the threshold
   (LLM / human / default), and where it runs (blitsol-prod CPU, GPU host, the user's machine).
5. Write docs/decision-models.md with these sections: Decisions found, Test sets, Results
   (tables), Chosen design, Implementation plan (files to change, hosting, rollout in shadow mode
   first), MCP improvements that would help, Open questions. Commit it on a new branch.
Do not change product code yet; this is research only.
````

### Projects, in priority order

"Decisions to test" comes from the playbook's first pass. The research should confirm, change or
drop each of them.

| # | status | project (repo) | decisions to test | notes |
|---|---|---|---|---|
| 1 | [ ] | **Kōro** (`MrBlueStudios/Koro_tour_guide_application`) | feedback triage (`type`, `product_area`, `severity`, `pii_present`) for the parked `classify.py`; pre-ticking accessibility/dietary needs from the brief; a second opinion on `parse_brief`'s `budget_tier` / `pace` / `party.type`; `special_requests` as a ranking signal in `select.py` | Working code with 360 tests. Feedback arrives in EN/NL/JA, so use multilingual. The triage spec already plans a confidence < 0.60 → Claude escalation. |
| 2 | [ ] | **Outbound engine** (`blitsol-new-projects/outbound-engine`) | reply triage (interested / not now / unsubscribe / out of office / bounce / wrong person); pre-send guardrail ("makes a pricing, legal or guarantee claim?"); ICP fit score on a scraped company page | Unsubscribes must never be missed: set that class's threshold for recall. |
| 3 | [ ] | **Model routing layer** (`blitsol-new-projects/model-routing-layer`) | Laya's `router_questions()` preset (difficulty, domain, needs_tools, is_sensitive); CLM ranking model descriptions against the request | Compare with RouteLLM and OpenRouter's router before building. Needs logged outcomes to tune. |
| 4 | [ ] | **AI phone agent NL / white-label voice** (`ai-phone-agent-nl`, `white-label-voice-stack`) | per-turn intent (appointment / reschedule / emergency / quote / other); emergency `bool` for escalation | Test on transcribed Dutch speech, not typed text. Measure latency inside an ~800 ms voice budget. Set the emergency threshold for recall. |
| 5 | [ ] | **Moneybird automation** (`moneybird-automation`) | ledger account `choice` (≤ 20 per company) after OCR; VAT treatment (21% / 9% / 0% / reverse-charge); "matches an existing booking?" | Bookkeeping: auto-book only above a strict threshold, everything else goes to review. |
| 6 | [ ] | **AI search visibility / GEO** (`ai-search-visibility-geo`) | per AI answer: client mentioned? recommended / listed / negative / absent; sentiment | High volume, cheap decisions. A good first production use. |
| 7 | [ ] | **Paperwork, compliance, AI Act** (`ai-automation-retainers`, `compliance-dashboard-nl`, `dutch-compliance-platform`, `ai-act-transparency-audits`) | document-type routing; inbox triage; "is this regulatory update relevant to client profile X?"; Art. 50 triage signals | Dutch text. AI Act results are triage for a person, never a legal conclusion. |
| 8 | [ ] | **Game audio** (`game-audio-service`, `audio-wiring-plugin`, `full-game-audio-platform`) | lead scoring against the rubric in `research/demo-and-prospects.md`; [PAID] post triage; "needs a sound?" and category per game event; CLM ranking events against a sound library | Small volumes today. The library match is CLM's shape: check it on the CPU backend first. |
| 9 | [ ] | **Localisation and playtesting** (`game-localization-automation`, `ai-playtesting-agents`) | string type and translate-or-skip; bug severity from logs; CLM choosing the next action from candidates | Re-check playtesting once the multimodal CLM-35B is released (announced for early October). |
| 10 | [ ] | **3D projects** (`MrBlueStudios/3d-projects`) | "is this prompt a single static rigid object?" before a 130–190 s generation run; template choice | Low fit. The main quality check is visual, which these models can't read. |
| 11 | [ ] | **Trading** (`MrBlueStudios/trading_system_v1`) | news/filing event labels as features; meta-labelling (take or skip a signal); alert triage | Not reviewed yet: access wasn't granted in the first session. Follow playbook section 6: log labels and measure against forward returns before any money is involved. |
| 12 | [ ] | **sniper-tower and Roblox games**, `Music`, `ZEN_Projects` | confirm there's nothing to do, or find a text decision the first pass missed | Expected: no fit (Roblox has no text decisions in the game loop). |

---

## Phase 2: improve the MCP servers from the findings

Collect the "MCP improvements that would help" sections from every project's research, then pick
from this list and anything new:

- [ ] **Several questions per call.** Laya answers all questions about one state in one forward
      pass (10 questions ≈ 72 ms on a T4). Add a multi-question tool if projects ask 2+ questions
      per item.
- [ ] **Presets per project**, e.g. `kōro_feedback_triage`, so the wording and thresholds found in
      research live in one place instead of every prompt.
- [ ] **Decision log.** Optionally write every decision (question, answer, confidence, no raw
      input) to a local JSONL file, for threshold tuning and fine-tuning later.
- [ ] **Temperature refitting.** A script that takes a labelled log and fits Laya's per-bucket
      temperatures (and a CLM temperature), since Laya ships over-confident until refitted on your
      own data.
- [ ] **Language handling.** Make multilingual the default for Dutch projects, or detect Dutch
      reliably. The router sent 5 of 6 Dutch test messages to the English checkpoint.
- [ ] **A shared decision service on `blitsol-prod`** (`laya-serve` behind auth) if several products
      need decisions in production, instead of one model per process. Check the 8 GB memory budget
      first.
- [ ] **CLM hosting:** a GPU host, or stay on CPU for low-volume work; decide from phase 1 volumes
      and latency needs.
- [ ] **Fine-tuning:** Laya's typed-decisions notebook (0.362 → 0.766 on its benchmark) and CLM's
      cheap head fine-tuning, for the decisions with enough labelled data.
- [ ] **PII redaction** before inference, where projects send customer text (Laya's
      `on_predict_start` hook).

---

## Phase 3: implement per project

For each project whose research recommends it:

- [ ] Build the decision behind a feature flag, in **shadow mode** first: log the model's answer
      next to the current behaviour without acting on it.
- [ ] Compare on real traffic for 1–2 weeks, and confirm or adjust the threshold.
- [ ] Turn it on for the confident slice only. Everything below the threshold keeps the current
      path (LLM, person, or rules).
- [ ] Add monitoring: answer distribution, share escalated, and accuracy on a weekly labelled
      sample.

| project | research done | design agreed | shadow mode | live |
|---|---|---|---|---|
| Kōro | [ ] | [ ] | [ ] | [ ] |
| Outbound engine | [ ] | [ ] | [ ] | [ ] |
| Model routing layer | [ ] | [ ] | [ ] | [ ] |
| Phone agent NL | [ ] | [ ] | [ ] | [ ] |
| Moneybird automation | [ ] | [ ] | [ ] | [ ] |
| AI search visibility | [ ] | [ ] | [ ] | [ ] |
| Paperwork / compliance | [ ] | [ ] | [ ] | [ ] |
| Game audio | [ ] | [ ] | [ ] | [ ] |
| Localisation / playtesting | [ ] | [ ] | [ ] | [ ] |
| 3D projects | [ ] | [ ] | [ ] | [ ] |
| Trading | [ ] | [ ] | [ ] | [ ] |
