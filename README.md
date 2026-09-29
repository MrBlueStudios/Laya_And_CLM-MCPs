# Laya_And_CLM-MCPs

Two local MCP servers that give Claude Code "System One" decision models: typed questions
(choice / score / yes-no) answered with a probability distribution in one forward pass, with no
text generation.

| Folder | What it is |
|---|---|
| [`laya-mcp-server/`](laya-mcp-server/README.md) | `evaluate_laya_decision`: Laya (convaiinnovations/laya). Calibrated, runs on CPU, English and multilingual. |
| [`clm-mcp-server/`](clm-mcp-server/README.md) | `evaluate_clm_decision`, `rank_clm_candidates`: CLM-8B (Contrastive-LM). Ranks up to 1,000 candidates; needs a `clm-serve` backend. |
| [`DECISION-MODELS-PLAYBOOK.md`](DECISION-MODELS-PLAYBOOK.md) | How the models work, how they compare, and how to use them with RAG, scraping, trading and chatbots. |
| [`DECISION-MODELS-ROLLOUT.md`](DECISION-MODELS-ROLLOUT.md) | Rollout plan and per-project research checklist. |
| [`docs/projects-repo-prompt.md`](docs/projects-repo-prompt.md) | Claude Code prompt that builds the `Projects` repo, with this repo as a submodule. |

## Setup

On the machine that runs Claude Code, this repo lives at `~/Projects/Laya_And_CLM-MCPs`:

```bash
./register-decision-mcps.sh     # register both servers at user scope (every project gets them)
claude mcp list                 # check
```

Opening Claude Code inside this folder also works without registering, through `.mcp.json`.
The servers' READMEs cover the model download, the CLM backend and the tests.

This repo used to be called `Projects`; it was renamed when `Projects` became the repo for all
project folders.
