# laya-mcp-server

A local [MCP](https://modelcontextprotocol.io) server that lets Claude Code call the open-source
[Laya](https://huggingface.co/convaiinnovations/laya) decision model (Apache-2.0) as a fast,
calibrated classifier. Laya is a non-autoregressive encoder: it scores a fixed set of options in
**one forward pass** and returns a **temperature-calibrated probability distribution**. It never
generates text, so there is nothing to parse or hallucinate.

It exposes two tools:

| tool | what it does |
|---|---|
| `evaluate_laya_decision` | Answers one `choice`, `score` or `bool` question about a text/JSON state |
| `laya_server_status` | Reports whether the checkpoints are loaded, on which device, and the server's default model |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+. From the repository root:

```bash
uv sync --project laya-mcp-server                                        # torch, transformers, laya, mcp
uv run --project laya-mcp-server --frozen laya-decision-mcp --prefetch   # downloads the English checkpoint (~0.8 GB), runs one decision
```

`--prefetch` prints a sample decision and exits. Run it once so Claude Code's first launch
doesn't have to wait for the download.

## Claude Code

The repository root's [`.mcp.json`](../.mcp.json) registers the server, so Claude Code finds it
whenever it starts in this repository:

```json
{
  "mcpServers": {
    "laya-mcp-server": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--project", "${CLAUDE_PROJECT_DIR:-.}/laya-mcp-server", "--frozen", "--quiet", "laya-decision-mcp"],
      "env": { "LAYA_MCP_MODEL": "auto", "LAYA_MCP_PRELOAD": "english" }
    }
  }
}
```

* The first interactive `claude` run in the repository asks you to approve project servers. Approve
  it once, or check it with `claude mcp get laya-mcp-server`. `claude -p` and cloud sessions load
  project servers without asking.
* The tool shows up as `mcp__laya-mcp-server__evaluate_laya_decision`. You can ask Claude to use it
  ("classify these tickets with Laya") or allow it without prompts:
  `--allowedTools mcp__laya-mcp-server__evaluate_laya_decision`.
* The server answers the MCP handshake immediately and loads the model in the background. A
  decision that arrives while the model is still loading waits for it.

## The `evaluate_laya_decision` tool

| parameter | required | meaning |
|---|---|---|
| `state` | yes | Text, a JSON object (e.g. `{"subject": …, "body": …}`) or a JSON array (a conversation, oldest turn first). A string that holds JSON is parsed. |
| `question_type` | yes | `choice` (one of the options), `score` (ordinal rubric) or `bool` (yes/no) |
| `options` | for `choice` | `choice`: labels, or `{label: description}`. `score`: rubric levels, lowest first (default `very low … very high`). `bool`: optional `[yes_description, no_description]` |
| `instructions` | yes | The question to answer, e.g. "Which department should handle this?" |
| `model` | no | `auto` (default; English text goes to `english`, other languages to `multilingual`), `english`, `multilingual`, `typed-decisions` |

Example result (`choice`):

```json
{
  "answer": "billing",
  "probabilities": {"billing": 0.9826, "technical": 0.0055, "sales": 0.0074, "other": 0.0044},
  "confidence": 0.9826,
  "question_type": "choice",
  "model": "english",
  "routing_reason": "English Latin text",
  "device": "cpu",
  "calibration": {"method": "temperature_scaling", "bucket": "choice:3-5", "temperature": 1.7602, "calibrated": true},
  "latency_ms": 235.8,
  "input_tokens": 43,
  "generated_tokens": 0,
  "warnings": []
}
```

* `probabilities` is always the full distribution over the options, in option order. `answer` is
  its highest-probability key (for `bool` that is `"true"` or `"false"`).
* `confidence` is the calibrated probability of `answer`. `score` also returns `expected_score`
  (0 = first level) and `expected_score_normalized` (0–1). `bool` also returns `probability_true`.
* `calibration` gives the fitted temperature the logits were divided by. `calibrated: false`
  (with a warning) marks a bucket that ships no valid temperature.
* `warnings` flags truncated input, uncalibrated buckets, too many options, and a default rubric.

## Configuration

| variable | default | meaning |
|---|---|---|
| `LAYA_DEVICE` | best available | torch device: `cpu`, `cuda`, `cuda:1`, `mps` |
| `LAYA_MCP_MODEL` | `auto` | default checkpoint for calls that don't name one |
| `LAYA_MCP_PRELOAD` | follows `LAYA_MCP_MODEL` (`auto` → `english`) | checkpoints to load at startup, comma separated, or `none` |
| `HF_HUB_OFFLINE` | unset | set to `1` to run from the Hugging Face cache without network access |

Memory: the English checkpoint holds 1.7 GB of fp32 weights (the process peaks at ~3.3 GB RSS
while loading it). Multilingual holds about 1.3 GB.

## Latency

A decision is one forward pass. Laya publishes **~33 ms** per decision on a T4 GPU (39.5 ms on the
English checkpoint, 32.8 ms on multilingual). On CPU it is several times slower. Measured with
`scripts/benchmark.py` in a 4-vCPU Intel Xeon container with no GPU:

| checkpoint | device | choice p50 | score p50 | bool p50 |
|---|---|---|---|---|
| english | cpu (4 threads) | 225 ms | 232 ms | 287 ms |
| multilingual | cpu (4 threads) | 90 ms | 95 ms | 97 ms |

The MCP layer adds about 2 ms per call. To measure your own machine:

```bash
uv run --project laya-mcp-server --frozen python laya-mcp-server/scripts/benchmark.py
```

## Tests

```bash
cd laya-mcp-server
uv run --frozen pytest -m "not model"   # 35 validation tests, no model, < 1 s
uv run --frozen pytest                   # everything, ~1 min on CPU (downloads the English checkpoint)
```

| file | verifies |
|---|---|
| `test_questions.py` | argument validation, and how `choice`/`score`/`bool` map onto Laya questions |
| `test_tool.py` | the tool over an in-process MCP client: schema, a valid distribution for every type, sanity answers, JSON states, truncation and calibration warnings, errors |
| `test_inference.py` | one forward pass per decision, `generate()` is never called, the model is an encoder with no LM head, and the probabilities equal `softmax(logits / T)` with the checkpoint's fitted `T` |
| `test_latency.py` | p50 model time under a device budget: 50 ms on CUDA (the ~33 ms claim plus headroom), 500 ms on CPU; override with `LAYA_LATENCY_BUDGET_MS` |
| `test_stdio.py` | launches the server from `.mcp.json` the way Claude Code does, and checks that laya's stray stdout prints can't corrupt the protocol |

## Honest limits

These come from Laya's model card and the checks above:

* **Zero-shot accuracy is task-dependent.** The base checkpoints score well on routing, topic,
  intent and moderation tasks, and near chance on complex domain workflows unless fine-tuned. Treat
  `confidence` below ~0.6 as uncertain, and check high-stakes decisions another way.
* **`bool` uses neutral labels.** Laya's native yes/no type (`noul`) renders its options as
  `false:`/`true:`, and on the English checkpoint that can pin clearly positive inputs at P(yes)=0
  (upstream issue #156). This server sends `A: yes` / `B: no` instead. On a spot check of 8
  unambiguous questions that scored 8/8, against 7/8 for the default labels.
* **The multilingual checkpoint is weak on English yes/no questions** (4–6 of 8 on the same spot
  check). That is why `auto` routes English text to the English checkpoint.
* **Many options:** options share a fixed token budget, so accuracy drops past ~20 options. For
  `choice` with 11+ options, the English checkpoint's fitted temperature is invalid, so that bucket
  is reported as `calibrated: false`.
* **Context:** the English checkpoint reads ~320 tokens of state (512 in total). Longer input is
  truncated and flagged in `warnings`. `multilingual` reads 1,024 tokens.

`laya` itself also ships a generic `laya-mcp-server` console script (tools `laya_predict`,
`laya_route`, `laya_preset`, `laya_status`). This package's script is named `laya-decision-mcp` so
the two don't clash.
