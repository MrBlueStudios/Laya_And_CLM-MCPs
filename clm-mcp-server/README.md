# clm-mcp-server

A local [MCP](https://modelcontextprotocol.io) server that lets Claude Code use
[CLM-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (Contrastive-LM, Apache-2.0), a
"System One" decision model. CLM embeds a state and each candidate with a frozen Qwen3-8B encoder
plus two small trained heads, and returns a probability distribution over the candidates in one
pass. It never generates text.

It complements [`laya-mcp-server`](../laya-mcp-server):

| use | tool |
|---|---|
| Pick the best of **many** candidates (tools, retrieved passages, draft answers, next actions; up to 1,000) | `rank_clm_candidates` |
| Ask a typed question (choice / score / bool), same shape as Laya | `evaluate_clm_decision` |
| Is the backend up, which heads does it serve, how is this server configured | `clm_server_status` |
| Calibrated triage of a few labels, Dutch or other languages, CPU-only hosts | use Laya instead |

## How it fits together

```text
Claude Code ──stdio──▶ clm-mcp-server ──HTTP(S)──▶ clm-serve ──▶ Qwen3-8B encoder
                       (this package:              (contrastive-lm:    (vLLM on a GPU,
                        mcp + httpx, no torch)      the two heads)      or llama.cpp on CPU)
```

This package is a thin client: it doesn't load any model and has two runtime dependencies (`mcp`,
`httpx`), so it starts instantly. The model runs in `clm-serve`, which can be on this machine or on
a GPU box elsewhere.

## Setup

### 1. The MCP server

```bash
uv sync --project clm-mcp-server
uv run --project clm-mcp-server --frozen clm-decision-mcp --check   # backend status plus one sample ranking
```

The repository root's [`.mcp.json`](../.mcp.json) registers it as `clm-mcp-server`. It reads
`CLM_BASE_URL` and `CLM_API_KEY` from your environment, so the key is never committed:

```bash
export CLM_BASE_URL=http://127.0.0.1:8700     # the default
export CLM_API_KEY=...                        # the key clm-serve was started with
claude
```

### 2. The CLM backend

CLM needs Qwen3-8B embeddings (last-token pooling) from an OpenAI-compatible `/v1/embeddings`
endpoint, and `clm-serve` in front of it.

**A. On a machine with an NVIDIA GPU (recommended, ~28 ms per decision on an RTX 4090).** Needs
~16 GB of GPU memory for the bf16 encoder.

```bash
pip install contrastive-lm "torch>=2.6"
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --enable-prefix-caching \
    --max-model-len 2048 --host 127.0.0.1 --port 8090 &

# Fetch the head at a pinned revision and check it before loading (it is a pickle-format .pt file)
hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt \
    --revision e939398d4556fcd9400c76fa8c5a513202f42b0a --local-dir ~/clm
echo "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5  $HOME/clm/CLM_v0.1-8B.pt" | sha256sum -c

export CLM_API_KEY=$(openssl rand -hex 24)   # give the same value to the MCP server
clm-serve --host 127.0.0.1 --port 8700 --no-ui --no-download --ckpt ~/clm/CLM_v0.1-8B.pt
```

**B. GPU somewhere else** (a rented GPU, a desktop). Run A there, then reach it through an SSH
tunnel so nothing is exposed and traffic is encrypted:

```bash
ssh -N -L 8700:127.0.0.1:8700 user@gpu-box     # then CLM_BASE_URL=http://127.0.0.1:8700
```

Or put it behind an HTTPS reverse proxy and use `CLM_BASE_URL=https://...`. This server refuses
plain http to any host but this machine, unless you set `CLM_ALLOW_INSECURE_HTTP=1` for a network
that is already private and encrypted (WireGuard, Tailscale).

**C. CPU only, for trying it out (slow).** No NVIDIA GPU needed.
[`scripts/llamacpp_embeddings.py`](scripts/llamacpp_embeddings.py) serves the same endpoint from a
quantised Qwen3-8B GGUF with llama.cpp:

```bash
pip install contrastive-lm llama-cpp-python "torch>=2.6"
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf --local-dir ~/clm      # 8.7 GB
python clm-mcp-server/scripts/llamacpp_embeddings.py --gguf ~/clm/Qwen3-8B-Q8_0.gguf --port 8090 &
clm-serve --host 127.0.0.1 --port 8700 --no-ui --no-download --ckpt ~/clm/CLM_v0.1-8B.pt --device cpu
```

Measured in a 4-vCPU Xeon container (28 September 2026):

- **Speed.** Each new text costs about 1 s per 30 tokens to embed; repeated texts come from CLM's
  cache and answer in ~1–5 ms. Ranking the same candidate list against new states only pays for
  the new state.
- **Fidelity.** It reproduces the model card's examples: the tides ranking gave 0.994 for "the
  Moon" (card: 0.993), and the billing ticket was routed to billing at 0.990 (card: 0.939). So the
  decisions match, while the probabilities differ by a few points. Nothing more was compared,
  because there was no GPU to run the bf16 reference.
- **Memory.** ~9 GB of RAM for the Q8_0 GGUF. A build of llama.cpp with Vulkan or SYCL can use an
  Intel Arc GPU instead.

### Securing the backend

`clm-serve`'s defaults are made for a local playground, so override them on anything shared:

| default | risk | do this |
|---|---|---|
| binds `0.0.0.0` | anyone on the network can query it | `--host 127.0.0.1`, and reach it by SSH tunnel |
| no auth unless `CLM_API_KEY` is set | open API | always set `CLM_API_KEY` |
| web playground at `/` | extra surface | `--no-ui` |
| downloads the head from `main` and loads a pickle `.pt` | a changed upstream file runs through `torch.load` | pinned revision plus `sha256sum -c` above, `--no-download`, and torch ≥ 2.6 (its `torch.load` defaults to `weights_only=True`) |
| `vllm serve` listens on all interfaces | an open embeddings endpoint | `--host 127.0.0.1` |

## The tools

### `rank_clm_candidates`

| parameter | required | meaning |
|---|---|---|
| `state` | yes | What to rank against: a query, a task, a situation (text or JSON) |
| `candidates` | yes | 2–1,000 distinct texts, up to 4,000 characters each |
| `instructions` | no | A question appended to the state, e.g. "Which tool should handle this?" |
| `top_k` | no | How many to return (default 10, max 50) |
| `model` | no | Head name (default `clm-latest`) |

It returns `ranking` (`rank`, `index` into your list, a 200-character `candidate_preview`,
`probability`, `lift`), plus `candidates_total`, `uniform_probability`, `top_margin` and timing.
The probabilities sum to 1 over *all* candidates, so they shrink as the list grows. Judge them with
`lift` (probability × candidates; 1.0 means no better than chance) and `top_margin`. A warning
flags a winner under 2× chance.

### `evaluate_clm_decision`

It takes the same parameters as `evaluate_laya_decision` (`state`, `question_type`: choice / score /
bool, `options`, `instructions`, `model`) and returns the same shape: `answer`, `probabilities`,
`confidence`, `expected_score` or `probability_true`, `calibration`, timing, `input_tokens` and
`generated_tokens` (always 0).

### How to phrase options

CLM compares meanings, and it reads each option's *description* when one is given, otherwise the
label. Measured on the CPU backend above:

| request | options written as | result |
|---|---|---|
| "Book a table for two at an Italian place tonight" | `reserve_restaurant: book a restaurant table` style (name: description) | wrong tool first |
| same | `Book a table at a restaurant.` style (plain descriptions) | right tool first (0.62) |
| "Production has been down for every customer since 10am!" | levels `not urgent` / `soon` / `blocking` | "not urgent" at 1.00 (wrong) |
| same | `Critical: an outage that needs action right now.` style | critical first (0.72) |

So write every option as a short sentence describing that answer, keep candidates in one form,
and map indices back to your own IDs. Adding `instructions` helped in some tests and hurt in
others, so test both ways on your own data.

### What the numbers mean

- **Relative, not calibrated.** CLM always ranks something first, even if nothing fits, so its
  probabilities are shares of the candidates you gave it. `calibration.calibrated` is always
  `false`. For an absolute check ("is any of these passages relevant?"), ask a `bool` question, or
  use Laya.
- **English only.** The heads are trained on English.
- **Context.** Each text is truncated at 2,048 tokens unless the encoder and `clm-serve` are started
  with a larger `--max-model-len` / `--max-tokens`.

## Configuration

| variable | default | meaning |
|---|---|---|
| `CLM_BASE_URL` | `http://127.0.0.1:8700` | where `clm-serve` listens |
| `CLM_API_KEY` | unset | bearer key; sent only to `CLM_BASE_URL`, never over plain http to another host |
| `CLM_MODEL` | `clm-latest` | default head |
| `CLM_TIMEOUT` | `120` | seconds per request (CPU backends need more for new texts) |
| `CLM_TEMPERATURE` | `1` | divides the logits before the softmax, for when you have fitted one |
| `CLM_ALLOW_INSECURE_HTTP` | unset | `1` allows plain http to another host on a private, encrypted network |

If the configuration is invalid, the server still starts, and every tool call (and
`clm_server_status`) explains what to fix.

## Security properties

- The backend address and key come only from the environment; no tool argument can change where
  requests go.
- Redirects are never followed, so the key can't be forwarded to another address.
- Plain http is refused except to this machine; credentials in the URL are refused.
- Responses are capped at 8 MB. Answers are validated (exactly the requested options, finite
  probabilities in [0, 1] that sum to 1). The output carries only numbers and your own strings: the
  answer label is recomputed from the distribution, never copied from the server.
- Inputs are bounded (100,000-character state, 1,000 candidates, 1 MB of candidate text), and model
  names must match `[A-Za-z0-9._-]`.
- The API key never appears in tool output, status or error messages.

## Tests

```bash
cd clm-mcp-server
uv run --frozen pytest                    # 93 tests against a fake clm-serve over real HTTP, ~15 s
CLM_TEST_URL=http://127.0.0.1:8700 CLM_TEST_API_KEY=... uv run --frozen pytest -m live   # a real backend
```

| file | verifies |
|---|---|
| `test_questions.py` | validation and limits, and decoding: bad distributions are rejected, labels come from the request |
| `test_settings.py` | the URL policy (loopback-only http, no credentials in URLs), key handling, value ranges |
| `test_backend.py` | the auth header, 401/422/502/500 handling, redirects not followed (the key never reaches the other host), timeouts, size caps, non-JSON bodies |
| `test_tool.py` | the three tools over an in-process MCP client, readable errors, the misconfigured and unreachable states |
| `test_stdio.py` | launching from `.mcp.json` like Claude Code, with the key coming from the environment |
| `test_live.py` | the model card's examples and the phrasing findings above, against a real CLM |
