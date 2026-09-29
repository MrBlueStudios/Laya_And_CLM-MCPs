#!/usr/bin/env bash
# Register laya-mcp-server and clm-mcp-server with Claude Code at user scope, so every project
# you open (Kōro, Blitsol, ...) gets the decision tools, not just this repository.
#
#   ./register-decision-mcps.sh            # add both (re-running replaces them)
#   ./register-decision-mcps.sh --remove   # remove both
#
# The CLM entry reads CLM_BASE_URL and CLM_API_KEY from your environment when Claude Code starts,
# so no key is written to ~/.claude.json.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
servers=(laya-mcp-server clm-mcp-server)

for name in "${servers[@]}"; do
  claude mcp remove --scope user "$name" >/dev/null 2>&1 || true
done
if [[ "${1:-}" == "--remove" ]]; then
  echo "Removed ${servers[*]} from user scope."
  exit 0
fi

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
for name in "${servers[@]}"; do
  uv sync --quiet --project "$here/$name"
done

claude mcp add-json --scope user laya-mcp-server "$(cat <<JSON
{"type": "stdio", "command": "uv",
 "args": ["run", "--project", "$here/laya-mcp-server", "--frozen", "--quiet", "laya-decision-mcp"],
 "env": {"LAYA_MCP_MODEL": "\${LAYA_MCP_MODEL:-auto}", "LAYA_MCP_PRELOAD": "\${LAYA_MCP_PRELOAD:-english}"}}
JSON
)"

claude mcp add-json --scope user clm-mcp-server "$(cat <<JSON
{"type": "stdio", "command": "uv",
 "args": ["run", "--project", "$here/clm-mcp-server", "--frozen", "--quiet", "clm-decision-mcp"],
 "env": {"CLM_BASE_URL": "\${CLM_BASE_URL:-http://127.0.0.1:8700}", "CLM_API_KEY": "\${CLM_API_KEY:-}"}}
JSON
)"

echo
echo "Registered at user scope. Check with: claude mcp list"
echo "Laya downloads its model (~0.8 GB) on first use; run this once to do it now:"
echo "  uv run --project $here/laya-mcp-server --frozen laya-decision-mcp --prefetch"
echo "CLM needs a running clm-serve; see $here/clm-mcp-server/README.md."
