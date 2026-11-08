# Sourced by the launcher scripts. One place for stream -> spec, model, effort, owned paths.
export CODEX_PROJECT_TAG=tdx
MAIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MAIN
PY="${TDX_PY:-$HOME/.venvs/tdx/bin/python}"

declare -A SPEC=([W0]=W0_HARNESS [W1]=W1_GLM [W2]=W2_GBDT [W3]=W3_SEQ [W4]=W4_REGIME [W5]=W5_STACK [W6]=W6_PACKAGE)
# Model routing from claude_output/specs (slugs checked with `codex debug models` on 2026-11-07).
declare -A MODEL=([W0]=gpt-6.1-sol [W1]=gpt-6.1-sol [W2]=gpt-6.1-sol [W3]=gpt-6-astra [W4]=gpt-6.1-sol [W5]=gpt-6.1-sol [W6]=gpt-6.1-sol)
declare -A EFFORT=([W0]=medium [W1]=high [W2]=medium [W3]=high [W4]=medium [W5]=high [W6]=high)
# Paths each stream may change; collect_stream.sh copies exactly these back to the main checkout.
declare -A OWNED=(
  [W0]="src/tdx tests runs/pooled claude_output/notes"
  [W1]="src/tdx/arms/glm tests/glm runs/glm claude_output/notes"
  [W2]="src/tdx/arms/gbdt tests/gbdt runs/gbdt claude_output/notes"
  [W3]="src/tdx/arms/seq tests/seq runs/seq claude_output/notes"
  [W4]="src/tdx/arms/regime tests/regime runs/regime claude_output/notes"
  [W5]="src/tdx/stack tests/stack runs/stack claude_output/notes"
  [W6]="src/tdx/package src/tdx/pipeline.py config tests/package claude_output/sh/repro_check.sh claude_output/sh/refit_round.sh claude_output/notes"
)
clone_dir() { echo "$(dirname "$MAIN")/$(basename "$MAIN")-${1,,}"; }
