#!/usr/bin/env bash
# _ensure_variant_mask.sh <VARIANT>
# Ensures run_m1b_freelunch_sweep.sh's mask_for_variant knows <VARIANT>.
# If missing, derives {N:[bot-K heads]} from botK_freq_alllayers28.json and
# inserts a hardcoded case (cp -a backup first). VARIANT must match L<N>K<K>.
#
# SAFETY: only call this when NO run_m1b_freelunch_sweep.sh process is running
# (i.e. between dispatchers), because editing a running bash script corrupts it.
# The driver enforces this by calling ensure only between sequential dispatchers.
set -uo pipefail
ROOT="/apdcephfs/private_shayladeng/tokenrl_autoVLA"; cd "$ROOT"
PY="/apdcephfs/private_shayladeng/miniconda3/envs/autovla/bin/python"
SWEEP="scripts/run_m1b_freelunch_sweep.sh"
JSON="exp/m1b2_phase2_v0/botK_freq_alllayers28.json"
V="${1:?usage: $0 <LNKk>}"

# already known?
if grep -qE "^[[:space:]]*${V}\)" "$SWEEP"; then exit 0; fi
# refuse if sweep is live (would corrupt running shards)
if pgrep -f 'run_m1b_freelunch_sweep.sh' >/dev/null; then
  echo "[ensure] REFUSE: sweep is running; not editing (unsafe). V=$V"; exit 2
fi
# parse L<N>K<K>
if [[ ! "$V" =~ ^L([0-9]+)K([0-9]+)$ ]]; then
  echo "[ensure] cannot parse variant $V"; exit 3
fi
N="${BASH_REMATCH[1]}"; K="${BASH_REMATCH[2]}"
MASK=$("$PY" - "$N" "$K" <<'PY'
import json, sys
N, K = sys.argv[1], sys.argv[2]
d = json.load(open("exp/m1b2_phase2_v0/botK_freq_alllayers28.json"))["per_layer"]
heads = d[N][K]  # already sorted list
print("{%s: %s}" % (N, heads))
PY
)
[[ -z "$MASK" ]] && { echo "[ensure] empty mask for $V"; exit 4; }
cp -a "$SWEEP" "${SWEEP}.bak_ensure_$(date +%Y%m%d_%H%M%S)"
# insert before the default '*)' case line
LINE="    ${V})  echo '${MASK}' ;;"
awk -v ins="$LINE" '/^[[:space:]]*\*\)[[:space:]]*echo .*return 1/ && !done {print ins; done=1} {print}' \
    "$SWEEP" > "${SWEEP}.tmp" && mv "${SWEEP}.tmp" "$SWEEP"
chmod +x "$SWEEP"
echo "[ensure] added ${V} -> ${MASK}"
