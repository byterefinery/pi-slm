#!/bin/bash
# Watchdog: wait for the student endpoint to recover, run the draft-selection
# round, then auto-commit + push the results (score in the message).
# Detached via setsid; survives session aborts. Log: auto-run.log
cd /home/mtasic/projects-b/pi-slm/optim/skills-usage-2 || exit 1
exec >> auto-run.log 2>&1
echo "[$(date '+%F %T')] watchdog started"

# API base/key from pi's models.json (never hardcoded)
read -r BASE KEY < <(.venv/bin/python -c "
import json, re
cfg = json.loads(re.sub(r',\s*([}\]])', r'\1', open('/home/mtasic/.pi/agent/models.json').read()))
p = next(x for x in cfg['providers'].values() if x.get('baseUrl') and x.get('apiKey'))
print(p['baseUrl'].rstrip('/'), p['apiKey'])")

# --- wait for the student endpoint (5-min interval, up to 24h)
code=000
for i in $(seq 1 288); do
  code=$(curl -sS --max-time 60 "$BASE/chat/completions" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -H "x-session-affinity: dspy-optim" \
    -d '{"model":"LiquidAI/LFM2.5-2.6B","messages":[{"role":"user","content":"Reply: ok"}],"max_tokens":16,"temperature":0.1,"top_k":50,"repeat_penalty":1.1,"reasoning_effort":"high"}' \
    -o /dev/null -w '%{http_code}' 2>/dev/null)
  echo "[$(date '+%F %T')] probe $i: HTTP $code"
  [ "$code" = "200" ] && break
  sleep 300
done
[ "$code" = "200" ] || { echo "student endpoint never recovered"; exit 1; }

echo "[$(date '+%F %T')] student ready - running selection (DRAFTS=20)"
DRAFTS=20 .venv/bin/python -u train.py --select-only > train.log 2>&1
echo "[$(date '+%F %T')] selection finished (exit $?)"

# --- final avg from the log
AVG=$(grep -oE 'FINAL \(with pair\) tzip-[a-z0-9-]+ score=[0-9.]+' train.log | grep -oE 'score=[0-9.]+' | cut -d= -f2 | awk '{s+=$1; n++} END {if (n>0) printf "%.3f", s/n; else print "n/a"}')
echo "[$(date '+%F %T')] final avg=$AVG"

# --- auto commit + push (score in the message)
cd /home/mtasic/projects-b/pi-slm || exit 1
git add optim/skills-usage-2/
if git diff --cached --quiet; then
  echo "[$(date '+%F %T')] nothing to commit"
  exit 0
fi
git commit -q -m "chore(skills-usage-2): auto selection after endpoint recovery - final score ${AVG:-?}
  (18-case matrix: default/lite/full/on/ultra/off + 12 mode switches)"
git push
echo "[$(date '+%F %T')] committed and pushed"
