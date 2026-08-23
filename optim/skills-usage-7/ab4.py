"""A/B: live pair (skill-pair.json) vs restructured candidate (pair-candidate.json)."""
import importlib.util
import json
import statistics

spec = importlib.util.spec_from_file_location("train", "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

live = train.load_pair()
cand = json.load(open("pair-candidate.json"))
cases = train.load_cases()

acc = {"live": [], "cand": []}
leak = {"live": [], "cand": []}
for rnd in range(3):
    order = (("live", live), ("cand", cand)) if rnd % 2 == 0 else (("cand", cand), ("live", live))
    for label, pair in order:
        r = train.evaluate_pair(dict(pair), cases, f"ab4-{label}-r{rnd}")
        acc[label].append(r["accuracy_pct"])
        leak[label].append(r["skill_leak_pct"])
        print(f"r{rnd} {label}: acc={r['accuracy_pct']}% skill_leak={r['skill_leak_pct']}% "
              f"self_echo={r['self_echo_pct']}%", flush=True)

ml, mc = statistics.mean(acc["live"]), statistics.mean(acc["cand"])
print(f"MEAN live: acc={ml:.2f}% leak={statistics.mean(leak['live']):.2f}% rounds={acc['live']}")
print(f"MEAN cand: acc={mc:.2f}% leak={statistics.mean(leak['cand']):.2f}% rounds={acc['cand']}")
# The candidate has the clean role-separated design (no markdown, content=
# usage explanation, thinking=true CoT). Prefer it unless it clearly loses.
if mc >= ml - 3.0:
    winner, which = cand, "cand (design-preferred, within 3 pts)"
else:
    winner, which = live, "live"
train.PAIR_FILE.write_text(json.dumps(winner, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"WINNER kept live: {which}", flush=True)
