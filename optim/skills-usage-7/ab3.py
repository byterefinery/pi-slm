"""Multi-round A/B between two saved best pairs."""
import importlib.util
import json
import statistics
import sys

spec = importlib.util.spec_from_file_location("train", "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

run_a, run_b = sys.argv[1], sys.argv[2]
label_a, label_b = "A", "B"
pa = json.load(open(f"runs/{run_a}/best-pair.json"))
pb = json.load(open(f"runs/{run_b}/best-pair.json"))
cases = train.load_cases()

acc = {label_a: [], label_b: []}
leak = {label_a: [], label_b: []}
for rnd in range(3):
    order = ((label_a, pa), (label_b, pb)) if rnd % 2 == 0 else ((label_b, pb), (label_a, pa))
    for label, pair in order:
        r = train.evaluate_pair(dict(pair), cases, f"ab3-{label}-r{rnd}")
        acc[label].append(r["accuracy_pct"])
        leak[label].append(r["skill_leak_pct"])
        print(f"r{rnd} {label}: acc={r['accuracy_pct']}% skill_leak={r['skill_leak_pct']}% "
              f"self_echo={r['self_echo_pct']}%", flush=True)

ma, mb = statistics.mean(acc[label_a]), statistics.mean(acc[label_b])
la, lb = statistics.mean(leak[label_a]), statistics.mean(leak[label_b])
print(f"MEAN {label_a}: acc={ma:.2f}% leak={la:.2f}% rounds={acc[label_a]}")
print(f"MEAN {label_b}: acc={mb:.2f}% leak={lb:.2f}% rounds={acc[label_b]}")
winner = pa if ma >= mb else pb
which = label_a if ma >= mb else label_b
train.PAIR_FILE.write_text(json.dumps(winner, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"WINNER kept live: {which} ({run_a if which == label_a else run_b})", flush=True)
