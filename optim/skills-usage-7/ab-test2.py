"""Multi-round A/B: Phase-4 best pair vs Phase-5 best pair, 3 full evals each."""
import importlib.util
import json
import statistics

spec = importlib.util.spec_from_file_location("train", "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

p4 = json.load(open("runs/gepa-0823-121325-873603/best-pair.json"))
p5 = json.load(open("runs/gepa-0823-125518-926797/best-pair.json"))
cases = train.load_cases()

acc = {"p4": [], "p5": []}
leak = {"p4": [], "p5": []}
for rnd in range(3):
    order = (("p4", p4), ("p5", p5)) if rnd % 2 == 0 else (("p5", p5), ("p4", p4))
    for label, pair in order:
        r = train.evaluate_pair(dict(pair), cases, f"ab2-{label}-r{rnd}")
        acc[label].append(r["accuracy_pct"])
        leak[label].append(r["skill_leak_pct"])
        print(f"r{rnd} {label}: acc={r['accuracy_pct']}% skill_leak={r['skill_leak_pct']}%", flush=True)

m4, m5 = statistics.mean(acc["p4"]), statistics.mean(acc["p5"])
l4, l5 = statistics.mean(leak["p4"]), statistics.mean(leak["p5"])
print(f"MEAN p4: acc={m4:.2f}% leak={l4:.2f}%  rounds={acc['p4']}")
print(f"MEAN p5: acc={m5:.2f}% leak={l5:.2f}%  rounds={acc['p5']}")
winner = p4 if m4 >= m5 else p5
train.PAIR_FILE.write_text(json.dumps(winner, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"WINNER kept live: {'p4' if winner is p4 else 'p5'}", flush=True)
