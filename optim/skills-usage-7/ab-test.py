"""A/B: re-evaluate the Phase-4 seed pair vs the Phase-4 best pair, full set."""
import importlib.util
import json

importlib.util.spec_from_file_location  # noqa
spec = importlib.util.spec_from_file_location("train", "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

run_dir = "runs/gepa-0823-121325-873603"
seed = json.load(open(f"{run_dir}/seed-baseline.json"))["pair"]
best = json.load(open(f"{run_dir}/best-pair.json"))
cases = train.load_cases()

for label, pair in (("ab-seed", seed), ("ab-best", best)):
    result = train.evaluate_pair(dict(pair), cases, label)
    json.dump(result, open(f"{run_dir}/{label}.json", "w"), indent=2)
    print(f"== {label}: accuracy={result['accuracy_pct']}% "
          f"skill_leak={result['skill_leak_pct']}% self_echo={result['self_echo_pct']}%", flush=True)

# leave the better pair live
a = json.load(open(f"{run_dir}/ab-seed.json"))
b = json.load(open(f"{run_dir}/ab-best.json"))
winner = seed if a["accuracy_pct"] >= b["accuracy_pct"] else best
train.PAIR_FILE.write_text(json.dumps(winner, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"WINNER kept live: {'seed' if winner is seed else 'best'}", flush=True)
