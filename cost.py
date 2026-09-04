"""Сколько потратил агент: сводка по telemetry/events.jsonl."""

import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "telemetry/events.jsonl")
if not path.exists():
    sys.exit(f"Нет файла {path} — агент ещё не отвечал ни на один запрос.")

total = Counter()
runs = set()
for line in path.read_text(encoding="utf-8").splitlines():
    e = json.loads(line)
    if e.get("event") != "llm_call":
        continue
    runs.add(e["run_id"])
    for k in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens"):
        total[k] += e.get(k, 0)
    total["calls"] += 1
    total["cost"] += e.get("cost", 0.0)

cacheable = total["input_tokens"] + total["cached_input_tokens"]
hit = total["cached_input_tokens"] / cacheable * 100 if cacheable else 0
print(f"Запросов к боту:      {len(runs)}")
print(f"Вызовов модели:       {total['calls']}")
print(f"Токенов входных:      {total['input_tokens']:,} (не из кэша)")
print(f"  из кэша:            {total['cached_input_tokens']:,}  ← до оптимизации здесь был бы 0")
print(f"  записано в кэш:     {total['cache_write_input_tokens']:,}")
print(f"Токенов выходных:     {total['output_tokens']:,}")
print(f"Попаданий в кэш:      {hit:.1f}%")
print(f"ПОТРАЧЕНО:            ${total['cost']:.4f}")
