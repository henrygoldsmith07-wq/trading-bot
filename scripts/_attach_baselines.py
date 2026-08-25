"""Attach baselines_out to metrics after construction (one-shot)."""
lines = open("bot/__main__.py", encoding="utf-8").read().splitlines(keepends=True)

metrics_line = None
timing_line = None
for i, line in enumerate(lines):
    if line.strip() == "metrics = {":
        metrics_line = i
    if '"timing_s"' in line:
        timing_line = i

assert metrics_line is not None and timing_line is not None, (metrics_line, timing_line)

# find the closing '},' of the metrics dict starting from timing_s
j = timing_line
while "}," not in lines[j]:
    j += 1

lines.insert(j + 1, '        metrics["baselines"] = baselines_out\n')
open("bot/__main__.py", "w", encoding="utf-8").write("".join(lines))
print("attached at line", j + 2)
