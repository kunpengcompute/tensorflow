"""Summarizes MatMul wheel-comparison Markdown by production model."""

import argparse
from collections import defaultdict
from pathlib import Path


_GROUPS = ("h", "a", "p", "c")


def _parse_rows(path):
  rows = []
  in_results_table = False
  for line in path.read_text(encoding="utf-8").splitlines():
    if line.startswith("| Case | A shape | B shape |"):
      in_results_table = True
      continue
    if not in_results_table:
      continue
    if not line.startswith("|"):
      break
    if line.startswith("| ---"):
      continue
    fields = [field.strip() for field in line.strip().strip("|").split("|")]
    if len(fields) != 6:
      raise ValueError("invalid result row: %s" % line)
    name, unused_shape_a, unused_shape_b, baseline, candidate, unused_speedup = fields
    try:
      rows.append((name, float(baseline), float(candidate)))
    except ValueError as error:
      raise ValueError("invalid timing row: %s" % line) from error
  if not rows:
    raise ValueError("no MatMul result rows found in %s" % path)
  return rows


def _summarize(rows):
  totals = defaultdict(lambda: [0.0, 0.0])
  for name, baseline_us, candidate_us in rows:
    group = name.split("_", 1)[0]
    if group not in _GROUPS:
      raise ValueError("unrecognized model prefix in case: %s" % name)
    totals["all"][0] += baseline_us
    totals["all"][1] += candidate_us
    totals[group][0] += baseline_us
    totals[group][1] += candidate_us
  return totals


def _write_summary(path, source, totals):
  lines = [
      "# MatMul Performance Summary",
      "",
      "Source: `%s`" % source,
      "",
      "Each value is the sum of the per-case median times in microseconds. "
      "Lower time is better.",
      "",
      "| Group | baseline (us) | candidate (us) | Speedup (baseline/candidate) |",
      "| --- | ---: | ---: | ---: |",
  ]
  for group in ("all",) + _GROUPS:
    baseline_us, candidate_us = totals[group]
    if candidate_us == 0.0:
      # A partial run may not cover every model group.
      continue
    lines.append("| %s | %.3f | %.3f | %.3fx |" % (
        group, baseline_us, candidate_us, baseline_us / candidate_us))
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(
      description="Summarize MatMul timings from comparison.md.")
  parser.add_argument("comparison", type=Path,
                      help="Path to compare_wheels.py output comparison.md.")
  parser.add_argument("--output", type=Path,
                      help="Output Markdown path; defaults beside comparison.md.")
  args = parser.parse_args()

  if not args.comparison.is_file():
    raise ValueError("comparison file does not exist: %s" % args.comparison)
  output = args.output or args.comparison.with_name("summary.md")
  _write_summary(output, args.comparison, _summarize(_parse_rows(args.comparison)))
  print(output)


if __name__ == "__main__":
  main()
