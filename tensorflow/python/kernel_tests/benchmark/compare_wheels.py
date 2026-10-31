"""Installs two TensorFlow wheels, benchmarks them, and writes Markdown."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKER = _SCRIPT_DIR / "run_wheel_matmul.py"


def _run(command, **kwargs):
  return subprocess.run(command, check=True, text=True, **kwargs)


def _last_json_line(output):
  for line in reversed(output.splitlines()):
    try:
      return json.loads(line)
    except json.JSONDecodeError:
      pass
  raise RuntimeError("benchmark worker did not produce a JSON result")


def _install_wheel(python, wheel):
  # Retain the environment's dependencies.  Only TensorFlow itself is swapped
  # between the baseline and candidate runs.
  _run([python, "-m", "pip", "uninstall", "--yes", "tensorflow"])
  _run([
      python, "-m", "pip", "install", "--no-deps", wheel,
  ])


def _run_wheel(python, args):
  environment = os.environ.copy()
  environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
  environment["TF_ENABLE_KDNN_OPTS"] = "0"
  environment.pop("PYTHONPATH", None)
  command = [
      python,
      _WORKER,
      "--warmup", str(args.warmup),
      "--iterations", str(args.iterations),
      "--samples", str(args.samples),
      "--intra-threads", str(args.intra_threads),
  ]
  for name in args.case:
    command.extend(["--case", name])
  completed = _run(command, cwd=tempfile.gettempdir(), env=environment,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  return _last_json_line(completed.stdout)


def _format_us(value):
  return "%.3f" % value


def _write_markdown(path, baseline_label, baseline, candidate_label, candidate):
  candidate_by_name = {case["name"]: case for case in candidate["cases"]}
  lines = [
      "# MatMul Wheel Comparison",
      "",
      "KDNN is disabled for both processes (`TF_ENABLE_KDNN_OPTS=0`). "
      "Numbers are median wall time in microseconds per `tf.matmul`; lower is better.",
      "",
      "| Case | A shape | B shape | %s (us) | %s (us) | Speedup (%s/%s) |" % (
          baseline_label, candidate_label, baseline_label, candidate_label),
      "| --- | --- | --- | ---: | ---: | ---: |",
  ]
  for base in baseline["cases"]:
    comparison = candidate_by_name[base["name"]]
    speedup = base["median_us"] / comparison["median_us"]
    lines.append("| %s | %s | %s | %s | %s | %.3fx |" % (
        base["name"], base["shape_a"], base["shape_b"],
        _format_us(base["median_us"]), _format_us(comparison["median_us"]),
        speedup))
  lines.extend([
      "",
      "## Run Metadata",
      "",
      "| Wheel | TensorFlow version | Warmup | Timed iterations | Samples | Intra-op threads |",
      "| --- | --- | ---: | ---: | ---: | ---: |",
      "| %s | %s | %d | %d | %d | %d |" % (
          baseline_label, baseline["tensorflow_version"], baseline["warmup"],
          baseline["iterations"], baseline["samples"], baseline["intra_threads"]),
      "| %s | %s | %d | %d | %d | %d |" % (
          candidate_label, candidate["tensorflow_version"], candidate["warmup"],
          candidate["iterations"], candidate["samples"], candidate["intra_threads"]),
      "",
  ])
  path.write_text("\n".join(lines), encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(
      description="Compare two TensorFlow 2.20 wheels on logged MatMul shapes.")
  parser.add_argument(
      "--python", required=True, type=Path,
      help="Python executable for the dedicated benchmark environment.")
  parser.add_argument("--baseline-wheel", required=True, type=Path)
  parser.add_argument("--candidate-wheel", required=True, type=Path)
  parser.add_argument("--baseline-label", default="baseline")
  parser.add_argument("--candidate-label", default="candidate")
  parser.add_argument("--output-dir", type=Path,
                      default=Path("matmul_wheel_results"))
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--iterations", type=int, default=100)
  parser.add_argument("--samples", type=int, default=7)
  parser.add_argument("--intra-threads", type=int, default=16)
  parser.add_argument("--case", action="append", default=[])
  args = parser.parse_args()

  if args.baseline_wheel == args.candidate_wheel:
    raise ValueError("baseline-wheel and candidate-wheel must be different files")
  if not args.python.is_file():
    raise ValueError("Python executable does not exist: %s" % args.python)
  for wheel in (args.baseline_wheel, args.candidate_wheel):
    if not wheel.is_file() or wheel.suffix != ".whl":
      raise ValueError("wheel does not exist: %s" % wheel)

  args.output_dir.mkdir(parents=True, exist_ok=True)
  _install_wheel(args.python, args.baseline_wheel)
  baseline = _run_wheel(args.python, args)
  _install_wheel(args.python, args.candidate_wheel)
  candidate = _run_wheel(args.python, args)

  (args.output_dir / "baseline.json").write_text(
      json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (args.output_dir / "candidate.json").write_text(
      json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  _write_markdown(args.output_dir / "comparison.md", args.baseline_label,
                  baseline, args.candidate_label, candidate)
  print(args.output_dir / "comparison.md")


if __name__ == "__main__":
  main()
