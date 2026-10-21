"""Runs the logged MatMul shapes against one installed TensorFlow wheel.

CHANGE vs original: operands are materialised as tensors ONCE (tf.constant)
instead of being passed as NumPy arrays, so the NumPy -> Tensor conversion is
no longer inside the timed loop.  CLI, JSON schema and the kdnn switch are
unchanged.
"""

import argparse
import json
import statistics
import time

import numpy as np

from matmul_cases import MODEL_SHAPES


def _percentile(values, percentile):
  ordered = sorted(values)
  index = round((len(ordered) - 1) * percentile)
  return ordered[index]


def _to_numpy(x):
  return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _run_case(tf, case, warmup, iterations, samples):
  name, shape_a, shape_b, transpose_a, transpose_b = case
  rng = np.random.RandomState(20260826)
  a_np = rng.normal(0.0, 0.1, size=shape_a).astype(np.float32)
  b_np = rng.normal(0.0, 0.1, size=shape_b).astype(np.float32)

  # NumPy reference (unchanged).
  expected = np.matmul(
      np.swapaxes(a_np, -1, -2) if transpose_a else a_np,
      np.swapaxes(b_np, -1, -2) if transpose_b else b_np)

  # Materialise operands as tensors ONCE -- no conversion inside the loop.
  a = tf.constant(a_np)
  b = tf.constant(b_np)

  def matmul():
    return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)

  result = matmul()
  if not np.allclose(_to_numpy(result), expected, rtol=1e-4, atol=1e-5):
    raise RuntimeError("%s produced a result different from NumPy" % name)

  for _ in range(warmup):
    matmul()

  samples_us = []
  for _ in range(samples):
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
      matmul()
    samples_us.append((time.perf_counter_ns() - start_ns) / iterations / 1000.0)

  return {
      "name": name,
      "shape_a": list(shape_a),
      "shape_b": list(shape_b),
      "transpose_a": transpose_a,
      "transpose_b": transpose_b,
      "samples_us": samples_us,
      "mean_us": statistics.fmean(samples_us),
      "median_us": statistics.median(samples_us),
      "p95_us": _percentile(samples_us, 0.95),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--warmup", type=int, required=True)
  parser.add_argument("--iterations", type=int, required=True)
  parser.add_argument("--samples", type=int, required=True)
  parser.add_argument("--intra-threads", type=int, required=True)
  parser.add_argument("--case", action="append", dest="case_names")
  args = parser.parse_args()

  if min(args.warmup, args.iterations, args.samples, args.intra_threads) < 1:
    raise ValueError("warmup, iterations, samples, and intra-threads must be positive")

  # TensorFlow must be configured before the first op is executed.
  import tensorflow as tf  # pylint: disable=g-import-not-at-top
  tf.config.threading.set_intra_op_parallelism_threads(args.intra_threads)
  tf.config.threading.set_inter_op_parallelism_threads(1)

  cases = MODEL_SHAPES
  if args.case_names:
    selected = set(args.case_names)
    cases = tuple(case for case in cases if case[0] in selected)
    missing = selected.difference(case[0] for case in cases)
    if missing:
      raise ValueError("unknown case(s): %s" % ", ".join(sorted(missing)))

  report = {
      "tensorflow_version": tf.__version__,
      "warmup": args.warmup,
      "iterations": args.iterations,
      "samples": args.samples,
      "intra_threads": args.intra_threads,
      "kdnn_enabled": False,
      "operands": "tf.constant",
      "cases": [
          _run_case(tf, case, args.warmup, args.iterations, args.samples)
          for case in cases
      ],
  }
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
