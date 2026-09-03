#!/usr/bin/env python3
import argparse
import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow RunMetadata protobuf to Chrome trace JSON."
    )
    parser.add_argument("input", help="Input .runmeta.pb file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .json file. Defaults to replacing .runmeta.pb with .json.",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Include memory counters in the generated Chrome trace.",
    )
    return parser.parse_args()


def default_output_path(input_path):
    suffix = ".runmeta.pb"
    if input_path.endswith(suffix):
        return input_path[: -len(suffix)] + ".json"
    return input_path + ".json"


def main():
    args = parse_args()
    output_path = args.output or default_output_path(args.input)

    run_metadata = config_pb2.RunMetadata()
    with open(args.input, "rb") as f:
        run_metadata.ParseFromString(f.read())

    trace = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = trace.generate_chrome_trace_format(
        show_memory=args.show_memory
    )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        f.write(chrome_trace)

    print(output_path)


if __name__ == "__main__":
    main()
