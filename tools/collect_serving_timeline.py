#!/usr/bin/env python3
import argparse
import datetime as dt
import posixpath
import shlex
import subprocess
import sys
import time


def run(cmd, *, check=True, capture_output=True):
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )


def docker_exec(container, script, *, check=True, capture_output=True):
    return run(["docker", "exec", container, "bash", "-lc", script], check=check, capture_output=capture_output)


def q(value):
    return shlex.quote(str(value))


def choose_port(container, preferred):
    script = f"""
python3 - <<'PY'
import socket
preferred = {int(preferred)}
for port in list(range(preferred, preferred + 100)) + [9001, 9002, 9003]:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        print(port)
        break
    except OSError:
        pass
    finally:
        s.close()
else:
    raise SystemExit("no free port found")
PY
"""
    return int(docker_exec(container, script).stdout.strip())


def wait_for_port(container, port, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        script = f"""
python3 - <<'PY'
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", {int(port)}))
    print("ready")
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
"""
        proc = docker_exec(container, script, check=False)
        if proc.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError(f"server did not listen on port {port} within {timeout_s}s")


def wait_for_runmeta(container, timeline_dir, timeout_s):
    deadline = time.time() + timeout_s
    last = []
    while time.time() < deadline:
        proc = docker_exec(
            container,
            f"find {q(timeline_dir)} -maxdepth 1 -type f -name '*.runmeta.pb' -print | sort",
            check=False,
        )
        if proc.returncode == 0:
            files = proc.stdout.strip().splitlines()
            if files:
                return files
            last = files
        time.sleep(1)
    return last


def container_path_to_host(path):
    if path.startswith("/workspace/"):
        return "/home/c00913906/" + path[len("/workspace/") :]
    return path


def resolve_binary_path(tensorflow_dir, binary):
    if binary.startswith("/"):
        return binary
    return posixpath.join(tensorflow_dir, binary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect one predictor_server TensorFlow timeline through the benchmark inference container."
    )
    parser.add_argument("--container", default="benchmark_infer_workspace")
    parser.add_argument("--model-name", default="din_mmoe")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--input-data-path", default=None)
    parser.add_argument("--tensorflow-dir", default="/workspace/tensorflow")
    parser.add_argument("--timeline-root", default="/workspace/timeline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of samples grouped into each client RPC request.",
    )
    parser.add_argument("--server-thread-num", type=int, default=16)
    parser.add_argument("--client-thread-num", type=int, default=1)
    parser.add_argument("--max-qps", type=int, default=2)
    parser.add_argument("--max-inflight", type=int, default=10)
    parser.add_argument("--warmup-duration-s", type=int, default=0)
    parser.add_argument("--test-duration-s", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--enable-kdnn", choices=["true", "false"], default="false")
    parser.add_argument("--timeline-every-n", type=int, default=1)
    parser.add_argument("--timeline-max-dumps", type=int, default=1)
    parser.add_argument("--server-bin", default="./bazel-bin/predictor_server")
    parser.add_argument("--client-bin", default="./bazel-bin/brpc_client")
    parser.add_argument("--convert-script", default="/workspace/tensorflow/tools/runmetadata_to_timeline.py")
    parser.add_argument("--startup-timeout-s", type=int, default=120)
    parser.add_argument("--timeline-wait-s", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path or f"/workspace/{args.model_name}/1"
    input_data_path = args.input_data_path or f"/workspace/dataset/{args.model_name}.tsv"
    run_name = args.run_name or f"batch_{args.batch_size}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timeline_dir = f"{args.timeline_root.rstrip('/')}/{args.model_name}/{run_name}"
    port = choose_port(args.container, args.port)
    server_bin = resolve_binary_path(args.tensorflow_dir, args.server_bin)
    client_bin = resolve_binary_path(args.tensorflow_dir, args.client_bin)

    print(f"container: {args.container}")
    print(f"model_path: {model_path}")
    print(f"input_data_path: {input_data_path}")
    print(f"port: {port}")
    print(f"timeline_dir: {timeline_dir}")

    docker_exec(
        args.container,
        f"mkdir -p {q(timeline_dir)} && test -x {q(server_bin)} && "
        f"test -x {q(client_bin)} && test -f {q(model_path + '/saved_model.pb')} && "
        f"test -f {q(input_data_path)}",
    )

    server_log = f"{timeline_dir}/server.log"
    client_log = f"{timeline_dir}/client.log"
    pid_file = f"{timeline_dir}/server.pid"

    server_cmd = (
        f"cd {q(args.tensorflow_dir)} && "
        f"TF_NUM_INTEROP_THREADS=16 TF_NUM_INTRAOP_THREADS=16 "
        f"{q(server_bin)} "
        f"--enable_kdnn={args.enable_kdnn} "
        f"--model_path={q(model_path)} "
        f"--thread_num={args.server_thread_num} "
        f"--port={port} "
        f"--enable_tf_timeline=true "
        f"--tf_timeline_every_n={args.timeline_every_n} "
        f"--tf_timeline_max_dumps={args.timeline_max_dumps} "
        f"--tf_timeline_dump_warmup=false "
        f"--tf_timeline_dir={q(timeline_dir)}"
    )

    start_script = (
        f"nohup bash -lc {q(server_cmd)} > {q(server_log)} 2>&1 & "
        f"echo $! > {q(pid_file)} && cat {q(pid_file)}"
    )
    pid = docker_exec(args.container, start_script).stdout.strip()
    print(f"server_pid: {pid}")

    try:
        wait_for_port(args.container, port, args.startup_timeout_s)

        client_cmd = (
            f"cd {q(args.tensorflow_dir)} && "
            f"{q(client_bin)} "
            f"--server=127.0.0.1:{port} "
            f"--input_data_path={q(input_data_path)} "
            f"--thread_num={args.client_thread_num} "
            f"--warmup_duration_s={args.warmup_duration_s} "
            f"--test_duration_s={args.test_duration_s} "
            f"--max_qps={args.max_qps} "
            f"--max_inflight={args.max_inflight} "
            f"--request_min_batch_size={args.batch_size} "
            f"--request_max_batch_size={args.batch_size} "
            f"--timeout_ms={args.timeout_ms}"
        )
        client = docker_exec(args.container, f"{client_cmd} 2>&1 | tee {q(client_log)}", check=False)
        sys.stdout.write(client.stdout)
        if client.returncode != 0:
            raise RuntimeError(f"client failed with exit code {client.returncode}")

        files = wait_for_runmeta(args.container, timeline_dir, args.timeline_wait_s)
        if not files:
            raise RuntimeError(f"no runmeta.pb generated under {timeline_dir}; see {server_log}")

        for runmeta in files:
            proc = docker_exec(args.container, f"python3 {q(args.convert_script)} {q(runmeta)}", check=False)
            sys.stdout.write(proc.stdout)
            if proc.returncode != 0:
                raise RuntimeError(f"timeline conversion failed for {runmeta}")

        print("timeline files:")
        listing = docker_exec(
            args.container,
            f"find {q(timeline_dir)} -maxdepth 1 -type f \\( -name '*.runmeta.pb' -o -name '*.json' -o -name '*.log' \\) -printf '%p %s\\n' | sort",
        ).stdout
        sys.stdout.write(listing)
        print(f"host_timeline_dir: {container_path_to_host(timeline_dir)}")
    finally:
        docker_exec(
            args.container,
            f"if test -f {q(pid_file)}; then kill $(cat {q(pid_file)}) >/dev/null 2>&1 || true; fi",
            check=False,
        )


if __name__ == "__main__":
    main()
