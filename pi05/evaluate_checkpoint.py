"""Serve one checkpoint and evaluate its closed-loop policy with a visible VNC window."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import numpy as np

from pipeline import ROOT, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    if args.count <= 0 or not args.checkpoint.is_dir() or args.output.exists():
        parser.error("Need an existing checkpoint, positive count and unused output directory")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    stream = output / "live_state.npz"
    environment = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                               "PYTHONUNBUFFERED": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30"}
    environment.pop("JAX_PLATFORMS", None)
    processes = []
    with (output / "server.log").open("w") as server_log, (output / "viewer.log").open("w") as viewer_log:
        try:
            server = subprocess.Popen([
                str(ROOT / "vendor/openpi/.venv/bin/python"), str(ROOT / "openpi_run.py"), "serve",
                "--repo-id", args.repo_id, "--checkpoint", str(args.checkpoint.resolve()),
                "--port", str(args.port)], cwd=ROOT.parent, env=environment,
                stdout=server_log, stderr=subprocess.STDOUT)
            processes.append(server)
            viewer = subprocess.Popen(["bash", str(ROOT.parent / "grasp/grasp.sh"), "live", "--envs", "1",
                                       "--stream", str(stream)], cwd=ROOT.parent,
                                      stdout=viewer_log, stderr=subprocess.STDOUT)
            processes.append(viewer)
            for _ in range(240):
                if server.poll() is not None or viewer.poll() is not None:
                    raise RuntimeError("Server or viewer stopped; inspect logs")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("Policy server did not become ready")
            with (output / "evaluation.log").open("w") as evaluation_log:
                started = time.monotonic()
                subprocess.run(["bash", str(ROOT / "evaluate.sh"), "--split", args.split,
                                "--count", str(args.count), "--port", str(args.port),
                                "--output", str(output / "results"), "--stream", str(stream)],
                               cwd=ROOT.parent, env=environment, stdout=evaluation_log,
                               stderr=subprocess.STDOUT, check=True)
            summary_path = output / "results/summary.json"
            summary = json.loads(summary_path.read_text())
            rows = [json.loads(path.read_text()) for path in sorted((output / "results").glob("episode_*.json"))]
            timings = [value for row in rows for value in row["inference_ms"]]
            n, p, z = summary["episodes"], summary["success_rate"], 1.959963984540054
            center = (p + z*z/(2*n)) / (1 + z*z/n)
            half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
            failures = Counter()
            for row in rows:
                if not row["is_success"]:
                    reason = ("physics" if row.get("physics_failure") or row.get("warnings") else
                              "ground_contact" if row["table_collision"] else
                              "dropped" if row["dropped"] else "timeout_or_incomplete")
                    failures[reason] += 1
            summary.update(success_95ci=[max(0., center-half), min(1., center+half)],
                           inference_mean_ms=float(np.mean(timings)), inference_p95_ms=float(np.percentile(timings, 95)),
                           inference_median_ms=float(np.median(timings)), failures=dict(failures),
                           evaluation_wall_seconds=time.monotonic()-started,
                           checkpoint=str(args.checkpoint.resolve()))
            write_json(summary_path, summary)
            write_json(output / "checkpoint.json", {"checkpoint": str(args.checkpoint.resolve()),
                       "repo_id": args.repo_id, "split": args.split, "episodes": args.count})
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
