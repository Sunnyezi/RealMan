"""Collect disjoint chunks after the teacher gate, then expose one LeRobot-convertible directory."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from pipeline import ROOT, fingerprint, validate_episode, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--start-index", type=int, default=1000)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--stream", type=Path)
    args = parser.parse_args()
    if args.count <= 0 or not 1 <= args.workers <= 16 or args.start_index < 0:
        parser.error("Positive count, 1-16 workers, and nonnegative start index required")
    gate = json.loads(args.gate.read_text())
    version = fingerprint()
    if not (gate.get("fingerprint") == version and gate.get("episodes", 0) >= 100
            and gate.get("success_rate", 0) >= .9 and gate.get("split") == "train"):
        parser.error("Need matching >=90% teacher gate over >=100 trials")
    workers = min(args.workers, args.count)
    sizes = [args.count // workers + (i < args.count % workers) for i in range(workers)]
    stride = max(1000, 2 * math.ceil(args.count / workers) + 1)
    if args.start_index + workers * stride >= 90000:
        parser.error("Worker seed ranges exceed partition")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ | {"MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1",
                               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def worker(index: int, count: int) -> Path:
        dest = output / "chunks" / f"worker_{index:02d}"
        command = [sys.executable, str(ROOT / "pipeline.py"), "collect", "--output", str(dest),
                   "--split", args.split, "--count", str(count), "--max-attempts", str(2 * count),
                   "--start-index", str(args.start_index + index * stride), "--gate", str(args.gate.resolve())]
        if index == 0 and args.stream:
            command.extend(["--stream", str(args.stream.resolve())])
        with (output / f"worker_{index:02d}.log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        return dest

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(worker, index, count): index for index, count in enumerate(sizes)}
        for job in as_completed(jobs):
            job.result()
            print(f"completed worker {jobs[job]}", flush=True)
    sources = sorted((output / "chunks").glob("worker_*/episode_*"))
    seen = set()
    for index, path in enumerate(sources):
        meta = validate_episode(path)
        if (meta["seed"] in seen or meta["fingerprint"] != version or meta["split"] != args.split
                or not meta["is_success"]):
            raise RuntimeError("Invalid, duplicated, or mismatched demonstration")
        seen.add(meta["seed"])
        (output / f"episode_{index:06d}").symlink_to(path.relative_to(output), target_is_directory=True)
    if len(sources) != args.count:
        raise RuntimeError(f"Expected {args.count} episodes, got {len(sources)}")
    write_json(output / "summary.json", dict(episodes=len(sources), accepted=len(sources),
               requested=args.count, split=args.split, fingerprint=version, workers=workers,
               unique_seeds=len(seen), teacher_gate=str(args.gate.resolve()), complete=True))
    print(json.dumps(dict(complete=True, episodes=len(sources), unique_seeds=len(seen))), flush=True)


if __name__ == "__main__":
    main()
