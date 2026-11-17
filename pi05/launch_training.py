"""Run training and final-checkpoint validation in tmux, independent of SSH sessions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from pipeline import ROOT, write_json


def worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    run = spec_path.parent
    status = {"stage": "training", "started_utc": datetime.now(timezone.utc).isoformat(), **spec}
    write_json(run / "job_status.json", status)
    try:
        with (run / "job.log").open("a") as log:
            subprocess.run(["bash", str(ROOT / "train_visible.sh"), spec["repo_id"], spec["experiment"],
                            spec["raw"], str(spec["steps"]), str(spec["batch_size"])],
                           cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
            status["stage"] = "validation"
            write_json(run / "job_status.json", status)
            checkpoint = ROOT / "checkpoints/pi05_realman_absolute12" / spec["experiment"] / str(spec["steps"] - 1)
            subprocess.run([sys.executable, str(ROOT / "evaluate_checkpoint.py"), "--repo-id", spec["repo_id"],
                            "--checkpoint", str(checkpoint), "--count", str(spec["validation_count"]),
                            "--output", str(run / "validation")], cwd=ROOT.parent,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status["stage"] = "trained_and_validated"
        status["validation"] = json.loads((run / "validation/results/summary.json").read_text())
    except Exception as error:
        status["failed_stage"] = status["stage"]
        status["stage"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(run / "job_status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--experiment")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--validation-count", type=int, default=100)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if (not args.repo_id or not args.experiment or not args.raw or not args.raw.is_dir()
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.experiment) or args.steps <= 0 or args.validation_count <= 0):
        parser.error("Need repo-id, safe experiment name, existing raw data and positive counts")
    run = ROOT / "runs" / args.experiment
    checkpoint = ROOT / "checkpoints/pi05_realman_absolute12" / args.experiment
    if checkpoint.exists():
        parser.error("Checkpoint directory already exists; refusing to overwrite")
    session = "pi05-" + args.experiment
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        parser.error("tmux session already exists")
    run.mkdir(parents=True, exist_ok=False)
    spec = {"repo_id": args.repo_id, "experiment": args.experiment, "raw": str(args.raw.resolve()),
            "steps": args.steps, "batch_size": args.batch_size,
            "validation_count": args.validation_count, "tmux_session": session}
    spec_path = run / "job_spec.json"
    write_json(spec_path, spec)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)])
    subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
    print(json.dumps(spec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
