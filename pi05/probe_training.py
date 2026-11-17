"""Probe batches 1/2/4 with real weights and a visible demonstration window."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from pipeline import ROOT, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = args.output.name
    results = []
    for batch in (1, 2, 4):
        experiment = f"{prefix}_b{batch}"
        command = ["bash", str(ROOT / "train_visible.sh"), args.repo_id, experiment,
                   str(args.raw.resolve()), "1", str(batch)]
        log_path = args.output / f"batch_{batch}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT.parent, env=os.environ,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        results.append(dict(batch_size=batch, returncode=result.returncode, experiment=experiment))
        report = dict(trials=results, selected_batch_size=max(
            (r["batch_size"] for r in results if r["returncode"] == 0), default=None))
        write_json(args.output / "selection.json", report)
        print(json.dumps(results[-1]), flush=True)
        if result.returncode:
            message = log_path.read_text().lower()
            if not any(word in message for word in ("out of memory", "resource_exhausted", "cuda_error_out_of_memory")):
                raise RuntimeError(f"Non-OOM failure: {log_path}")
            break
    if not report["selected_batch_size"]:
        raise RuntimeError("No feasible batch size")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
