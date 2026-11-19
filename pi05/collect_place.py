"""Gate, collect and replay pick-and-place teacher demonstrations (pi05/place_teacher.py); simulation only.

`gate` runs clean teacher episodes on train seeds without recording.  `collect` saves successful episodes
(two RGB images, measured joints, the executed absolute targets as labels) once a matching gate has reached
>=95% over >=100 episodes.  Both modes can fan out over worker processes; workers use disjoint seed blocks
and the parent merges them.  `replay` re-executes recorded labels and checks state agreement and success.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pipeline import BASES, validate_episode, write_json
from place_task import HORIZON, PLACE_PROMPT, PlaceEnv, place_fingerprint
from place_teacher import PHASES, PlaceTeacher

STRIDE = 1000


def run_episode(env: PlaceEnv, seed: int, record: bool, stage: str):
    teacher = PlaceTeacher()
    env.reset(seed)
    teacher.reset()
    e = env.base
    frames, phases, info = [], [], {"is_success": False}
    for _ in range(HORIZON):
        obs = env.observation() if record else None
        state = e.d.qpos[env.qids].astype(np.float32).copy()
        command = teacher(env).astype(np.float32)
        phases.append(teacher.phase)
        if record:
            frames.append({"image": obs["observation/image"], "wrist_image": obs["observation/wrist_image"],
                           "state": state, "actions": command, "timestamp": e.steps * .05})
        done, info = env.step(command)
        if env.stream is not None:
            env.publish(stage)
        if done:
            break
    return frames, phases, info, teacher.params()


def chunk(a: argparse.Namespace) -> None:
    """One worker: seeds start..start+attempts-1, stop after `count` successes in collect mode."""
    a.output.mkdir(parents=True, exist_ok=False)
    env = PlaceEnv(a.stream)
    accepted = 0
    try:
        with (a.output / "attempts.jsonl").open("w") as log:
            for i in range(a.attempts):
                seed = BASES["train"] + a.start_index + i
                started = time.monotonic()
                try:
                    frames, phases, info, params = run_episode(env, seed, a.mode == "collect",
                                                               f"pi05 place teacher / {a.mode}")
                except (ValueError, RuntimeError) as error:
                    frames, phases, params = [], [], PlaceTeacher().params()
                    info = {"is_success": False, "error": str(error)}
                row = dict(seed=seed, split="train", task="place", prompt=PLACE_PROMPT, fingerprint=place_fingerprint(),
                           teacher="pi05/place_teacher.py", teacher_params=params,
                           phase_frames={p: phases.count(p) for p in PHASES if p in phases},
                           failure=None if "place_milestones" not in info else PlaceEnv.failure(info),
                           wall_seconds=time.monotonic() - started, **info)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if a.mode == "collect" and row["is_success"]:
                    dest = a.output / f"episode_{i:06d}"
                    dest.mkdir()
                    np.savez_compressed(dest / "trajectory.npz",
                                        **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
                    write_json(dest / "metadata.json", row)
                    accepted += 1
                print(f"seed={seed} success={row['is_success']} failure={row['failure']} accepted={accepted}", flush=True)
                if a.mode == "collect" and accepted >= a.count:
                    break
    finally:
        env.close()


def fan_out(a: argparse.Namespace) -> None:
    if a.mode == "collect":
        gate = json.loads(a.gate.read_text()) if a.gate else {}
        if not (gate.get("fingerprint") == place_fingerprint() and gate.get("teacher") == PlaceTeacher().params()
                and gate.get("episodes", 0) >= 100 and gate.get("success_rate", 0) >= .95):
            raise SystemExit("Need a matching >=95% clean teacher gate over >=100 train episodes")
    a.output.mkdir(parents=True, exist_ok=False)
    chunks = a.output / "chunks"
    workers = []
    for w in range(a.workers):
        if a.mode == "gate":
            per = -(-a.count // a.workers)
            start, attempts, count = a.start_index + w * per, min(per, a.count - w * per), 0
        else:
            start, count = a.start_index + w * STRIDE, -(-a.count // a.workers)
            attempts = min(STRIDE, 2 * count)
        if attempts <= 0:
            continue
        if start + attempts > 90000:
            raise SystemExit("Seed range leaves the train partition")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--chunk", "--mode", a.mode,
               "--output", str(chunks / f"worker_{w:02d}"), "--start-index", str(start),
               "--attempts", str(attempts), "--count", str(max(count, 1))]
        if a.stream is not None and w == 0:
            cmd += ["--stream", str(a.stream)]
        log = (a.output / f"worker_{w:02d}.log").open("w")
        workers.append((w, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            env=os.environ | {"OMP_NUM_THREADS": "1"}), log))
    failed = []
    for w, proc, log in workers:
        if proc.wait() != 0:
            failed.append(w)
        log.close()
    if failed:
        raise SystemExit(f"Workers failed: {failed}")
    rows = [json.loads(line) for w, _, _ in workers
            for line in (chunks / f"worker_{w:02d}" / "attempts.jsonl").read_text().splitlines()]
    if len({r["seed"] for r in rows}) != len(rows):
        raise SystemExit("Duplicate seeds across workers")
    report = dict(mode=a.mode, episodes=len(rows), successes=sum(r["is_success"] for r in rows), split="train",
                  task="place", fingerprint=place_fingerprint(), teacher=PlaceTeacher().params(),
                  failures={k: sum(str(r["failure"]) == k for r in rows if not r["is_success"])
                            for k in sorted({str(r["failure"]) for r in rows if not r["is_success"]})},
                  steps_success=dict(zip(("min", "mean", "max"), _stats([r["phase_frames"] for r in rows if r["is_success"]]))))
    report["success_rate"] = report["successes"] / max(1, len(rows))
    if a.mode == "collect":
        episodes = sorted(p for w, _, _ in workers for p in (chunks / f"worker_{w:02d}").glob("episode_*"))[:]
        accepted = 0
        for path in episodes:
            if accepted >= a.count:
                break
            meta = validate_episode(path)
            if not (meta["is_success"] and meta["fingerprint"] == place_fingerprint() and meta["split"] == "train"):
                raise SystemExit(f"Invalid episode {path}")
            (a.output / f"episode_{accepted:06d}").symlink_to(path.resolve())
            accepted += 1
        report.update(accepted=accepted, requested=a.count, unique_seeds=accepted, complete=accepted == a.count)
        report["episodes_saved"] = accepted
    write_json(a.output / "summary.json", report)
    print(json.dumps(report), flush=True)
    if a.mode == "collect" and report["accepted"] < a.count:
        raise SystemExit(f"Only collected {report['accepted']}/{a.count}")


def _stats(phase_frames: list[dict]) -> tuple:
    lengths = [sum(f.values()) for f in phase_frames] or [0]
    return min(lengths), float(np.mean(lengths)), max(lengths)


def replay(a: argparse.Namespace) -> None:
    reports = []
    env = PlaceEnv(a.stream)
    try:
        for path in a.episodes:
            meta = validate_episode(path)
            env.reset(meta["seed"])
            errors = []
            with np.load(path / "trajectory.npz", allow_pickle=False) as data:
                for state, action in zip(data["state"], data["actions"], strict=True):
                    errors.append(float(np.max(np.abs(env.base.d.qpos[env.qids] - state))))
                    _, info = env.step(action)
            reports.append(dict(episode=str(path), max_state_error=max(errors), frames=len(errors),
                                is_success=info["is_success"]))
            print(json.dumps(reports[-1]), flush=True)
    finally:
        env.close()
    if any(r["max_state_error"] > 1e-4 or not r["is_success"] for r in reports):
        raise SystemExit("Replay diverged or failed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["gate", "collect", "replay"], required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gate", type=Path)
    p.add_argument("--stream", type=Path)
    p.add_argument("--chunk", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--attempts", type=int, help=argparse.SUPPRESS)
    p.add_argument("episodes", nargs="*", type=Path)
    a = p.parse_args()
    if a.mode == "replay":
        replay(a)
    elif a.chunk:
        chunk(a)
    else:
        if a.output is None or a.count <= 0 or a.workers <= 0 or a.start_index < 0:
            p.error("Need --output, positive --count/--workers and a non-negative --start-index")
        fan_out(a)


if __name__ == "__main__":
    main()
