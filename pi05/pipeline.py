"""RGB imitation pipeline for the existing server contact scene; simulation only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
GRASP = ROOT.parent / "grasp"
PROMPT = "Pick up the blue cube and hold it steadily above the ground."
BASES = {"train": 10000, "validation": 100000, "test": 200000}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in [GRASP / n for n in ("scene.xml", "config.json", "env.py", "scripted.py", "planner.py", "vision.py")]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


class RGBEnv:
    """Expose images and measured joints, execute absolute targets without the old planner."""

    def __init__(self, stream: Path | None = None):
        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.path.insert(0, str(GRASP))
        import mujoco
        from env import GraspEnv

        self.mujoco = mujoco
        self.base = GraspEnv(randomize=True)
        self.ids = np.r_[self.base.arm, self.base.hand]
        self.qids = self.base.m.jnt_qposadr[self.base.m.actuator_trnid[self.ids, 0]]
        self.renderer = mujoco.Renderer(self.base.m, height=224, width=224)
        self.stream = stream

    def reset(self, seed: int) -> None:
        self.base.reset(seed=seed)
        self.publish("pi05 reset")

    def observation(self) -> dict:
        e = self.base
        images = []
        option = self.mujoco.MjvOption()
        option.geomgroup[3] = 0
        for camera in ("overview", "rgbd"):
            self.renderer.update_scene(e.d, camera=camera, scene_option=option)
            images.append(self.renderer.render().copy())
        return {"observation/image": images[0], "observation/wrist_image": images[1],
                "observation/state": e.d.qpos[self.qids].astype(np.float32), "prompt": PROMPT}

    def step(self, action: np.ndarray) -> tuple[bool, dict]:
        e = self.base
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Expected 12 finite absolute joint targets in radians")
        e.d.ctrl[self.ids] = np.clip(action, *e.m.actuator_ctrlrange[self.ids].T)
        force = np.zeros(17)
        for _ in range(25):
            self.mujoco.mj_step(e.m, e.d)
            force += e.d.sensordata[e.sensor_adrs]
        e._update_touch(force / 25)
        e.steps += 1
        info = e._metrics()
        bad = bool(info["warnings"] or not np.isfinite(e.d.qpos).all())
        info["physics_failure"] = bad
        done = bool(info["is_success"] or info["dropped"] or bad or e.steps >= e.cfg["horizon"])
        self.publish("pi05 policy / absolute targets")
        return done, info

    def publish(self, stage: str) -> None:
        if self.stream is None:
            return
        self.stream.parent.mkdir(parents=True, exist_ok=True)
        e = self.base
        temp = self.stream.with_name(f".stream_{os.getpid()}.npz")
        np.savez(temp, qpos=e.d.qpos, qvel=e.d.qvel, ctrl=e.d.ctrl, time=e.d.time,
                 touch_force=e.touch_force, stage=stage)
        os.replace(temp, self.stream)

    def close(self) -> None:
        self.renderer.close()
        self.base.close()


def collect(args: argparse.Namespace) -> None:
    from scripted import action as teacher_action

    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "collect" and args.count > 50:
        gate = json.loads(args.gate.read_text()) if args.gate else {}
        if not (gate.get("fingerprint") == fingerprint() and gate.get("episodes", 0) >= 100
                and gate.get("success_rate", 0) >= 0.9 and gate.get("split") == "train"):
            raise ValueError("Need >=90% success over >=100 matching teacher trials before bulk collection")
    env = RGBEnv(args.stream)
    results = []
    accepted = 0
    try:
        with (args.output / "attempts.jsonl").open("w") as log:
            for index in range(args.max_attempts if args.mode == "collect" else args.count):
                seed = BASES[args.split] + args.start_index + index
                frames = []
                info = {"is_success": False}
                started = time.monotonic()
                try:
                    env.reset(seed)
                    e = env.base
                    for _ in range(e.cfg["horizon"]):
                        obs = env.observation() if args.mode == "collect" else None
                        context = e._advance(teacher_action(e))
                        target = e.d.ctrl[env.ids].astype(np.float32).copy()
                        if obs is not None:
                            frames.append({"image": obs["observation/image"],
                                           "wrist_image": obs["observation/wrist_image"],
                                           "state": obs["observation/state"], "actions": target,
                                           "timestamp": e.steps * 0.05})
                        _, _, term, trunc, info = e._finish(context)
                        env.publish("pi05 teacher / data collection")
                        if term or trunc:
                            break
                except (ValueError, RuntimeError) as error:
                    info = {"is_success": False, "error": str(error)}
                row = dict(seed=seed, split=args.split, prompt=PROMPT, fingerprint=fingerprint(),
                           wall_seconds=time.monotonic() - started, **info)
                results.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if args.mode == "collect" and row["is_success"]:
                    dest = args.output / f"episode_{index:06d}"
                    dest.mkdir()
                    np.savez_compressed(dest / "trajectory.npz", **{
                        key: np.stack([frame[key] for frame in frames]) for key in frames[0]})
                    write_json(dest / "metadata.json", row)
                    accepted += 1
                print(f"seed={seed} success={row['is_success']} accepted={accepted}", flush=True)
                if args.mode == "collect" and accepted >= args.count:
                    break
    finally:
        env.close()
    report = dict(episodes=len(results), successes=sum(r["is_success"] for r in results),
                  accepted=accepted, requested=args.count, split=args.split, fingerprint=fingerprint())
    report["success_rate"] = report["successes"] / max(1, len(results))
    write_json(args.output / "summary.json", report)
    if args.mode == "collect" and accepted < args.count:
        raise RuntimeError(f"Only collected {accepted}/{args.count} successful episodes")


def validate_episode(path: Path) -> dict:
    meta = json.loads((path / "metadata.json").read_text())
    with np.load(path / "trajectory.npz", allow_pickle=False) as data:
        n = len(data["state"])
        if n < 10:
            raise ValueError("Episode shorter than action horizon")
        for key in ("state", "actions"):
            if data[key].shape != (n, 12) or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid {key}")
        for key in ("image", "wrist_image"):
            if data[key].shape != (n, 224, 224, 3) or data[key].dtype != np.uint8:
                raise ValueError(f"Invalid {key}")
        if not np.isfinite(data["timestamp"]).all() or not np.allclose(data["timestamp"], np.arange(n) * .05):
            raise ValueError("Invalid 20 Hz timestamps")
    return meta


def convert(args: argparse.Namespace) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    paths = sorted(args.raw.glob("episode_*"))
    metas = [validate_episode(p) for p in paths]
    if not metas or len({m["split"] for m in metas}) != 1 or metas[0]["split"] == "test":
        raise ValueError("Convert one nonempty train or validation partition only")
    if not all(m["is_success"] for m in metas) or len({m["fingerprint"] for m in metas}) != 1:
        raise ValueError("Only successful episodes from a single pipeline version may be converted")
    if (HF_LEROBOT_HOME / args.repo_id).exists():
        raise FileExistsError(args.repo_id)
    dataset = LeRobotDataset.create(repo_id=args.repo_id, robot_type="rm65_rh56_absolute12", fps=20,
        features={"image": {"dtype": "image", "shape": (224,224,3), "names": ["height","width","channel"]},
                  "wrist_image": {"dtype": "image", "shape": (224,224,3), "names": ["height","width","channel"]},
                  "state": {"dtype": "float32", "shape": (12,), "names": ["state"]},
                  "actions": {"dtype": "float32", "shape": (12,), "names": ["actions"]}},
        image_writer_threads=2, image_writer_processes=0)
    for path, meta in zip(paths, metas, strict=True):
        with np.load(path / "trajectory.npz", allow_pickle=False) as data:
            for i in range(len(data["state"])):
                dataset.add_frame({key: data[key][i] for key in ("image","wrist_image","state","actions")}
                                  | {"task": meta["prompt"]})
        dataset.save_episode()
    write_json(ROOT / "outputs" / (args.repo_id.replace("/", "_") + "_conversion.json"),
               dict(repo_id=args.repo_id, raw=str(args.raw.resolve()), episodes=len(paths),
                    fingerprint=metas[0]["fingerprint"], split=metas[0]["split"]))


def replay(args: argparse.Namespace) -> None:
    validate_episode(args.episode)
    meta = json.loads((args.episode / "metadata.json").read_text())
    env = RGBEnv(args.stream)
    errors = []
    try:
        env.reset(meta["seed"])
        with np.load(args.episode / "trajectory.npz", allow_pickle=False) as data:
            for state, action in zip(data["state"], data["actions"], strict=True):
                errors.append(float(np.max(np.abs(env.base.d.qpos[env.qids] - state))))
                _, info = env.step(action)
        report = dict(max_state_error=max(errors), frames=len(errors), is_success=info["is_success"])
        print(json.dumps(report), flush=True)
        write_json(args.episode / "replay.json", report)
        if max(errors) > 1e-4 or not info["is_success"]:
            raise RuntimeError("Absolute-action replay diverged or grasp failed")
    finally:
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    env = RGBEnv(args.stream)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for i in range(args.count):
            env.reset(BASES[args.split] + args.start_index + i)
            timings = []
            done = False
            while not done:
                tick = time.perf_counter()
                chunk = np.asarray(policy.infer(env.observation())["actions"])
                timings.append(1000 * (time.perf_counter() - tick))
                if chunk.ndim != 2 or chunk.shape[1] != 12 or len(chunk) < 2:
                    raise ValueError(f"Invalid action chunk: {chunk.shape}")
                for action in chunk[:2]:
                    done, info = env.step(action)
                    if done:
                        break
            row = dict(seed=BASES[args.split] + args.start_index + i, inference_ms=timings, **info)
            results.append(row)
            write_json(args.output / f"episode_{i:06d}.json", row)
            print(i, info["is_success"], flush=True)
    finally:
        env.close()
    write_json(args.output / "summary.json", dict(episodes=len(results), successes=sum(r["is_success"] for r in results),
               success_rate=float(np.mean([r["is_success"] for r in results])), split=args.split,
               fingerprint=fingerprint(), task=PROMPT, simulation_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    for mode in ("check", "collect", "evaluate"):
        p = subs.add_parser(mode)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--split", choices=BASES, default="train" if mode != "evaluate" else "validation")
        p.add_argument("--count", type=int, default=10)
        p.add_argument("--start-index", type=int, default=0)
        p.add_argument("--stream", type=Path)
        if mode == "collect":
            p.add_argument("--max-attempts", type=int, default=1000)
            p.add_argument("--gate", type=Path)
        if mode == "evaluate":
            p.add_argument("--host", default="localhost")
            p.add_argument("--port", type=int, default=8000)
    p = subs.add_parser("convert")
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p = subs.add_parser("replay")
    p.add_argument("episode", type=Path)
    p.add_argument("--stream", type=Path)
    args = parser.parse_args()
    if hasattr(args, "count"):
        attempts = getattr(args, "max_attempts", args.count)
        if args.count <= 0 or args.start_index < 0 or attempts <= 0 or args.start_index + attempts > 90000:
            parser.error("Positive counts and indices within the disjoint 90000-seed partition are required")
        if args.mode == "collect" and args.split == "test":
            parser.error("Test data cannot be collected for training")
    if args.mode in ("check", "collect"):
        sys.path.insert(0, str(GRASP))
        collect(args)
    else:
        {"convert": convert, "replay": replay, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
