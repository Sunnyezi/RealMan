"""Evaluate pi0.5 plus RGB-D alignment/tactile stabilization, with step diagnostics."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from pipeline import BASES, GRASP, PROMPT, RGBEnv, fingerprint, write_json
from stabilizer import RobotState, Settings, Stabilizer


def make_controller(env: RGBEnv) -> Stabilizer:
    sys.path.insert(0, str(GRASP))
    from build_scene import solve

    e = env.base
    fk_data = env.mujoco.MjData(e.m)
    ik_data = env.mujoco.MjData(e.m)
    arm_q = env.qids[:6]

    def forward(q: np.ndarray) -> np.ndarray:
        fk_data.qpos[arm_q] = q
        env.mujoco.mj_forward(e.m, fk_data)
        return fk_data.xpos[e.bid].copy()

    def inverse(point: np.ndarray, rotation: np.ndarray, seed: np.ndarray) -> np.ndarray:
        return solve(e.m, ik_data, point, rotation, seed)

    closed = np.asarray(e.cfg["hand_target"], dtype=float).copy()
    closed[1:3] += .12
    if not e.vision.visible or e.d.time-e.vision.last_seen > .15:
        raise ValueError("Need a fresh RGB-D grasp estimate before executing")
    # Camera-derived estimate plus fixed hand-to-cube calibration; never read object qpos/xpos.
    visual_grasp = e.vision.last.copy() + np.array(e.cfg["grasp_xyz"]) - np.array(e.cfg["object_xyz"])
    return Stabilizer(e.d.ctrl[env.ids], e.m.actuator_ctrlrange[env.ids], closed,
                      np.array(e.cfg["rotation"]), forward, inverse, visual_grasp, Settings())


def robot_state(env: RGBEnv, thumb_ids: np.ndarray, index_ids: np.ndarray) -> RobotState:
    e = env.base
    return RobotState(e.d.qpos[env.qids].copy(), e.d.xpos[e.bid].copy(),
                      float(np.sum(e.touch_force[thumb_ids])), float(np.sum(e.touch_force[index_ids])))


def main() -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", choices=BASES, default="validation")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", type=Path)
    args = parser.parse_args()
    if args.count <= 0 or args.start_index < 0 or args.start_index+args.count > 90000:
        parser.error("Invalid episode count or seed range")
    args.output.mkdir(parents=True, exist_ok=False)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    if policy.get_server_metadata().get("action_dim") != 12:
        raise ValueError("Policy server does not advertise twelve action channels")
    env = RGBEnv()
    regions = json.loads((GRASP / "sensor_map.json").read_text())
    thumb = np.array([i for i, row in enumerate(regions) if row["region"].startswith("right_thumb_")])
    index = np.array([i for i, row in enumerate(regions) if row["region"].startswith("right_index_")])
    if not len(thumb) or not len(index):
        raise ValueError("Missing thumb/index tactile mapping")
    results = []
    try:
        for i in range(args.count):
            seed = BASES[args.split]+args.start_index+i
            env.reset(seed)
            e = env.base
            controller = make_controller(env)
            write_json(args.output / "controller.json", controller.describe())
            timings = []
            trace = []
            chunk = None
            cursor = 0
            started = time.monotonic()
            while True:
                raw = None
                if controller.needs_policy:
                    if chunk is None or cursor >= 2:
                        tick = time.perf_counter()
                        chunk = np.asarray(policy.infer(env.observation())["actions"])
                        timings.append(1000*(time.perf_counter()-tick))
                        if chunk.ndim != 2 or chunk.shape[1] != 12 or len(chunk) < 2 or not np.isfinite(chunk).all():
                            raise ValueError("Invalid policy action chunk")
                        cursor = 0
                    raw = chunk[cursor].copy()
                    cursor += 1
                state = robot_state(env, thumb, index)
                previous_phase = controller.phase
                command = controller.update(state, raw)
                if previous_phase != "policy" and controller.needs_policy:
                    chunk, cursor = None, 0
                frame = {"timestamp": e.steps*.05, "state": state.joints.copy(),
                         "joint_velocity": e.d.qvel[e.m.jnt_dofadr[e.m.actuator_trnid[env.ids, 0]]].copy(),
                         "raw_action": raw if raw is not None else np.full(12, np.nan),
                         "has_policy_action": raw is not None, "command": command.copy(),
                         "palm_position": state.palm_position.copy(),
                         "palm_rotation": e.d.xmat[e.bid].reshape(3, 3).copy(),
                         "tactile": e.touch_force.copy(), "phase": controller.phase}
                done, info = env.step(command)
                # Evaluator-only truth is recorded after controller.update; it never enters RobotState.
                frame.update(post_timestamp=e.steps*.05, object_position=e.d.xpos[e.ob].copy(),
                             height=info["height"], hold_seconds=info["hold_seconds"],
                             hand_contacts=info["hand_contacts"], is_success=info["is_success"])
                trace.append(frame)
                if args.stream:
                    env.stream = args.stream
                    env.publish("pi05 + RGBD/tactile controller / " + controller.phase)
                    env.stream = None
                if done:
                    break
            episode = args.output / f"episode_{i:06d}"
            np.savez_compressed(episode.with_suffix(".npz"),
                                **{key: np.stack([f[key] for f in trace]) for key in trace[0]})
            row = dict(seed=seed, inference_ms=timings, steps=e.steps, sim_seconds=e.steps*.05,
                       wall_seconds=time.monotonic()-started, controller_phase=controller.phase,
                       controller_transitions=controller.transitions, ik_projection_rejections=controller.ik_rejections,
                       visual_grasp_m=controller.visual_grasp.tolist(), visual_estimate_m=e.vision.last.tolist(),
                       phase_counts=dict(Counter(f["phase"] for f in trace)), **info)
            write_json(episode.with_suffix(".json"), row)
            results.append(row)
            print(i, seed, info["is_success"], controller.phase, round(info["max_height"],4), flush=True)
        write_json(args.output / "summary.json", {
            "episodes": len(results), "successes": sum(r["is_success"] for r in results),
            "success_rate": float(np.mean([r["is_success"] for r in results])), "split": args.split,
            "fingerprint": fingerprint(), "task": PROMPT, "simulation_only": True,
            "execution": "pi05_rgbd_align_tactile_hold_v4",
            "execution_fingerprint": hashlib.sha256(Path(__file__).read_bytes() +
                Path(__file__).with_name("stabilizer.py").read_bytes() + fingerprint().encode()).hexdigest(),
            "uses_controller_tactile": True,
            "uses_controller_rgbd_estimate": True,
            "uses_controller_object_truth": False, "step_traces_saved": True})
    finally:
        env.close()


if __name__ == "__main__":
    main()
