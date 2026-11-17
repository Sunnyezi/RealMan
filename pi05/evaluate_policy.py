"""Evaluate the learned policy for the full task, without geometric/tactile takeover."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from pipeline import BASES, PROMPT, RGBEnv, fingerprint, write_json


def main() -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--split', choices=['validation','test'], default='validation')
    p.add_argument('--start-index', type=int, default=10000)
    p.add_argument('--port', type=int, default=8095)
    p.add_argument('--stream', type=Path)
    a = p.parse_args()
    if a.count<=0 or a.start_index<0 or a.start_index+a.count>90000:
        p.error('Invalid evaluation seed range')
    a.output.mkdir(parents=True, exist_ok=False)
    policy = WebsocketClientPolicy(port=a.port)
    if policy.get_server_metadata().get('action_dim') != 12:
        raise ValueError('Need action_dim=12')
    env = RGBEnv()
    rows = []
    try:
        for i in range(a.count):
            seed = BASES[a.split]+a.start_index+i
            env.reset(seed)
            e = env.base
            frames, timings = [], []
            started = time.monotonic()
            while True:
                tick = time.perf_counter()
                chunk = np.asarray(policy.infer(env.observation())['actions'])
                timings.append((time.perf_counter()-tick)*1000)
                if chunk.ndim!=2 or chunk.shape[1]!=12 or len(chunk)<2 or not np.isfinite(chunk).all():
                    raise ValueError('Invalid model action chunk')
                for raw in chunk[:2]:
                    command = np.clip(raw, *e.m.actuator_ctrlrange[env.ids].T)
                    frame = dict(timestamp=e.steps*.05, state=e.d.qpos[env.qids].copy(),
                                 raw_action=raw.copy(), command=command.copy(), palm_position=e.d.xpos[e.bid].copy(),
                                 tactile=e.touch_force.copy())
                    done, info = env.step(command)
                    frame.update(height=info['height'], hold_seconds=info['hold_seconds'],
                                 object_position=e.d.xpos[e.ob].copy(), hand_contacts=info['hand_contacts'],
                                 is_success=info['is_success'])
                    frames.append(frame)
                    if a.stream:
                        env.stream = a.stream
                        env.publish('pi05 learned policy / full task / no takeover')
                        env.stream = None
                    if done:
                        break
                if done:
                    break
            dest = a.output/f'episode_{i:06d}'
            np.savez_compressed(dest.with_suffix('.npz'), **{k:np.stack([f[k] for f in frames]) for k in frames[0]})
            row = dict(seed=seed, inference_ms=timings, steps=e.steps, sim_seconds=e.steps*.05,
                       wall_seconds=time.monotonic()-started, **info)
            rows.append(row)
            write_json(dest.with_suffix('.json'), row)
            print(i, seed, info['is_success'], round(info['max_height'],4), flush=True)
        write_json(a.output/'summary.json', dict(episodes=len(rows), successes=sum(r['is_success'] for r in rows),
                   success_rate=float(np.mean([r['is_success'] for r in rows])), split=a.split,
                   fingerprint=fingerprint(), task=PROMPT, simulation_only=True,
                   execution='pi05_policy_absolute12_no_takeover', uses_controller_tactile=False,
                   uses_controller_rgbd_estimate=False, uses_controller_object_truth=False,
                   step_traces_saved=True, execution_fingerprint=hashlib.sha256(Path(__file__).read_bytes()+fingerprint().encode()).hexdigest()))
    finally:
        env.close()


if __name__ == '__main__':
    main()
