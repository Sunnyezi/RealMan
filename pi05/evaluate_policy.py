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
    p.add_argument('--execute-steps', type=int, default=2, help='actions executed per model query')
    p.add_argument('--ensemble-decay', type=float, default=None,
                   help='ACT-style temporal ensembling of overlapping model chunks; weight exp(-decay*i), i=0 oldest')
    a = p.parse_args()
    if a.count<=0 or a.start_index<0 or a.start_index+a.count>90000:
        p.error('Invalid evaluation seed range')
    if not 1 <= a.execute_steps <= 10 or (a.ensemble_decay is not None and a.ensemble_decay < 0):
        p.error('Invalid execution settings')
    a.output.mkdir(parents=True, exist_ok=False)
    # Build the scene before connecting: slow scene loading must not stall websocket keepalive pings.
    env = RGBEnv()
    policy = WebsocketClientPolicy(port=a.port)
    if policy.get_server_metadata().get('action_dim') != 12:
        raise ValueError('Need action_dim=12')
    rows = []
    close_threshold = .25
    try:
        for i in range(a.count):
            seed = BASES[a.split]+a.start_index+i
            env.reset(seed)
            e = env.base
            frames, timings, history = [], [], []
            started = time.monotonic()
            while True:
                tick = time.perf_counter()
                chunk = np.asarray(policy.infer(env.observation())['actions'])
                timings.append((time.perf_counter()-tick)*1000)
                if chunk.ndim!=2 or chunk.shape[1]!=12 or len(chunk)<a.execute_steps or not np.isfinite(chunk).all():
                    raise ValueError('Invalid model action chunk')
                history = [(t0, c) for t0, c in history if t0+len(c) > e.steps] + [(e.steps, chunk)]
                for k in range(a.execute_steps):
                    raw = chunk[k]
                    if a.ensemble_decay is not None:
                        # Average every stored model prediction for this control step, oldest first.
                        preds = [c[e.steps-t0] for t0, c in history if 0 <= e.steps-t0 < len(c)]
                        weights = np.exp(-a.ensemble_decay*np.arange(len(preds)))
                        raw = np.average(preds, axis=0, weights=weights)
                    command = np.clip(raw, *e.m.actuator_ctrlrange[env.ids].T)
                    frame = dict(timestamp=e.steps*.05, state=e.d.qpos[env.qids].copy(),
                                 model_action=chunk[k].copy(), raw_action=raw.copy(), command=command.copy(), palm_position=e.d.xpos[e.bid].copy(),
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
            milestones = {
                'close_command': any(f['command'][8] >= close_threshold for f in frames),
                'two_hand_contacts': any(f['hand_contacts'] >= 2 for f in frames),
                'lift_10cm': any(f['height'] >= .10 for f in frames),
                'hold_2s': bool(info['is_success']),
            }
            row = dict(seed=seed, inference_ms=timings, steps=e.steps, sim_seconds=e.steps*.05,
                       milestones=milestones,
                       wall_seconds=time.monotonic()-started, **info)
            rows.append(row)
            write_json(dest.with_suffix('.json'), row)
            print(i, seed, info['is_success'], round(info['max_height'],4), flush=True)
        milestone_counts = {key: sum(row['milestones'][key] for row in rows) for key in rows[0]['milestones']}
        write_json(a.output/'summary.json', dict(episodes=len(rows), successes=sum(r['is_success'] for r in rows),
                   success_rate=float(np.mean([r['is_success'] for r in rows])), split=a.split,
                   fingerprint=fingerprint(), task=PROMPT, simulation_only=True,
                   milestone_counts=milestone_counts,
                   milestone_rates={key: value/len(rows) for key, value in milestone_counts.items()},
                   close_command_threshold_rad=close_threshold, close_command_actuator_index=8,
                   execution='pi05_policy_absolute12_no_takeover', execute_steps=a.execute_steps,
                   ensemble_decay=a.ensemble_decay, uses_controller_tactile=False,
                   uses_controller_rgbd_estimate=False, uses_controller_object_truth=False,
                   step_traces_saved=True, execution_fingerprint=hashlib.sha256(Path(__file__).read_bytes()+fingerprint().encode()).hexdigest()))
    finally:
        env.close()


if __name__ == '__main__':
    main()
