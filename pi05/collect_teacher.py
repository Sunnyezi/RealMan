"""Collect demonstrations from the imitation-friendly teacher (pi05/teacher.py); simulation only.

Labels are the clean absolute targets of the teacher.  With --noise, smooth joint-space noise is added
to the executed arm targets of a deterministic subset of episodes (DART-style), so the recorded images
and joint states include small deviations while the labels steer back to the clean teacher path.
Only successful episodes are saved; every attempt is logged.
"""
import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path

import numpy as np

from pipeline import BASES, PROMPT, RGBEnv, fingerprint, write_json
from teacher import OUNoise, Teacher, advance

TEACHER = dict(index_extra=.12, thumb_extra=.12, settle_steps=4, dwell_steps=4,
               slow_lift=.2, slow_speed=.0015, grasp_dz=.003)
NOISE_LEVELS = (0., .002, .004)   # arm joint noise sigma in rad, chosen per episode by seed


def run_episode(env: RGBEnv, seed: int, sigma: float, record: bool):
    teacher = Teacher(**TEACHER)
    env.reset(seed)
    e = env.base
    teacher.reset()
    noise = OUNoise(np.random.default_rng(seed + 7_000_000), np.r_[np.full(6, sigma), np.zeros(6)])
    frames, phases, info = [], [], {'is_success': False}
    for _ in range(e.cfg['horizon']):
        obs = env.observation() if record else None
        state = e.d.qpos[env.qids].astype(np.float32).copy()
        context, label = advance(e, teacher(e), noise() if sigma else None, teacher)
        phases.append(teacher.phase)
        if record:
            frames.append({'image': obs['observation/image'], 'wrist_image': obs['observation/wrist_image'],
                           'state': state, 'actions': label, 'timestamp': (e.steps) * .05})
        _, _, term, trunc, info = e._finish(context)
        env.publish(f'pi05 teacher v2 / data collection / noise {sigma:.3f} rad')
        if term or trunc:
            break
    return frames, phases, info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['gate', 'collect'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--max-attempts', type=int)
    p.add_argument('--start-index', type=int, required=True)
    p.add_argument('--noise', action='store_true')
    p.add_argument('--gate', type=Path)
    p.add_argument('--stream', type=Path)
    a = p.parse_args()
    attempts = a.max_attempts or (a.count if a.mode == 'gate' else 2 * a.count)
    if a.count <= 0 or a.start_index < 0 or a.start_index + attempts > 90000:
        p.error('Invalid train seed range')
    if a.mode == 'collect':
        gate = json.loads(a.gate.read_text()) if a.gate else {}
        if not (gate.get('fingerprint') == fingerprint() and gate.get('teacher') == TEACHER
                and gate.get('episodes', 0) >= 100 and gate.get('success_rate', 0) >= .95):
            p.error('Need a matching >=95% clean teacher gate over >=100 train episodes')
    a.output.mkdir(parents=True, exist_ok=False)
    env = RGBEnv(a.stream)
    rows, accepted = [], 0
    try:
        with (a.output / 'attempts.jsonl').open('w') as log:
            for i in range(attempts):
                seed = BASES['train'] + a.start_index + i
                sigma = NOISE_LEVELS[seed % len(NOISE_LEVELS)] if a.noise else 0.
                started = time.monotonic()
                try:
                    frames, phases, info = run_episode(env, seed, sigma, a.mode == 'collect')
                except (ValueError, RuntimeError) as error:
                    frames, phases, info = [], [], {'is_success': False, 'error': str(error)}
                row = dict(seed=seed, split='train', prompt=PROMPT, fingerprint=fingerprint(),
                           teacher='pi05/teacher.py', teacher_params=TEACHER, noise_sigma_rad=sigma,
                           phase_frames={k: phases.count(k) for k in sorted(set(phases))},
                           wall_seconds=time.monotonic() - started, **info)
                rows.append(row)
                log.write(json.dumps(row) + '\n')
                log.flush()
                if a.mode == 'collect' and row['is_success']:
                    dest = a.output / f'episode_{i:06d}'
                    dest.mkdir()
                    np.savez_compressed(dest / 'trajectory.npz', **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
                    write_json(dest / 'metadata.json', row)
                    accepted += 1
                print(f"seed={seed} sigma={sigma} success={row['is_success']} accepted={accepted}", flush=True)
                if a.mode == 'collect' and accepted >= a.count:
                    break
    finally:
        env.close()
    report = dict(episodes=len(rows), successes=sum(r['is_success'] for r in rows), accepted=accepted,
                  requested=a.count, split='train', fingerprint=fingerprint(), teacher=TEACHER,
                  noise_levels=NOISE_LEVELS if a.noise else [0.], mode=a.mode)
    report['success_rate'] = report['successes'] / max(1, len(rows))
    write_json(a.output / 'summary.json', report)
    if a.mode == 'collect' and accepted < a.count:
        raise RuntimeError(f'Only collected {accepted}/{a.count} successful episodes')


if __name__ == '__main__':
    main()
