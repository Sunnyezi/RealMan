"""Verify recorded correction suffixes against their saved simulator states."""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline import RGBEnv, validate_episode, write_json
from recovery import restore_initial


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = sorted(a.raw.glob('episode_*'))
    if not paths or a.samples<=0:
        p.error('Need episodes and positive sample count')
    metas = [validate_episode(path) for path in paths]
    if len({m['seed'] for m in metas}) != len(metas) or any(m['split']!='train' or not m['is_success'] for m in metas):
        raise ValueError('Corrections must use unique successful train episodes')
    samples = np.unique(np.linspace(0,len(paths)-1,min(a.samples,len(paths)),dtype=int))
    env = RGBEnv()
    rows = []
    try:
        for idx in samples:
            path, meta = paths[int(idx)], metas[int(idx)]
            env.reset(meta['seed'])
            restore_initial(env, path/'initial_state.npz')
            max_error = 0.
            with np.load(path/'trajectory.npz', allow_pickle=False) as data:
                for state, action in zip(data['state'],data['actions'],strict=True):
                    max_error = max(max_error,float(np.max(np.abs(env.base.d.qpos[env.qids]-state))))
                    _, info = env.step(action)
                frames = len(data['state'])
            rows.append(dict(seed=meta['seed'], frames=frames, max_state_error=max_error,
                             is_success=info['is_success'], physics_failure=info['physics_failure']))
            if max_error>1e-4 or not info['is_success'] or info['physics_failure']:
                raise RuntimeError(json.dumps(rows[-1]))
    finally:
        env.close()
    write_json(a.output, dict(episodes_checked=len(paths), replayed=len(rows), replays=rows,
                             max_state_error=max(r['max_state_error'] for r in rows), passed=True))
    print(json.dumps(dict(episodes=len(paths), replayed=len(rows), passed=True)), flush=True)


if __name__ == '__main__':
    main()
