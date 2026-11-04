"""Compare scripted, imitation and PPO policies on the same floor scenarios."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from protocol import TEST_SEEDS, TRAINING_SEEDS, VALIDATION_SEEDS, episode, fingerprint, stress_yaw


def evaluate_chunk(task):
    candidate, checkpoint, seeds, stress, profile = task
    from domain import make_env
    from scripted import action as scripted_action

    env = make_env(profile, randomize=True)
    vector = None
    try:
        if checkpoint is None:
            def action(obs):
                return scripted_action(env)
        else:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            torch.set_num_threads(1)
            checkpoint = Path(checkpoint)
            manifest = json.loads((checkpoint.parent / 'run.json').read_text())
            if manifest['fingerprint'] != fingerprint():
                raise ValueError(f'Checkpoint scene differs: {checkpoint}')
            vector = DummyVecEnv([lambda: env])
            vector = VecNormalize.load(str(checkpoint.with_name(checkpoint.stem + '_vecnormalize.pkl')), vector)
            vector.training = False
            model = PPO.load(str(checkpoint), device='cpu')

            def action(obs):
                return model.predict(vector.normalize_obs(obs[None, :]), deterministic=True)[0][0]

        rows = []
        for seed in seeds:
            options = None
            if stress:
                options = {'object_yaw_rad': stress_yaw(seed)}
            row = episode(env, action, seed, options=options)
            if options:
                row.update(options)
            rows.append(row)
        return candidate, rows
    finally:
        if vector is None:
            env.close()
        else:
            vector.close()


def wilson(successes, total):
    z = 1.95996398454
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denominator
    return [float(max(0., center - half_width)), float(min(1., center + half_width))]


def summarize(rows):
    successes = sum(row['is_success'] for row in rows)
    failures = {}
    for row in rows:
        if row['failure']:
            failures[row['failure']] = failures.get(row['failure'], 0) + 1
    return dict(episodes=len(rows), successes=successes, success_rate=successes / len(rows),
                ci95=wilson(successes, len(rows)), failures=failures,
                episodes_with_physics_warnings=sum(row['warnings'] > 0 for row in rows),
                mean_sim_cycle_s=.05 * sum(row['steps'] for row in rows) / len(rows),
                results=rows)


def ppo_expansion(root):
    """Use paired validation outcomes only, resampling scenes shared across training seeds."""
    differences = []
    for seed in TRAINING_SEEDS:
        path = root / str(seed)
        rows = [json.loads((path / name).read_text())['results']
                for name in ('validation_imitation.json', 'best_policy_metrics.json')]
        if any(tuple(row['seed'] for row in candidate) != VALIDATION_SEEDS for candidate in rows):
            raise ValueError(f'Wrong validation split for run {seed}')
        imitation, ppo = ([int(row['is_success']) for row in candidate] for candidate in rows)
        differences.append(np.asarray(ppo) - imitation)
    differences = np.stack(differences)
    rng = np.random.default_rng(5103)
    samples = rng.integers(len(VALIDATION_SEEDS), size=(10000, len(VALIDATION_SEEDS)))
    lower, upper = np.quantile(differences[:, samples].mean(axis=(0, 2)), [.025, .975])
    mean_by_seed = differences.mean(axis=1)
    return dict(validation_gain_by_training_seed=dict(zip(TRAINING_SEEDS, mean_by_seed.tolist())),
                paired_ci95=[float(lower), float(upper)],
                expand_ppo=bool(np.all(mean_by_seed > 0) and lower > 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['script', 'compare', 'gate', 'pose'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, help='Calibrated domain ranges shared across candidates')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', choices=['validation', 'test', 'yaw'], default='test')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.mode in ('compare', 'gate') and args.root is None:
        parser.error('--root is required for compare and gate')
    if args.mode == 'pose':
        from yaw_probe import probe
        report = probe()
    elif args.mode == 'gate':
        report = ppo_expansion(args.root)
    else:
        from domain import load_profile
        profile = load_profile(args.profile) if args.profile else None
        seeds = TEST_SEEDS if args.split == 'test' else VALIDATION_SEEDS
        candidates = [('script', None)]
        if args.mode == 'compare':
            candidates += [(f'{kind}_{seed}', args.root / str(seed) / file)
                           for seed in TRAINING_SEEDS
                           for kind, file in [('imitation', 'warmstart.zip'), ('ppo', 'best_policy.zip')]]
        size = max(1, len(seeds) // max(4, args.workers // len(candidates)))
        tasks = [(name, checkpoint, seeds[start:start + size], args.split == 'yaw', profile)
                 for name, checkpoint in candidates for start in range(0, len(seeds), size)]
        rows = {name: [] for name, _ in candidates}
        started = time.monotonic()
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for name, batch in pool.map(evaluate_chunk, tasks):
                rows[name].extend(batch)
                print(name, len(rows[name]), '/', len(seeds), flush=True)
        report = dict(split=args.split, domain_profile=profile, wall_seconds=time.monotonic() - started,
                      candidates={name: summarize(results) for name, results in rows.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
