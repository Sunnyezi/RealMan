"""Fixed disjoint splits and full-episode evaluation for the floor task."""
import hashlib
from pathlib import Path
import time

import numpy as np

VALIDATION_SEEDS = tuple(range(71001, 71031))
TEST_SEEDS = tuple(range(81001, 81201))
TRAINING_SEEDS = (42, 43, 44)




def fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ('scene.xml', 'config.json', 'env.py', 'vision.py', 'planner.py',
                 'sensor_map.json', 'check.py', 'scripted.py'):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def stress_yaw(seed):
    return float(np.random.default_rng(seed + 1000000).uniform(-np.pi / 4, np.pi / 4))


def episode(env, action, seed, options=None):
    started = time.monotonic()
    try:
        obs, _ = env.reset(seed=seed, options=options)
    except (ValueError, RuntimeError) as error:
        return dict(seed=seed, is_success=False, reward=0., warnings=0,
                    steps=0, failure='reset', detail=str(error),
                    wall_seconds=time.monotonic() - started)
    reward = 0.
    for step in range(1, env.cfg['horizon'] + 1):
        obs, value, terminated, truncated, info = env.step(action(obs))
        reward += value
        if terminated or truncated:
            break
    if info['is_success']:
        failure = None
    elif info['warnings'] or info.get('physics_failure'):
        failure = 'physics'
    elif info['table_collision']:
        failure = 'ground_contact'
    elif info['dropped']:
        failure = 'dropped'
    elif info['ik_failures'] or info['ik_rejections']:
        failure = 'ik'
    else:
        failure = 'incomplete'
    return dict(seed=seed, reward=float(reward), steps=step, failure=failure,
                wall_seconds=time.monotonic() - started, **info)
