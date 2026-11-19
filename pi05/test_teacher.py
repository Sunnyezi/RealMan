"""Regression checks for pi05/teacher.py against the original environment step and scripted teacher."""
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / 'grasp'), str(ROOT)]

from env import GraspEnv  # noqa: E402
import scripted  # noqa: E402
from teacher import Teacher, advance, OUNoise  # noqa: E402

ORIGINAL = dict(index_extra=.12, thumb_extra=.12, settle_steps=0, dwell_steps=0, slow_lift=0.)


def test_advance_matches_env_and_original_teacher(seed: int = 81000) -> None:
    a, b = GraspEnv(randomize=True), GraspEnv(randomize=True)
    try:
        a.reset(seed=seed)
        b.reset(seed=seed)
        teacher = Teacher(**ORIGINAL)
        teacher.reset()
        for _ in range(a.cfg['horizon']):
            act_a, act_b = scripted.action(a), teacher(b)
            assert np.array_equal(act_a, act_b), (a.steps, act_a, act_b)
            ctx_a = a._advance(act_a)
            ctx_b, label = advance(b, act_b)
            assert np.array_equal(a.d.qpos, b.d.qpos) and np.array_equal(a.d.ctrl, b.d.ctrl)
            assert np.array_equal(label, b.d.ctrl[np.r_[b.arm, b.hand]].astype(np.float32))
            _, _, term, trunc, info_a = a._finish(ctx_a)
            _, _, _, _, info_b = b._finish(ctx_b)
            assert info_a == info_b
            if term or trunc:
                break
        assert info_a['is_success']
    finally:
        a.close()
        b.close()


def test_noise_changes_execution_not_labels(seed: int = 81001) -> None:
    e = GraspEnv(randomize=True)
    try:
        e.reset(seed=seed)
        teacher = Teacher()
        teacher.reset()
        noise = OUNoise(np.random.default_rng(0), np.r_[np.full(6, .004), np.zeros(6)])
        for _ in range(5):
            n = noise()
            _, label = advance(e, teacher(e), n, teacher)
            ids = np.r_[e.arm, e.hand]
            assert np.allclose(e.d.ctrl[ids], np.clip(label + n, *e.m.actuator_ctrlrange[ids].T), atol=1e-6)
            assert np.array_equal(label[:6], e.q_target.astype(np.float32))
            e._finish((np.zeros(9), np.zeros(9), False, False))
    finally:
        e.close()


def test_teacher_phases_in_order(seed: int = 81002) -> None:
    e = GraspEnv(randomize=True)
    try:
        e.reset(seed=seed)
        teacher = Teacher(settle_steps=4, dwell_steps=4, slow_lift=.2, slow_speed=.0015, grasp_dz=.003)
        teacher.reset()
        phases = []
        for _ in range(e.cfg['horizon']):
            ctx, _ = advance(e, teacher(e), None, teacher)
            phases.append(teacher.phase)
            _, _, term, trunc, info = e._finish(ctx)
            if term or trunc:
                break
        order = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
        assert order == ['approach', 'close', 'dwell', 'lift'], order
        assert phases.count('dwell') == 4 and info['is_success']
    finally:
        e.close()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('passed', name, flush=True)
