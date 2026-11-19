"""Checks for the pick-and-place task, teacher and the unchanged pick-and-hold contract."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / 'grasp'), str(ROOT)]

from pipeline import RGBEnv, fingerprint  # noqa: E402
from place_task import (HORIZON, SCENE, TOLERANCE, PlaceEnv, build_scene_xml, place_ok,  # noqa: E402
                        place_fingerprint)
from place_teacher import PHASES, PlaceTeacher  # noqa: E402


def test_hold_contract_unchanged() -> None:
    interface = json.loads((ROOT / 'interface.json').read_text())
    assert fingerprint() == interface['simulation_fingerprint']
    manifest = json.loads((ROOT / 'runs/baseline_20261117/training_manifest.json').read_text())
    assert hashlib.sha256((ROOT / 'adapter.py').read_bytes()).hexdigest() == manifest['adapter_sha256']
    assert place_fingerprint() != fingerprint()


def test_scene_regenerates_with_same_state_layout() -> None:
    import mujoco
    assert SCENE.read_text() == build_scene_xml()
    hold = mujoco.MjModel.from_xml_path(str(ROOT.parent / 'grasp/scene.xml'))
    place = mujoco.MjModel.from_xml_path(str(SCENE))
    assert (place.nq, place.nv, place.nu, place.nsensor) == (hold.nq, hold.nv, hold.nu, hold.nsensor)
    assert place.nmocap == 1 and place.nbody == hold.nbody + 1
    g = place.geom('place_target_geom').id
    assert place.geom_contype[g] == 0 and place.geom_conaffinity[g] == 0


def test_same_physics_as_hold_env(seed: int = 120) -> None:
    place, hold = PlaceEnv(), RGBEnv()
    try:
        place.reset(seed)
        hold.reset(seed)
        assert np.array_equal(place.base.d.qpos, hold.base.d.qpos)
        teacher = PlaceTeacher()
        teacher.reset()
        for _ in range(60):
            command = teacher(place).astype(np.float32)
            place.step(command)
            hold.step(command)
        assert np.allclose(place.base.d.qpos, hold.base.d.qpos, atol=1e-9, rtol=0)
        assert 0 < place.base.d.mocap_pos[0, 2] < .001
    finally:
        place.close()
        hold.close()


def test_teacher_places_cube(seed: int = 10) -> None:
    env = PlaceEnv()
    try:
        env.reset(10000 + seed)
        teacher = PlaceTeacher()
        teacher.reset()
        phases = []
        for _ in range(HORIZON):
            done, info = env.step(teacher(env).astype(np.float32))
            phases.append(teacher.phase)
            if done:
                break
        order = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
        assert order == list(PHASES), order
        assert info['is_success'] and info['target_error_xy'] <= TOLERANCE, info
        assert all(info['place_milestones'].values()) and PlaceEnv.failure(info) is None
    finally:
        env.close()


def test_place_conditions() -> None:
    good = dict(lifted=True, error=.01, resting=True, contacts=0, clearance=.06, floor=False, release_height=.004)
    assert place_ok(**good)
    for key, bad in (('lifted', False), ('error', TOLERANCE + .001), ('resting', False), ('contacts', 1),
                     ('clearance', .01), ('floor', True), ('release_height', .08), ('release_height', None)):
        assert not place_ok(**(good | {key: bad})), key


def test_unlifted_or_outside_cube_fails(seed: int = 11) -> None:
    env = PlaceEnv()
    try:
        for lifted, inside in ((False, True), (True, False)):
            env.reset(10000 + seed)
            e = env.base
            xy = env.target[:2] if inside else env.target[:2] + [.06, 0.]
            e.d.qpos[e.oa:e.oa + 2] = xy
            e.d.qvel[:] = 0
            env.lifted, env.release_height, env.prev_contacts = lifted, .004 if lifted else None, 0
            hold = e.d.ctrl[env.ids].copy()
            for _ in range(40):
                done, info = env.step(hold)
                if done:
                    break
            assert not info['is_success'], (lifted, inside)
            assert info['dropped'] == (lifted and not inside), (lifted, inside, info['dropped'])
    finally:
        env.close()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('passed', name, flush=True)
