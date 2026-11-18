"""Recovery demonstrations from policy-visited states; RGB-D/touch expert labels only."""
from pathlib import Path

import numpy as np

from evaluate_stabilized import robot_state
from pipeline import GRASP, RGBEnv
from stabilizer import Stabilizer


def save_initial(env: RGBEnv, path: Path) -> None:
    e = env.base
    spec = int(env.mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(env.mujoco.mj_stateSize(e.m, spec))
    env.mujoco.mj_getState(e.m, e.d, state, spec)
    np.savez_compressed(path, integration=state, spec=spec, touch=e.touch_force.copy(),
                        z0=e.z0, ever_lifted=e.ever_lifted, vision=e.vision.last.copy(),
                        last_seen=e.vision.last_seen, visible=e.vision.visible)


def restore_initial(env: RGBEnv, path: Path) -> None:
    e = env.base
    with np.load(path, allow_pickle=False) as data:
        env.mujoco.mj_setState(e.m, e.d, data['integration'], int(data['spec']))
        e.touch_force = data['touch'].copy()
        e.touch_flags = (e.touch_force > e.cfg['contact_threshold_N']).astype(float)
        e.z0 = float(data['z0'])
        e.ever_lifted = bool(data['ever_lifted'])
        e.vision.last = data['vision'].copy()
        e.vision.last_seen = float(data['last_seen'])
        e.vision.visible = bool(data['visible'])
    env.mujoco.mj_forward(e.m, e.d)
    e.steps = 0
    e.hold = 0.
    e.max_height = 0.


class RecoveryExpert:
    """Reopen/retract/re-estimate after misses; stabilize an already bilateral grip."""
    def __init__(self, env: RGBEnv, thumb: np.ndarray, index: np.ndarray, *, direct_align: bool = False):
        self.env, self.thumb, self.index = env, thumb, index
        e = env.base
        # Explicitly use the initial RGB-D estimate for retreat height, then refresh after opening.
        import sys
        sys.path.insert(0, str(GRASP))
        from build_scene import solve
        fk_data = env.mujoco.MjData(e.m)
        ik_data = env.mujoco.MjData(e.m)
        def forward(q: np.ndarray) -> np.ndarray:
            fk_data.qpos[env.qids[:6]] = q
            env.mujoco.mj_forward(e.m, fk_data)
            return fk_data.xpos[e.bid].copy()
        def inverse(point: np.ndarray, rotation: np.ndarray, seed: np.ndarray) -> np.ndarray:
            return solve(e.m, ik_data, point, rotation, seed)
        closed = np.asarray(e.cfg['hand_target'], float).copy()
        closed[1:3] += .12
        visual = e.vision.last + np.asarray(e.cfg['grasp_xyz'])-np.asarray(e.cfg['object_xyz'])
        self.control = Stabilizer(e.d.ctrl[env.ids], e.m.actuator_ctrlrange[env.ids], closed,
                                  np.asarray(e.cfg['rotation']), forward, inverse, visual)
        self.control.open_hand = np.asarray(e.cfg['hand_target'], float).copy()
        self.control.open_hand[1:3] = [e.cfg['opened_thumb2'], 0.]
        self.initial_grasp = self.control.visual_grasp.copy()
        state = robot_state(env, thumb, index)
        grip = (state.thumb_force >= .5 and state.index_force >= .5
                and state.joints[7] >= .2 and state.joints[8] >= .25)
        point = self.control.forward(self.control.command[:6])
        if grip:
            self.kind = 'grip_stabilization'
            self.control.anchor = point.copy()
            self.control.path = point.copy()
            self.control.goal = point.copy()
            self.control.goal[2] = max(point[2], self.initial_grasp[2]+.13)
            self.control.phase = 'secure'
            self.phase = 'control'
        elif direct_align:
            self.kind = 'direct_align_close'
            self.control.phase = 'align'
            self.phase = 'control'
        else:
            self.kind = 'reopen_realign'
            self.phase = 'release'
            self.retract = point.copy()
            self.retract[2] = max(point[2]+.06, self.initial_grasp[2]+.08)
        self.steps = 0

    def update(self) -> np.ndarray:
        c, e = self.control, self.env.base
        self.steps += 1
        if self.phase == 'release':
            point = c.forward(c.command[:6])
            delta = np.clip(self.retract-point, -.0025, .0025)
            arm, _ = c._project(point+delta)
            desired = np.r_[arm, c.open_hand]
            rate = np.r_[np.full(6, .06), np.full(6, .02)]
            c.command += np.clip(desired-c.command, -rate, rate)
            if np.linalg.norm(point-self.retract) < .001 and np.max(np.abs(c.command[6:]-c.open_hand)) < .001:
                estimate, visible, _ = e.vision.update(e.d)
                if not visible:
                    raise ValueError('Cube not visible after opening and retracting')
                c.visual_grasp = estimate + np.asarray(e.cfg['grasp_xyz'])-np.asarray(e.cfg['object_xyz'])
                c.phase = 'align'
                self.phase = 'control'
            return c.command.copy()
        if c.needs_policy:
            raise ValueError('Recovery expert failed to confirm grip')
        return c.update(robot_state(self.env, self.thumb, self.index))
