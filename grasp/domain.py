"""Optional domain ranges for calibrated training and shared stress evaluation."""
import json
from pathlib import Path

import mujoco
import numpy as np

from env import GraspEnv
from scipy.spatial.transform import Rotation

DEFAULTS = {
    'mass_scale': (1., 1.), 'friction_scale': (1., 1.),
    'actuator_scale': (1., 1.), 'light_scale': (1., 1.),
    'yaw_deg': (0., 0.), 'camera_offset_m': (0., 0.),
    'vision_error_m': (0., 0.), 'vision_drop_probability': (0., 0.),
    'touch_noise_N': (0., 0.), 'action_delay_steps': (0., 0.),
    'joint_position_noise_rad': (0., 0.), 'joint_velocity_noise_rad_s': (0., 0.),
    'camera_rotation_deg': (0., 0.),
}


def load_profile(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) - DEFAULTS.keys():
        raise ValueError('Unknown domain profile fields')
    ranges = {}
    for key, default in DEFAULTS.items():
        bounds = np.asarray(data.get(key, default), dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] < bounds[0]:
            raise ValueError(f'{key}: expected ordered finite [min, max]')
        if key.endswith('_scale') and bounds[0] <= 0:
            raise ValueError(f'{key}: scales must be positive')
        if key != 'yaw_deg' and bounds[0] < 0:
            raise ValueError(f'{key}: expected nonnegative magnitude')
        if key == 'vision_drop_probability' and bounds[1] > 1:
            raise ValueError(f'{key}: expected probability in [0, 1]')
        if key == 'action_delay_steps' and (any(value != int(value) for value in bounds) or bounds[1] > 10):
            raise ValueError(f'{key}: expected integers in [0, 10]')
        ranges[key] = tuple(bounds.tolist())
    return ranges


class DomainEnv(GraspEnv):
    def __init__(self, ranges, **kwargs):
        super().__init__(**kwargs)
        self.ranges = ranges
        self.rng = np.random.default_rng()
        floor = self.m.geom('floor').id
        self.friction_ids = [floor, self.og]
        self.camera_id = self.m.camera('rgbd').id
        self.nominal = dict(mass=float(self.m.body_mass[self.ob]), inertia=self.m.body_inertia[self.ob].copy(),
                            friction=self.m.geom_friction[self.friction_ids].copy(),
                            gain=self.m.actuator_gainprm[:, 0].copy(), bias=self.m.actuator_biasprm[:, 1:3].copy(),
                            diffuse=self.m.light_diffuse.copy(), camera=self.m.cam_pos[self.camera_id].copy(),
                            camera_quat=self.m.cam_quat[self.camera_id].copy())

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([seed, 11025]))
        self.draw = {name: self.rng.uniform(*bounds) for name, bounds in self.ranges.items()}
        delay_min, delay_max = self.ranges['action_delay_steps']
        self.delay = int(self.rng.integers(int(delay_min), int(delay_max)+1))
        self.commands = [np.zeros(self.action_space.shape, dtype=np.float32) for _ in range(self.delay)]
        nominal, draw = self.nominal, self.draw
        self.m.body_mass[self.ob] = nominal['mass'] * draw['mass_scale']
        self.m.body_inertia[self.ob] = nominal['inertia'] * draw['mass_scale']
        self.m.geom_friction[self.friction_ids] = nominal['friction'] * draw['friction_scale']
        self.m.actuator_gainprm[:, 0] = nominal['gain'] * draw['actuator_scale']
        self.m.actuator_biasprm[:, 1:3] = nominal['bias'] * draw['actuator_scale']
        self.m.light_diffuse[:] = nominal['diffuse'] * draw['light_scale']
        offset = self.rng.uniform(-draw['camera_offset_m'], draw['camera_offset_m'], 3)
        self.m.cam_pos[self.camera_id] = nominal['camera'] + offset
        angle_deg = draw['camera_rotation_deg']
        rotation = Rotation.from_rotvec(np.deg2rad(self.rng.uniform(-angle_deg, angle_deg, 3)))
        delta = rotation.as_quat()[[3, 0, 1, 2]]
        mujoco.mju_mulQuat(self.m.cam_quat[self.camera_id], nominal['camera_quat'], delta)
        mujoco.mj_setConst(self.m, self.d)
        options = {'object_yaw_rad': np.deg2rad(draw['yaw_deg']), **(options or {})}
        self.initializing = True
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self.initializing = False

    def _vision(self, force=False):
        if not force and self.d.time+1e-8 < self.next_vision:
            return
        previous = self.vision.last.copy(), self.vision.last_seen
        super()._vision(force)
        if self.initializing or not self.vision.visible:
            return
        if self.rng.random() < self.draw['vision_drop_probability']:
            self.vision.last[:], self.vision.last_seen = previous
            self.vision.visible = False
        else:
            self.vision.last += self.rng.normal(0., self.draw['vision_error_m'], 3)

    def _obs(self):
        observation = super()._obs()
        observation[:self.robot_nq] += self.rng.normal(
            0., self.draw['joint_position_noise_rad'], self.robot_nq)
        velocity = slice(self.robot_nq, self.robot_nq+self.robot_nv)
        observation[velocity] += self.rng.normal(
            0., self.draw['joint_velocity_noise_rad_s'], self.robot_nv)
        return observation

    def _update_touch(self, mean_force):
        noise = self.rng.normal(0., self.draw['touch_noise_N'], len(mean_force))
        super()._update_touch(np.maximum(0., mean_force + noise))

    def step(self, action):
        if self.delay:
            self.commands.append(np.asarray(action).copy())
            action = self.commands.pop(0)
        return super().step(action)


def make_env(profile=None, **kwargs):
    return GraspEnv(**kwargs) if profile is None else DomainEnv(profile, **kwargs)
