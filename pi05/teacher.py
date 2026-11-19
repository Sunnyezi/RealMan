"""Imitation-friendly scripted teacher for the floor cube pickup task; simulation only.

The original `grasp/scripted.py` lifts on the same step the closing command reaches its target, so a
learned policy that is a few steps early lifts with half-closed fingers.  This teacher keeps the same
geometry, planner and IK path, but adds observable pauses (settle before closing, dwell after closing),
a slower first part of the lift, and a firmer closing target.  `advance()` mirrors
`GraspEnv._advance` and can add execution noise for DART-style demonstrations: noisy targets are executed
while the clean targets are recorded as labels.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Teacher:
    index_extra: float = .20     # added to config hand_target[2]; original teacher uses .12
    thumb_extra: float = .16     # added to config hand_target[1]; original teacher uses .12
    settle_steps: int = 4        # hold still at the grasp pose before closing
    dwell_steps: int = 10        # hold still after the closing command reaches its target
    slow_lift: float = .03       # first part of the lift at slow_speed
    slow_speed: float = .00125
    speed: float = .0025
    lift_distance: float = .12   # palm rise above the grasp pose; the env planner uses .12
    grasp_dz: float = 0.         # vertical offset of the grasp pose relative to the planner
    settle: int = field(default=0, init=False)
    dwell: int = field(default=0, init=False)
    phase: str = field(default='approach', init=False)

    def reset(self) -> None:
        self.settle = self.dwell = 0
        self.phase = 'approach'

    def hand_goal(self, env) -> np.ndarray:
        goal = np.array(env.cfg['hand_target'], dtype=float)
        goal[1] += self.thumb_extra
        goal[2] += self.index_extra
        return np.clip(goal, *env.m.actuator_ctrlrange[env.hand].T)

    def __call__(self, env) -> np.ndarray:
        grasp = env.grasp_goal + np.array([0., 0., self.grasp_dz])
        up, goal = grasp + np.array([0., 0., self.lift_distance]), self.hand_goal(env)
        speed = self.speed
        if env.hand_target[2] < .01 and self.phase == 'approach':
            if np.linalg.norm(env.target_xyz - grasp) > .003:
                xyz, hand = grasp, env.hand_target.copy()
            elif self.settle < self.settle_steps:
                self.settle += 1
                xyz, hand = grasp, env.hand_target.copy()
            else:
                self.phase = 'close'
                xyz, hand = grasp, goal
        elif np.any(env.hand_target[1:3] < goal[1:3] - .001):
            self.phase = 'close'
            xyz, hand = grasp, goal
        elif self.dwell < self.dwell_steps:
            self.phase = 'dwell'
            self.dwell += 1
            xyz, hand = grasp, goal
        else:
            self.phase = 'lift'
            xyz, hand = up, goal
            if env.target_xyz[2] - grasp[2] < self.slow_lift:
                speed = self.slow_speed
        delta = np.clip(xyz - env.target_xyz, -speed, speed)
        dh = np.clip(hand - env.hand_target, -.02, .02)
        return np.clip(np.r_[delta / .005, dh / .06], -1, 1).astype(np.float32)


def advance(env, action: np.ndarray, noise: np.ndarray | None = None, teacher: Teacher | None = None):
    """Same as GraspEnv._advance, except `noise` (12,) is added to the executed arm/hand targets only.

    With `teacher`, the Cartesian target box also admits that teacher's grasp offset and lift height.
    """
    import mujoco
    from build_scene import solve, R

    self = env
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (9,) or not np.isfinite(action).all():
        raise ValueError('Action must be finite and have shape (9,)')
    action = np.clip(action, -1, 1)
    if (not self.grasp_locked and self.vision.visible and self.d.time - self.vision.last_seen < .15
            and np.linalg.norm(self.vision.last - (self.grasp_goal - np.array(self.cfg['grasp_xyz'])
                                                   + np.array(self.cfg['object_xyz']))) > .002):
        try:
            self._set_plan(self.vision.last)
        except ValueError:
            self.plan_rejections += 1
    low = self.grasp_goal - np.array([.025, .025, .004])
    high = np.maximum(self.lift_goal, self.cfg['home_xyz']) + np.array([.025, .025, 0.])
    if teacher is not None:
        low[2] += min(0., teacher.grasp_dz)
        high[2] = max(high[2], self.grasp_goal[2] + teacher.grasp_dz + teacher.lift_distance)
    desired = self.target_xyz + .005 * action[:3]
    candidate = np.clip(desired, low, high)
    projected = not np.allclose(candidate, desired, atol=1e-9)
    accepted = rejected = False
    for scale in [1., .5, .25, .125]:
        point = self.target_xyz + scale * (candidate - self.target_xyz)
        try:
            if self.planner.margin(point) < .003:
                raise ValueError('Reach margin')
            q = solve(self.m, self.ikd, point, R, self.q_target)
            self.q_target = q
            self.target_xyz = point
            accepted = True
            projected |= scale < 1
            break
        except ValueError:
            rejected = True
    self.ik_rejections += int(rejected)
    self.action_projections += int(projected)
    if not accepted:
        self.ik_failures += 1
    self.hand_target = np.clip(self.hand_target + .06 * action[3:], self.m.actuator_ctrlrange[self.hand, 0],
                               self.m.actuator_ctrlrange[self.hand, 1])
    clean = np.r_[self.q_target, self.hand_target]
    executed = clean if noise is None else clean + noise
    ids = np.r_[self.arm, self.hand]
    self.d.ctrl[ids] = np.clip(executed, *self.m.actuator_ctrlrange[ids].T)
    self.grasp_locked |= self.hand_target[2] > .01
    oldaction = self.prev_action.copy()
    self.prev_action = action.copy()
    force_sum = np.zeros(17)
    for _ in range(25):
        mujoco.mj_step(self.m, self.d)
        force_sum += self.d.sensordata[self.sensor_adrs]
    self._update_touch(force_sum / 25)
    return (action, oldaction, rejected, projected), clean.astype(np.float32)


class OUNoise:
    """Smooth zero-mean execution noise in joint space; sigma in radians per active joint."""

    def __init__(self, rng: np.random.Generator, sigma: np.ndarray, theta: float = .15):
        self.rng, self.sigma, self.theta = rng, np.asarray(sigma, dtype=float), theta
        self.x = np.zeros_like(self.sigma)

    def __call__(self) -> np.ndarray:
        self.x += -self.theta * self.x + self.sigma * np.sqrt(2 * self.theta) * self.rng.standard_normal(self.x.shape)
        return self.x.copy()
