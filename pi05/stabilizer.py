"""RGB-D grasp alignment and robot/tactile stabilization, without object ground truth."""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Settings:
    dt: float = 0.05
    arm_alpha: float = 0.7
    arm_speed: float = 1.2
    hand_speed: float = 0.4
    translation_speed: float = 0.06
    lift_speed: float = 0.05
    lift_distance: float = 0.13
    contact_force: float = 0.1
    contact_steps: int = 3
    secure_seconds: float = 0.5
    thumb_closed_min: float = 0.20
    index_closed_min: float = 0.25
    close_request: float = 0.04
    align_tolerance: float = 0.001
    secure_timeout: float = 2.0
    policy_seconds: float = 2.0


@dataclass(frozen=True)
class RobotState:
    joints: np.ndarray
    palm_position: np.ndarray
    thumb_force: float
    index_force: float


class Stabilizer:
    def __init__(self, initial_command: np.ndarray, limits: np.ndarray, closed_hand: np.ndarray,
                 rotation: np.ndarray, forward: Callable[[np.ndarray], np.ndarray],
                 inverse: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                 visual_grasp: np.ndarray,
                 settings: Settings | None = None):
        self.cfg = settings or Settings()
        self.command = np.asarray(initial_command, dtype=float).copy()
        self.limits = np.asarray(limits, dtype=float)
        self.closed_hand = np.asarray(closed_hand, dtype=float).copy()
        self.rotation = np.asarray(rotation, dtype=float).copy()
        self.visual_grasp = np.asarray(visual_grasp, dtype=float).copy()
        if (self.command.shape != (12,) or self.limits.shape != (12, 2)
                or self.closed_hand.shape != (6,) or self.rotation.shape != (3, 3)
                or self.visual_grasp.shape != (3,)):
            raise ValueError("Invalid actuator or pose configuration")
        if not all(np.isfinite(x).all() for x in (self.command, self.limits, self.closed_hand, self.rotation, self.visual_grasp)):
            raise ValueError("Nonfinite controller configuration")
        self.command = np.clip(self.command, *self.limits.T)
        self.filtered = self.command.copy()
        self.open_hand = self.command[6:].copy()
        self.forward = forward
        self.inverse = inverse
        self.phase = "policy"
        self.contact_count = 0
        self.phase_time = 0.
        self.anchor: np.ndarray | None = None
        self.path: np.ndarray | None = None
        self.goal: np.ndarray | None = None
        self.hold_arm: np.ndarray | None = None
        self.ik_rejections = 0
        self.transitions: list[dict] = []
        self.steps = 0

    @property
    def needs_policy(self) -> bool:
        return self.phase == "policy"

    def describe(self) -> dict:
        return {"name": "pi05_rgbd_align_tactile_hold_v4", "settings": asdict(self.cfg),
                "control_inputs": ["active_joint_positions", "palm_forward_kinematics", "thumb_and_index_touch", "initial_rgbd_grasp_estimate"],
                "uses_object_ground_truth": False, "closed_hand_rad": self.closed_hand.tolist(),
                "visual_grasp_m": self.visual_grasp.tolist(),
                "visual_source": "calibrated_rgbd_blue_segmentation_known_4cm_cube"}

    def _transition(self, phase: str) -> None:
        self.transitions.append({"step": self.steps, "from": self.phase, "to": phase})
        self.phase = phase
        self.phase_time = 0.

    def _project(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_position = np.asarray(self.forward(self.command[:6]), dtype=float)
        for scale in (1., .5, .25, .125, 0.):
            point = base_position + scale * (target - base_position)
            try:
                q = np.asarray(self.inverse(point, self.rotation, self.command[:6]), dtype=float)
                if q.shape != (6,) or not np.isfinite(q).all():
                    raise ValueError("Invalid inverse kinematics result")
                return q, point
            except ValueError:
                self.ik_rejections += 1
        return self.command[:6].copy(), base_position.copy()

    def update(self, state: RobotState, raw_action: np.ndarray | None = None) -> np.ndarray:
        if (state.joints.shape != (12,) or state.palm_position.shape != (3,)
                or not np.isfinite(state.joints).all() or not np.isfinite(state.palm_position).all()
                or not np.isfinite([state.thumb_force, state.index_force]).all()):
            raise ValueError("Invalid robot feedback")
        touch = (state.thumb_force >= self.cfg.contact_force and state.index_force >= self.cfg.contact_force)
        closed = state.joints[7] >= self.cfg.thumb_closed_min and state.joints[8] >= self.cfg.index_closed_min
        if self.phase == "policy":
            if raw_action is None:
                raise ValueError("Policy action required before contact confirmation")
            raw = np.asarray(raw_action, dtype=float)
            if raw.shape != (12,) or not np.isfinite(raw).all():
                raise ValueError("Expected twelve finite policy targets")
            if raw[8] > self.open_hand[2] + self.cfg.close_request or self.phase_time >= self.cfg.policy_seconds:
                self._transition("align")
            else:
                self.filtered[:6] += self.cfg.arm_alpha * (np.clip(raw[:6], *self.limits[:6].T) - self.filtered[:6])
                target = np.asarray(self.forward(self.filtered[:6]), dtype=float)
                # RGB-D workspace guard prevents premature descent or lifting before alignment.
                target[:2] = np.clip(target[:2], self.visual_grasp[:2]-.025, self.visual_grasp[:2]+.025)
                target[2] = max(target[2], self.visual_grasp[2])
                base_position = np.asarray(self.forward(self.command[:6]), dtype=float)
                delta = target - base_position
                delta *= min(1., self.cfg.translation_speed * self.cfg.dt / max(np.linalg.norm(delta), 1e-12))
                arm, _ = self._project(base_position + delta)
                desired = np.r_[arm, self.open_hand]
        if self.phase != "policy":
            if self.phase == "align":
                base_position = np.asarray(self.forward(self.command[:6]), dtype=float)
                delta = np.clip(self.visual_grasp-base_position, -self.cfg.lift_speed*self.cfg.dt, self.cfg.lift_speed*self.cfg.dt)
                arm, _ = self._project(base_position+delta)
                desired = np.r_[arm, self.open_hand]
                if np.linalg.norm(base_position-self.visual_grasp) <= self.cfg.align_tolerance:
                    self.anchor = self.visual_grasp.copy()
                    self.path = self.anchor.copy()
                    self.goal = self.anchor + [0., 0., self.cfg.lift_distance]
                    self.contact_count = 0
                    self._transition("secure")
            elif self.phase == "secure":
                self.contact_count = self.contact_count + 1 if touch and closed else 0
                if self.phase_time >= self.cfg.secure_timeout:
                    self.contact_count = 0
                    self.filtered = self.command.copy()
                    self._transition("policy")
                    desired = self.command.copy()
                else:
                    arm, _ = self._project(self.anchor)
                    desired = np.r_[arm, self.closed_hand]
                    hand_ready = np.max(np.abs(self.command[6:] - self.closed_hand)) <= self.cfg.hand_speed*self.cfg.dt + 1e-8
                    if (self.phase_time >= self.cfg.secure_seconds and self.contact_count >= self.cfg.contact_steps and hand_ready):
                        self._transition("lift")
            elif self.phase == "lift":
                target = self.path.copy()
                target[2] = min(self.goal[2], target[2] + self.cfg.lift_speed * self.cfg.dt)
                arm, accepted = self._project(target)
                self.path = accepted
                desired = np.r_[arm, self.closed_hand]
                if np.linalg.norm(self.path - self.goal) <= 1e-6:
                    self.hold_arm = arm.copy()
                    self._transition("hold")
            elif self.phase == "hold":
                desired = np.r_[self.hold_arm, self.closed_hand]
        rate = np.r_[np.full(6, self.cfg.arm_speed), np.full(6, self.cfg.hand_speed)]
        desired = np.clip(desired, *self.limits.T)
        self.command += np.clip(desired - self.command, -rate*self.cfg.dt, rate*self.cfg.dt)
        self.command = np.clip(self.command, *self.limits.T)
        self.phase_time += self.cfg.dt
        self.steps += 1
        return self.command.copy()
