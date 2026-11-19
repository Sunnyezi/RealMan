"""Scripted pick-and-place teacher for pi05/place_task.py; simulation only.

Phase changes depend only on observable progress (arm and hand targets, cube height) and never on hidden
timers or pauses, so the same observation does not map to different actions.  The grasp reuses the
geometry of the original teacher (`grasp/scripted.py`: planner grasp pose with on-line RGB-D replanning,
closing target +0.12 rad, lift starts when the hand reaches it) with the 3 mm higher grasp of
`pi05/teacher.py`.  The teacher reads the cube and pad positions (privileged); the policy only sees images
and joints.  Returned values are absolute 12-D joint targets, executed through `PlaceEnv.step` exactly as a
policy action, so recorded labels replay exactly.
"""
from dataclasses import asdict, dataclass, field

import numpy as np

PHASES = ("approach", "close", "lift", "carry", "lower", "open", "retreat", "hold")


@dataclass
class PlaceTeacher:
    grasp_dz: float = .003
    index_extra: float = .12
    thumb_extra: float = .12
    carry_height: float = .05     # palm rise above the grasp pose while carrying
    slow_lift: float = .03
    slow_speed: float = .00125
    speed: float = .0025
    carry_speed: float = .002     # .003 dropped 4/200 cubes in transit on the far side of the pad
    lower_speed: float = .002
    fine_speed: float = .001
    fine_height: float = .02      # cube height below which lowering is slow
    release_height: float = .003  # cube height at which the hand opens
    retreat: float = .06
    hand_speed: float = .02
    phase: str = field(default="approach", init=False)

    def params(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "phase"}

    def reset(self) -> None:
        self.phase = "approach"
        self.carry_xyz = self.retreat_xyz = None

    def hand_goal(self, e) -> np.ndarray:
        goal = np.array(e.cfg["hand_target"], dtype=float)
        goal[1] += self.thumb_extra
        goal[2] += self.index_extra
        return np.clip(goal, *e.m.actuator_ctrlrange[e.hand].T)

    @staticmethod
    def open_hand(e) -> np.ndarray:
        hand = np.array(e.cfg["hand_target"], dtype=float)
        hand[1], hand[2] = e.cfg["opened_thumb2"], 0.
        return hand

    def __call__(self, env) -> np.ndarray:
        from build_scene import solve, R

        e = env.base
        e._vision()
        if (self.phase in ("approach", "close") and not e.grasp_locked and e.vision.visible
                and e.d.time - e.vision.last_seen < .15
                and np.linalg.norm(e.vision.last - (e.grasp_goal - np.array(e.cfg["grasp_xyz"])
                                                    + np.array(e.cfg["object_xyz"]))) > .002):
            try:
                e._set_plan(e.vision.last)
            except ValueError:
                e.plan_rejections += 1
        grasp = e.grasp_goal + np.array([0., 0., self.grasp_dz])
        up = grasp + np.array([0., 0., self.carry_height])
        goal = self.hand_goal(e)
        cube = e.d.xpos[e.ob]
        height = float(cube[2] - e.z0)
        xyz, hand, speed = e.target_xyz.copy(), e.hand_target.copy(), self.speed
        if self.phase == "approach":
            if np.linalg.norm(e.target_xyz - grasp) > .003:
                xyz = grasp
            else:
                self.phase = "close"
        if self.phase == "close":
            if np.all(e.hand_target[1:3] >= goal[1:3] - .001):
                self.phase = "lift"
            else:
                xyz, hand = grasp, goal
        if self.phase == "lift":
            if np.linalg.norm(e.target_xyz - up) < 1e-9:
                self.phase = "carry"
                self.carry_xyz = up + np.r_[env.target[:2] - cube[:2], 0.]
            else:
                xyz, hand = up, goal
                if e.target_xyz[2] - grasp[2] < self.slow_lift:
                    speed = self.slow_speed
        if self.phase == "carry":
            if np.linalg.norm(e.target_xyz - self.carry_xyz) < 1e-9:
                self.phase = "lower"
            else:
                xyz, hand, speed = self.carry_xyz, goal, self.carry_speed
        if self.phase == "lower":
            if height <= self.release_height or e.target_xyz[2] <= grasp[2] + 1e-9:
                self.phase = "open"
            else:
                xyz = np.r_[self.carry_xyz[:2], grasp[2]]
                hand = goal
                speed = self.fine_speed if height < self.fine_height else self.lower_speed
        if self.phase == "open":
            opened = self.open_hand(e)
            if np.all(np.abs(e.hand_target - opened) < 1e-9):
                self.phase = "retreat"
                self.retreat_xyz = e.target_xyz + np.array([0., 0., self.retreat])
            else:
                hand = opened
        if self.phase == "retreat":
            if np.linalg.norm(e.target_xyz - self.retreat_xyz) < 1e-9:
                self.phase = "hold"
            else:
                xyz = self.retreat_xyz
        point = e.target_xyz + np.clip(xyz - e.target_xyz, -speed, speed)
        accepted = False
        for scale in (1., .5, .25, .125):
            candidate = e.target_xyz + scale * (point - e.target_xyz)
            try:
                if e.planner.margin(candidate) < .003:
                    raise ValueError("Reach margin")
                e.q_target = solve(e.m, e.ikd, candidate, R, e.q_target)
                e.target_xyz = candidate
                accepted = True
                break
            except ValueError:
                e.ik_rejections += 1
        if not accepted:
            e.ik_failures += 1
        e.hand_target = np.clip(e.hand_target + np.clip(hand - e.hand_target, -self.hand_speed, self.hand_speed),
                                *e.m.actuator_ctrlrange[e.hand].T)
        e.grasp_locked |= e.hand_target[2] > .01
        command = np.r_[e.q_target, e.hand_target]
        return np.clip(command, *e.m.actuator_ctrlrange[env.ids].T)
