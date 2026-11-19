"""Pick-and-place task on the existing contact scene: put the blue cube on a green target pad; simulation only.

The pad is a collision-free mocap body appended to `grasp/scene.xml`, so the robot/cube model, qpos layout,
physics and the pick-and-hold fingerprint stay unchanged.  The policy interface is the same as the hold task
(two 224x224 RGB images + 12 measured joints -> 12 absolute joint targets); only the prompt and the success
criterion differ.  Success needs a lifted cube resting inside the pad, released from low height, with the
hand open and withdrawn, for one continuous second.
"""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from pipeline import GRASP, ROOT, RGBEnv, fingerprint

PLACE_PROMPT = "Pick up the blue cube and place it on the green square."
SCENE = ROOT / "place" / "scene.xml"
HORIZON = 400                 # 20 s at 20 Hz
TARGET_OFFSET = (0., -.15)    # pad centre relative to the nominal cube position (reachable side, probed)
TARGET_RANGE = .04            # per-axis uniform randomisation of the pad centre
PAD_HALF = .04                # 8 x 8 cm pad
PAD_HALF_Z = .0005
TARGET_SEED_OFFSET = 9_000_000
LIFT = .03                    # the cube must have been lifted this high
TOLERANCE = .025              # cube centre to pad centre, xy
REST_HEIGHT = .005
REST_LINEAR = .02
REST_ANGULAR = .3
RELEASE_MAX = .03             # cube height when the last finger contact ended
CLEARANCE = .04               # palm rise above its grasp height relative to the cube
HOLD = 1.                     # seconds all conditions must hold


def build_scene_xml() -> str:
    root = ET.parse(GRASP / "scene.xml").getroot()
    root.set("model", "RGB-D floor pick and place")
    # Same meshes as grasp/scene.xml, resolved from pi05/place/.
    root.find("compiler").set("meshdir", "../../models/meshes")
    import json
    cfg = json.loads((GRASP / "config.json").read_text())
    centre = np.array(cfg["object_xyz"][:2]) + TARGET_OFFSET
    body = ET.SubElement(root.find("worldbody"), "body", name="place_target", mocap="true",
                         pos=f"{centre[0]:g} {centre[1]:g} {PAD_HALF_Z:g}")
    ET.SubElement(body, "geom", name="place_target_geom", type="box", size=f"{PAD_HALF:g} {PAD_HALF:g} {PAD_HALF_Z:g}",
                  rgba="0.1 0.75 0.2 1", contype="0", conaffinity="0", group="1")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def place_fingerprint() -> str:
    h = hashlib.sha256(fingerprint().encode())
    for path in (Path(__file__).resolve(), SCENE):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def place_ok(lifted: bool, error: float, resting: bool, contacts: int, clearance: float, floor: bool,
             release_height: float | None) -> bool:
    """All per-step conditions of a completed placement."""
    return bool(lifted and error <= TOLERANCE and resting and contacts == 0 and clearance >= CLEARANCE
                and not floor and release_height is not None and release_height <= RELEASE_MAX)


class PlaceEnv(RGBEnv):
    """RGBEnv on the place scene with a randomised pad and the placement success criterion."""

    def __init__(self, stream: Path | None = None):
        import os
        import sys
        os.environ.setdefault("MUJOCO_GL", "egl")
        sys.path.insert(0, str(GRASP))
        import mujoco
        from env import GraspEnv

        self.mujoco = mujoco
        self.base = GraspEnv(randomize=True, model=mujoco.MjModel.from_xml_path(str(SCENE)))
        self.ids = np.r_[self.base.arm, self.base.hand]
        self.qids = self.base.m.jnt_qposadr[self.base.m.actuator_trnid[self.ids, 0]]
        self.renderer = mujoco.Renderer(self.base.m, height=224, width=224)
        self.stream = stream
        self.mocap = int(self.base.m.body_mocapid[self.base.m.body("place_target").id])
        cfg = self.base.cfg
        self.nominal = np.r_[np.array(cfg["object_xyz"][:2]) + TARGET_OFFSET, PAD_HALF_Z]
        self.palm_cube_dz = float(cfg["grasp_xyz"][2] - cfg["object_xyz"][2])
        self.target = self.nominal.copy()

    def reset(self, seed: int) -> None:
        e = self.base
        e.reset(seed=seed)
        rng = np.random.default_rng(seed + TARGET_SEED_OFFSET)
        self.target = self.nominal + np.r_[rng.uniform(-TARGET_RANGE, TARGET_RANGE, 2), 0.]
        e.d.mocap_pos[self.mocap] = self.target
        self.mujoco.mj_kinematics(e.m, e.d)      # place the pad for rendering; no state integration
        self.lifted = self.grasped = self.over_target = self.released = False
        self.prev_contacts = 0
        self.release_height = None
        self.place_hold = self.outside = 0.
        self.floor_steps = 0
        self.publish("pi05 place reset")

    def observation(self) -> dict:
        obs = super().observation()
        obs["prompt"] = PLACE_PROMPT
        return obs

    def metrics(self) -> dict:
        e = self.base
        info = e._metrics()
        cube = e.d.xpos[e.ob]
        error = float(np.linalg.norm(cube[:2] - self.target[:2]))
        vel = np.zeros(6)
        self.mujoco.mj_objectVelocity(e.m, e.d, self.mujoco.mjtObj.mjOBJ_BODY, e.ob, vel, 0)   # [angular, linear]
        contacts, height, floor = info["hand_contacts"], info["height"], bool(info["table_collision"])
        if self.prev_contacts > 0 and contacts == 0:
            self.release_height = height
        self.prev_contacts = contacts
        self.grasped |= contacts >= 2
        self.lifted |= height >= LIFT
        self.over_target |= height >= LIFT and error <= TOLERANCE
        self.released |= self.lifted and contacts == 0
        self.floor_steps += floor
        resting = height < REST_HEIGHT and np.linalg.norm(vel[3:]) < REST_LINEAR and np.linalg.norm(vel[:3]) < REST_ANGULAR
        clearance = float(e.d.xpos[e.bid, 2] - cube[2] - self.palm_cube_dz)
        ok = place_ok(self.lifted, error, resting, contacts, clearance, floor, self.release_height)
        self.place_hold = self.place_hold + .05 if ok else 0.
        self.outside = self.outside + .05 if (self.lifted and contacts == 0 and resting and error > TOLERANCE) else 0.
        success = self.place_hold >= HOLD - 1e-8
        info.update(is_success=bool(success), dropped=bool(self.outside >= HOLD - 1e-8), hold_seconds=self.place_hold,
                    target_error_xy=error, place_target=self.target.tolist(), release_height=self.release_height,
                    retreat_clearance=clearance, resting=bool(resting), floor_contact_steps=self.floor_steps,
                    place_milestones=dict(grasped=self.grasped, lifted_3cm=self.lifted, over_target=self.over_target,
                                          released=self.released, placed=bool(success)))
        return info

    def step(self, action: np.ndarray) -> tuple[bool, dict]:
        e = self.base
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Expected 12 finite absolute joint targets in radians")
        e.d.ctrl[self.ids] = np.clip(action, *e.m.actuator_ctrlrange[self.ids].T)
        force = np.zeros(17)
        for _ in range(25):
            self.mujoco.mj_step(e.m, e.d)
            force += e.d.sensordata[e.sensor_adrs]
        e._update_touch(force / 25)
        e.steps += 1
        info = self.metrics()
        bad = bool(info["warnings"] or not np.isfinite(e.d.qpos).all())
        info["physics_failure"] = bad
        done = bool(info["is_success"] or info["dropped"] or bad or e.steps >= HORIZON)
        self.publish("pi05 place policy / absolute targets")
        return done, info

    @staticmethod
    def failure(info: dict) -> str | None:
        if info.get("physics_failure") or info.get("warnings"):
            return "physics"
        if info["is_success"]:
            return None
        m = info["place_milestones"]
        if info["dropped"]:
            return "placed_outside" if (info["release_height"] or 0.) <= RELEASE_MAX else "dropped_outside"
        if not m["grasped"]:
            return "no_grasp"
        if not m["lifted_3cm"]:
            return "no_lift"
        if not m["released"]:
            return "not_released"
        return "released_not_settled"

    def publish(self, stage: str) -> None:
        if self.stream is None:
            return
        import os
        self.stream.parent.mkdir(parents=True, exist_ok=True)
        e = self.base
        temp = self.stream.with_name(f".stream_{os.getpid()}.npz")
        np.savez(temp, qpos=e.d.qpos, qvel=e.d.qvel, ctrl=e.d.ctrl, time=e.d.time, touch_force=e.touch_force,
                 stage=stage, place_target=self.target)
        os.replace(temp, self.stream)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Write pi05/place/scene.xml from grasp/scene.xml")
    parser.add_argument("--check", action="store_true", help="fail if the committed scene differs")
    args = parser.parse_args()
    text = build_scene_xml()
    if args.check:
        if SCENE.read_text() != text:
            raise SystemExit("pi05/place/scene.xml is stale")
    else:
        SCENE.parent.mkdir(exist_ok=True)
        SCENE.write_text(text)
    print(SCENE, hashlib.sha256(text.encode()).hexdigest())
