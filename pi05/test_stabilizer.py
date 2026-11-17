import unittest

import numpy as np

from stabilizer import RobotState, Settings, Stabilizer


class StabilizerTests(unittest.TestCase):
    def create(self) -> Stabilizer:
        limits = np.tile([-2., 2.], (12, 1))
        return Stabilizer(np.zeros(12), limits, np.full(6, .5), np.eye(3),
                          lambda q: q[:3].copy(), lambda p, r, q: np.r_[p, np.zeros(3)], np.zeros(3))

    def state(self, c: Stabilizer, touch: bool = False) -> RobotState:
        q = c.command.copy()
        if touch:
            q[7:9] = [.3, .4]
        return RobotState(q, c.command[:3].copy(), .5 if touch else 0., .5 if touch else 0.)

    def test_rate_limits_and_nonfinite_rejection(self) -> None:
        c = self.create()
        before = c.command.copy()
        result = c.update(self.state(c), np.r_[np.full(6, 100.), np.zeros(6)])
        self.assertGreater(np.max(np.abs(result[:6]-before[:6])), 0.)
        self.assertTrue(np.all(np.abs(result[:6]-before[:6]) <= 1.2*.05+1e-9))
        c = self.create()
        self.start_secure(c)
        before = c.command.copy()
        result = c.update(self.state(c))
        self.assertGreater(np.max(np.abs(result[6:]-before[6:])), 0.)
        self.assertTrue(np.all(np.abs(result[6:]-before[6:]) <= .4*.05+1e-9))
        c = self.create()
        with self.assertRaises(ValueError):
            c.update(self.state(c), np.full(12, np.nan))

    def start_secure(self, c: Stabilizer) -> None:
        raw = np.zeros(12)
        raw[8] = .5
        c.update(self.state(c), raw)
        self.assertEqual(c.phase, "secure")

    def test_no_lift_without_bilateral_contact_and_closure(self) -> None:
        for thumb, index, closure in ((.5, 0., True), (0., .5, True), (.5, .5, False)):
            c = self.create()
            self.start_secure(c)
            for _ in range(30):
                state = self.state(c, closure)
                state = RobotState(state.joints, state.palm_position, thumb, index)
                if not closure:
                    state = RobotState(np.zeros(12), state.palm_position, .5, .5)
                c.update(state)
            self.assertEqual(c.phase, "secure")
            self.assertAlmostEqual(c.command[2], 0.)

    def test_contact_confirmation_lift_and_hold(self) -> None:
        c = self.create()
        self.start_secure(c)
        for _ in range(180):
            c.update(self.state(c, True))
        self.assertEqual(c.phase, "hold")
        locked = c.command.copy()
        for _ in range(20):
            np.testing.assert_allclose(c.update(self.state(c, True)), locked)
        self.assertAlmostEqual(locked[2], Settings().lift_distance)

    def test_no_contact_timeout_returns_to_policy(self) -> None:
        c = self.create()
        self.start_secure(c)
        for _ in range(42):
            raw = np.zeros(12) if c.needs_policy else None
            c.update(self.state(c, False), raw)
        self.assertEqual(c.phase, "policy")

    def test_open_hand_until_visual_alignment(self) -> None:
        c = self.create()
        c.visual_grasp[:] = [.02, .01, -.02]
        raw = np.ones(12)
        c.update(self.state(c), raw)
        self.assertEqual(c.phase, "align")
        np.testing.assert_allclose(c.command[6:], c.open_hand)
        for _ in range(30):
            c.update(self.state(c))
            if c.phase == "secure":
                break
            np.testing.assert_allclose(c.command[6:], c.open_hand)
        self.assertEqual(c.phase, "secure")
        np.testing.assert_allclose(c.command[:3], c.visual_grasp, atol=.001)

    def test_feedback_contract_has_no_object_truth(self) -> None:
        self.assertEqual(set(RobotState.__dataclass_fields__), {"joints", "palm_position", "thumb_force", "index_force"})
        self.assertFalse(self.create().describe()["uses_object_ground_truth"])

    def test_slow_policy_has_bounded_approach_time(self) -> None:
        c = self.create()
        for _ in range(42):
            c.update(self.state(c), np.zeros(12) if c.needs_policy else None)
        self.assertEqual(c.phase, "secure")
        self.assertLessEqual(c.transitions[0]["step"]*c.cfg.dt, c.cfg.policy_seconds+c.cfg.dt)

    def test_static_tracking_error_does_not_sink_command(self) -> None:
        c = self.create()
        joints = np.zeros(12)
        joints[2] = -.01
        state = RobotState(joints, np.array([0., 0., -.01]), 0., 0.)
        for _ in range(30):
            c.update(state, np.zeros(12))
        np.testing.assert_allclose(c.command[:6], 0., atol=1e-12)

    def test_contact_anchor_preserves_command_under_tracking_error(self) -> None:
        c = self.create()
        joints = np.zeros(12)
        joints[2] = -.01
        joints[7:9] = [.3, .4]
        state = RobotState(joints, np.array([0., 0., -.01]), .5, .5)
        raw = np.zeros(12)
        raw[8] = .5
        c.update(state, raw)
        self.assertEqual(c.phase, "secure")
        np.testing.assert_allclose(c.anchor, 0., atol=1e-12)
        np.testing.assert_allclose(c.command[:6], 0., atol=1e-12)


if __name__ == "__main__":
    unittest.main()
