"""Contracts that can be checked without allocating a full pi0.5 model."""

import unittest

import numpy as np

from adapter import RealManInputs, RealManOutputs


class ContractTests(unittest.TestCase):
    def sample(self) -> dict:
        return {"observation/image": np.zeros((224, 224, 3), np.uint8),
                "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
                "observation/state": np.zeros(12, np.float32), "prompt": "pick",
                "actions": np.zeros((10, 12), np.float32), "object_truth": np.ones(7)}

    def test_measured_state_and_images_only(self) -> None:
        result = RealManInputs()(self.sample())
        self.assertEqual(set(result), {"state", "image", "image_mask", "actions", "prompt"})
        self.assertFalse(result["image_mask"]["right_wrist_0_rgb"])
        self.assertEqual(result["actions"].shape, (10, 12))

    def test_policy_output_unpads_twelve_active_joints(self) -> None:
        result = RealManOutputs()({"actions": np.ones((10, 32))})
        self.assertEqual(result["actions"].shape, (10, 12))

    def test_old_synergy_and_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RealManInputs()(self.sample() | {"actions": np.zeros((10, 7))})
        with self.assertRaises(ValueError):
            RealManInputs()(self.sample() | {"observation/state": np.full(12, np.nan)})
        with self.assertRaises(ValueError):
            RealManOutputs()({"actions": np.zeros((10, 7))})
        with self.assertRaises(ValueError):
            RealManOutputs()({"actions": np.full((10, 32), np.nan)})


if __name__ == "__main__":
    unittest.main()
