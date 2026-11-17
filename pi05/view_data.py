"""Show training demonstrations during supervised updates, explicitly labeled as replay."""

import argparse
from pathlib import Path
import time

import numpy as np

from pipeline import RGBEnv, validate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    args = parser.parse_args()
    episodes = sorted(args.raw.glob("episode_*"))
    if not episodes:
        parser.error("No recorded episodes")
    env = RGBEnv()
    try:
        while True:
            for path in episodes:
                metadata = validate_episode(path)
                env.reset(metadata["seed"])
                with np.load(path / "trajectory.npz", allow_pickle=False) as data:
                    for action in data["actions"]:
                        start = time.monotonic()
                        env.step(action)
                        env.stream = args.stream
                        env.publish("pi05 supervised training / demonstration replay")
                        env.stream = None
                        time.sleep(max(0, 0.05 - (time.monotonic() - start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
