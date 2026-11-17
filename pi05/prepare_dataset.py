"""Convert recordings and compute OpenPI normalization without repeated NPZ/image decoding."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from adapter import training_config
from pipeline import ROOT, write_json


def main() -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    from openpi.shared import normalize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    paths = sorted(args.raw.glob("episode_*"))
    metas = [json.loads((p / "metadata.json").read_text()) for p in paths]
    if (not metas or len({m["split"] for m in metas}) != 1 or metas[0]["split"] not in ("train", "validation")
            or not all(m["is_success"] for m in metas) or len({m["fingerprint"] for m in metas}) != 1):
        parser.error("Need one successful train/validation partition from one simulation version")
    if (HF_LEROBOT_HOME / args.repo_id).exists():
        raise FileExistsError(args.repo_id)
    dataset = LeRobotDataset.create(repo_id=args.repo_id, robot_type="rm65_rh56_absolute12", fps=20,
        features={"image": {"dtype": "image", "shape": (224,224,3), "names": ["height","width","channel"]},
                  "wrist_image": {"dtype": "image", "shape": (224,224,3), "names": ["height","width","channel"]},
                  "state": {"dtype": "float32", "shape": (12,), "names": ["state"]},
                  "actions": {"dtype": "float32", "shape": (12,), "names": ["actions"]}},
        image_writer_threads=2, image_writer_processes=0)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    frames = 0
    for index, (path, meta) in enumerate(zip(paths, metas, strict=True)):
        with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
            # NpzFile.__getitem__ decompresses the whole array each time. Materialize once per episode.
            data = {key: archive[key] for key in ("image", "wrist_image", "state", "actions", "timestamp")}
        n = len(data["state"])
        if n < 10 or not np.allclose(data["timestamp"], np.arange(n) * .05):
            raise ValueError(f"Invalid episode timing: {path}")
        for key in ("state", "actions"):
            if data[key].shape != (n,12) or data[key].dtype != np.float32 or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid {key}: {path}")
        for key in ("image", "wrist_image"):
            if data[key].shape != (n,224,224,3) or data[key].dtype != np.uint8:
                raise ValueError(f"Invalid {key}: {path}")
        for i in range(n):
            dataset.add_frame({key: data[key][i] for key in ("image","wrist_image","state","actions")}
                              | {"task": meta["prompt"]})
        stats["state"].update(data["state"])
        # Match pinned LeRobot _get_query_indices, including repeated terminal actions.
        windows = np.minimum(np.arange(n)[:, None] + np.arange(10)[None, :], n - 1)
        stats["actions"].update(data["actions"][windows])
        dataset.save_episode()
        frames += n
        print(f"converted {index + 1}/{len(paths)} episodes; frames={frames}", flush=True)
    cfg = replace(training_config(args.repo_id, "prepare", 1, 1),
                  assets_base_dir=str(ROOT / "outputs/openpi_assets"))
    normalize.save(cfg.assets_dirs / args.repo_id, {key: value.get_statistics() for key, value in stats.items()})
    write_json(ROOT / "outputs" / (args.repo_id.replace("/", "_") + "_conversion.json"),
               dict(repo_id=args.repo_id, raw=str(args.raw.resolve()), episodes=len(paths), frames=frames,
                    fingerprint=metas[0]["fingerprint"], split=metas[0]["split"],
                    normalization="upstream RunningStats, per-episode batches, horizon=10, clamped episode boundaries"))


if __name__ == "__main__":
    main()
