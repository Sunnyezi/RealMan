"""Run in the pinned OpenPI environment; never patch upstream tracked files."""

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "vendor/openpi/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["norm", "train", "serve", "inspect", "check-data"])
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--peak-lr", type=float, default=2e-5)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from adapter import training_config

    cfg = replace(
        training_config(args.repo_id, args.experiment, args.batch_size, args.steps),
        resume=args.resume,
        assets_base_dir=str(ROOT / "outputs/openpi_assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
    )
    if args.init_checkpoint:
        from openpi.training import weight_loaders
        if args.resume or not (args.init_checkpoint / "params").is_dir():
            parser.error("Warm start needs an existing params directory and cannot use --resume")
        cfg = replace(cfg, weight_loader=weight_loaders.CheckpointWeightLoader(str(args.init_checkpoint.resolve()/"params")))
    if args.mode == "train" and args.peak_lr != 2e-5:
        from openpi.training import optimizer
        if not 0 < args.peak_lr <= 2e-5:
            parser.error("Peak LR must be in (0,2e-5]")
        cfg = replace(cfg, lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,
                      peak_lr=args.peak_lr, decay_steps=max(101, args.steps), decay_lr=args.peak_lr*.1))
    if args.mode == "norm":
        from openpi.training import config

        # Register only in this process for the upstream statistics entry point.
        config.get_config = lambda name: cfg if name == cfg.name else (_ for _ in ()).throw(KeyError(name))
        load_script("compute_norm_stats").main(cfg.name)
    elif args.mode == "train":
        import time
        import jax
        from pipeline import fingerprint, write_json

        conversion = ROOT / "outputs" / (args.repo_id.replace("/", "_") + "_conversion.json")
        dataset = json.loads(conversion.read_text())
        if dataset["fingerprint"] != fingerprint() or dataset["split"] != "train":
            raise RuntimeError("Training requires train data from the current simulation version")
        stream = ROOT / "runs" / args.experiment / "live_state.npz"
        status = stream.with_name("grid_status.json")
        if not status.exists() or time.time() - status.stat().st_mtime > 10:
            raise RuntimeError(f"Start VNC viewer for {stream} before training (AGENTS.md)")
        state = json.loads(status.read_text())
        if state.get("active_envs", 0) < 1:
            raise RuntimeError("Viewer has not received a simulation state")
        if not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("CUDA GPU required for pi0.5 training")
        write_json(stream.with_name("training_manifest.json"), {
            "repo_id": args.repo_id, "experiment": args.experiment,
            "simulation_fingerprint": fingerprint(),
            "adapter_sha256": hashlib.sha256((ROOT / "adapter.py").read_bytes()).hexdigest(),
            "action_dim": 12, "steps": args.steps, "batch_size": args.batch_size,
            "base_weights": str(args.init_checkpoint.resolve()/"params") if args.init_checkpoint else "gs://openpi-assets/checkpoints/pi05_base/params",
            "warm_started": args.init_checkpoint is not None, "peak_lr": args.peak_lr,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "task": "pick up and hold; not pick and place", "simulation_only": True,
        })
        load_script("train").main(cfg)
    elif args.mode == "check-data":
        import numpy as np
        from openpi.training import data_loader
        loader = data_loader.create_data_loader(cfg, num_batches=1, shuffle=False)
        observation, actions = next(iter(loader))
        assert observation.state.shape == (args.batch_size, 32)
        assert actions.shape == (args.batch_size, 10, 32)
        assert np.isfinite(np.asarray(actions)).all()
        print(json.dumps({"state": list(observation.state.shape), "actions": list(actions.shape),
                          "image_keys": list(observation.images), "finite": True}))
    elif args.mode == "inspect":
        import jax
        import numpy as np
        from flax import nnx

        model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
        params = nnx.state(model)
        trainable = params.filter(cfg.trainable_filter)
        def count(tree):
            return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(tree))
        total, active = count(params), count(trainable)
        if not 0 < active < total:
            raise RuntimeError("Invalid LoRA freeze mask")
        print({"total_parameters": total, "trainable_parameters": active, "devices": str(jax.devices())})
    else:
        if not args.checkpoint:
            parser.error("serve requires --checkpoint")
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server
        from pipeline import fingerprint

        experiment = Path(args.checkpoint).resolve().parent.name
        manifest = json.loads((ROOT / "runs" / experiment / "training_manifest.json").read_text())
        if (manifest["repo_id"] != args.repo_id or manifest["simulation_fingerprint"] != fingerprint()
                or manifest["adapter_sha256"] != hashlib.sha256((ROOT / "adapter.py").read_bytes()).hexdigest()):
            raise RuntimeError("Checkpoint data, simulation, or adapter differs from the training manifest")

        policy = policy_config.create_trained_policy(cfg, args.checkpoint)
        websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=cfg.policy_metadata).serve_forever()


if __name__ == "__main__":
    main()
