# RM65 V + RH56DFTP right: MuJoCo setup

The launch commands below run from the repository root. Use the `realman-mujoco` Conda environment (Python 3.11).

Start visible viewer on existing VNC desktop :15:

    bash launch_view.sh

Finger motion demo:

    bash launch_view.sh --demo

VNC server listens on localhost:5915. On Windows:

    ssh -N -L 15900:127.0.0.1:5915 uestc-server

Connect a VNC client to 127.0.0.1:15900 using your existing VNC password.
The Viewer right-side Control panel exposes 6 arm, 1 camera rotation, and 6 hand position targets.

Model sources and limits:
- Arm: https://github.com/RealManRobot/rm_models, RM65/urdf/RM65-6FB-V.
  Requested RM65-6F-V exact model was not found; 6FB-V is explicitly a provisional substitute.
- Hand: https://github.com/AlexArutiunian/tactile-rh56dftp, FTP_right_hand.urdf.
  FTP right CAD, not a DFX substitute. Exact -2R revision is not independently verified.
- Palm frame attached to link_6 with a zero translation and zero rotation (mounting faces aligned).
  Mesh mounting faces coincide at local Z=0; no added adapter geometry.
- All URDF mimic joints are represented by MuJoCo equality constraints.
- Gravity compensation is enabled. Gains/force limits are initial simulation settings.
- Robot collisions disabled for the initial visualization scene; no grasp/contact validation.
- Tactile sensor geometry included, but no calibrated tactile response or six-axis force sensor electronics.
- No connection to physical robot hardware.

## Model build and repository layout

Activate the environment, rebuild the model, and run the motion check from the repository root.

```bash
conda activate realman-mujoco
python build_model.py
python check_motion.py
```

| Path | Contents |
| --- | --- |
| [sources/](sources/) | Upstream URDFs, meshes, and source metadata |
| [models/](models/) | Combined URDF, MuJoCo scenes, and assembled meshes |
| [docs/model-validation/](docs/model-validation/) | Build report and recorded motion, mount, and arm replacement checks |
| [docs/screenshots/](docs/screenshots/) | Recorded VNC views and joint 6 annotation |
| [tools/legacy/](tools/legacy/) | Scripts from the initial model setup and troubleshooting |
| [grasp/](grasp/) | Grasp training, evaluation, and released checkpoints |

Server-local backups are in `backups/`, and archived setup logs are in `work/root-runtime/`. The grasp pipeline writes runtime files within `grasp/`.

## RGB-D floor grasp

The arm base and the cube share the ground plane at `z = 0`. The grasp pipeline uses 17 MuJoCo touch regions, RGB-D grasp targets, and reachability checks. See the [grasp setup and validation](grasp/README.zh-CN.md).

Start the VNC viewer with `bash grasp/grasp.sh live` and the scripted grasp loop with `bash grasp/grasp.sh demo`. The checkpoints in `grasp/runs/visual_touch/` were trained on the raised-platform scene. Training or evaluating a policy for the floor scene requires a checkpoint produced for that scene.

CUDA training is the default for the grasp policy. See [GPU setup and commands](grasp/GPU.zh-CN.md) and [physical limit provenance](docs/physical_limits.zh-CN.md).
