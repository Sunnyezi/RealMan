# RM65 V + RH56DFTP right: MuJoCo setup

Directory: /home/szy2025/Desktop/RealMan
Conda: realman-mujoco (Python 3.11)

Start visible viewer on existing VNC desktop :15:

    bash /home/szy2025/Desktop/RealMan/launch_view.sh

Finger motion demo:

    bash /home/szy2025/Desktop/RealMan/launch_view.sh --demo

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

Rebuild: conda activate realman-mujoco; python build_model.py
Files: models/scene.xml, models/combined.urdf, validation.json, viewer.log.
