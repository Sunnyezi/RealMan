import time,argparse
from pathlib import Path
import mujoco,mujoco.viewer
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');p.add_argument('--mark-joint6',action='store_true');args=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/'models/scene.xml'));d=mujoco.MjData(m)
with mujoco.viewer.launch_passive(m,d) as v:
 v.cam.lookat[:]=[0,0,0.6];v.cam.distance=2.6;v.cam.azimuth=135;v.cam.elevation=-20
 while v.is_running():
  start=time.monotonic()
  if args.demo:
   import numpy as np
   for i in range(m.nu):
    if not mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i).startswith('right_'): continue
    lo,hi=m.actuator_ctrlrange[i];d.ctrl[i]=lo+(hi-lo)*0.25*(1-np.cos(d.time))
  mujoco.mj_step(m,d)
  if args.mark_joint6:
   jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'joint_6')
   anchor=d.xanchor[jid].copy();tail=anchor+np.array([0.22,0.22,0.07])
   with v.lock():
    v.user_scn.ngeom=2
    arrow=v.user_scn.geoms[0]
    mujoco.mjv_initGeom(arrow,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1.,0.25,0.05,1.]))
    mujoco.mjv_connector(arrow,mujoco.mjtGeom.mjGEOM_ARROW,0.012,tail,anchor)
    label=v.user_scn.geoms[1]
    mujoco.mjv_initGeom(label,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),tail+np.array([0.,0.,0.04]),np.eye(3).ravel(),np.array([1.,0.8,0.1,1.]))
    label.label='joint_6 (wrist rotation)'
  v.sync();time.sleep(max(0,m.opt.timestep-(time.monotonic()-start)))


