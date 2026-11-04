from pathlib import Path
import datetime,shutil
P=Path('/home/szy2025/Desktop/RealMan');f=P/'view.py';shutil.copy2(f,P/'backups'/('view_before_joint6_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.py'))
s=f.read_text().replace('import mujoco,mujoco.viewer','import mujoco,mujoco.viewer\nimport numpy as np')
s=s.replace("p.add_argument('--demo',action='store_true');args=p.parse_args()","p.add_argument('--demo',action='store_true');p.add_argument('--mark-joint6',action='store_true');args=p.parse_args()")
s=s.replace('mujoco.mj_step(m,d);v.sync();time.sleep(max(0,m.opt.timestep-(time.monotonic()-start)))', '''mujoco.mj_step(m,d)
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
  v.sync();time.sleep(max(0,m.opt.timestep-(time.monotonic()-start)))''')
f.write_text(s)
