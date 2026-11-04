import os
os.environ['MUJOCO_GL']='glfw'
from pathlib import Path
import time,json,argparse
import numpy as np,mujoco,mujoco.viewer
from build_scene import solve,R
parser=argparse.ArgumentParser();parser.add_argument('--scripted',action='store_true');args=parser.parse_args()
P=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m)
cfg=json.load(open(P/'config.json'));home=np.array(cfg['home_q']);down=solve(m,d,np.array(cfg['grasp_xyz']),R,home);up=solve(m,d,np.array(cfg['grasp_xyz'])+np.array([0,0,.115]),R,down)
cycle=0;next_log=0
with mujoco.viewer.launch_passive(m,d) as v:
 v.cam.lookat[:]=[.35,0,.35];v.cam.distance=1.45;v.cam.azimuth=135;v.cam.elevation=-25;v.opt.geomgroup[3]=0
 while v.is_running():
  start=time.monotonic();live=P/'live_state.npz';external=False
  if live.exists() and not args.scripted:
   try:
    with np.load(live) as s:d.qpos[:]=s['qpos'];d.qvel[:]=s['qvel'];d.ctrl[:]=s['ctrl'];d.time=float(s['time'])
    mujoco.mj_forward(m,d);external=True
   except (OSError,ValueError):pass
  if not external:
   if d.time>=10 or d.time==0:
    mujoco.mj_resetData(m,d);d.qpos[:6]=home;d.ctrl[:6]=home
    opened=np.array(cfg['hand_target']);opened[1:3]=0
    for i in range(7,m.nu): d.qpos[m.jnt_qposadr[m.actuator_trnid[i,0]]]=opened[i-7]
    for _ in range(3):
     for i in range(m.neq):d.qpos[m.jnt_qposadr[m.eq_obj1id[i]]]=m.eq_data[i,1]*d.qpos[m.jnt_qposadr[m.eq_obj2id[i]]]
    mujoco.mj_forward(m,d);cycle+=1
   t=d.time
   if t<2:q=home+(down-home)*min(1,t/2)
   elif t<4:q=down
   else:q=down+(up-down)*min(1,(t-4)/2)
   d.ctrl[:6]=q
   target=np.array(cfg['hand_target']);target[1:3]+=.12;opened=target.copy();opened[1:3]=0
   d.ctrl[7:]=opened+(target-opened)*np.clip((t-2)/1.5,0,1)
   for _ in range(5):mujoco.mj_step(m,d)
  phase='SCRIPTED CHECK (not PPO)' if not external else ('LIVE WORKER STATE' if time.time()-live.stat().st_mtime<2 else 'WORKER PAUSED / LAST FRAME')
  with v.lock():
   v.user_scn.ngeom=1;g=v.user_scn.geoms[0]
   mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),np.array([.30,0,.62]),np.eye(3).ravel(),np.array([1.,.8,.1,1.]));g.label=f'{phase}  t={d.time:.1f}s'
  v.sync()
  if start>=next_log:
   (P/'live_status.json').write_text(json.dumps(dict(wall_time=time.time(),sim_time=d.time,cycle=cycle,external=external,pid=os.getpid())));next_log=start+1
  time.sleep(max(0,.01-(time.monotonic()-start)))
