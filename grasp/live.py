import os
os.environ['MUJOCO_GL']='glfw'
import argparse,time,json
from pathlib import Path
import numpy as np,mujoco,mujoco.viewer
P=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--stream',type=Path,default=P/'live_state.npz')
args=parser.parse_args()
stream=args.stream.resolve()
m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);cfg=json.load(open(P/'config.json'));d.qpos[:6]=cfg['home_q'];mujoco.mj_forward(m,d)
regions=json.load(open(P/'sensor_map.json'));site_ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,r['site']) for r in regions]
with mujoco.viewer.launch_passive(m,d) as v:
 v.cam.lookat[:]=[.32,0,.22];v.cam.distance=1.25;v.cam.azimuth=135;v.cam.elevation=-25;v.opt.geomgroup[3]=0;v.opt.sitegroup[4]=1
 while v.is_running():
  start=time.monotonic();force=np.zeros(17);source=stream;fresh=False
  if source.exists():
   try:
    with np.load(source) as s:
     d.qpos[:]=s['qpos'];d.qvel[:]=s['qvel'];d.ctrl[:]=s['ctrl'];d.time=float(s['time']);force=s['touch_force'].copy()
    mujoco.mj_forward(m,d);fresh=time.time()-source.stat().st_mtime<2
   except (OSError,ValueError):pass
  for i,sid in enumerate(site_ids):m.site_rgba[sid]=[min(1,force[i]/3),max(0,1-force[i]/3),0,.5]
  with v.lock():
   v.user_scn.ngeom=1;g=v.user_scn.geoms[0];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),np.array([.32,0,.68]),np.eye(3).ravel(),np.array([1.,.8,.1,1.]));g.label='LIVE' if fresh else 'WAITING / LAST FRAME'
   g.label+=f' t={d.time:.1f}s touch={sum(force>.02)}/17'
  v.sync()
  # The viewer starts before train.py creates the run directory.
  if stream.parent.exists():(stream.with_name('live_status.json')).write_text(json.dumps(dict(wall_time=time.time(),sim_time=d.time,fresh=fresh,active_regions=int(sum(force>.02)),touch_force_N=force.tolist(),pid=os.getpid())))
  time.sleep(max(0,.02-(time.monotonic()-start)))
