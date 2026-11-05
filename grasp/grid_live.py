"""Display genuine independent environment states in a single MuJoCo viewer."""
import os
os.environ['MUJOCO_GL']='glfw'
from pathlib import Path
import argparse,json,time,math
import numpy as np,mujoco,mujoco.viewer
P=Path(__file__).parent
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stream',type=Path,default=P/'live_state.npz');parser.add_argument('--envs',type=int,default=16);args=parser.parse_args()
 stream=args.stream.resolve();n=args.envs
 if not 1<=n<=16:raise ValueError('Viewer supports 1-16 cells')
 from view_assets import ensure_model
 visual=ensure_model() if n>1 else P/'scene.xml'
 m=mujoco.MjModel.from_xml_path(str(visual if visual.exists() and n>1 else P/'scene.xml'));data=[mujoco.MjData(m) for _ in range(n)]
 cfg=json.load(open(P/'config.json'));floor=m.geom('floor').id
 for i in range(m.ngeom):
  if i!=floor and m.geom_group[i]!=3:m.geom_group[i]=5
 m.geom_size[floor,:2]=20
 for d in data:d.qpos[:6]=cfg['home_q'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)
 regions=json.load(open(P/'sensor_map.json'));site_ids=[m.site(r['site']).id for r in regions]
 cols=math.ceil(math.sqrt(n));rows=math.ceil(n/cols);spacing=1.4
 draw=mujoco.MjvOption();draw.geomgroup[:]=0;draw.geomgroup[5]=1;draw.sitegroup[:]=0;draw.sitegroup[4]=1
 force=np.zeros((n,17));times=np.zeros(n);active=0;fresh=False;stage='waiting';last_stamp=0
 # The UI owns a separate immutable model/data; source pose updates never hold its lock.
 ui_model=mujoco.MjModel.from_xml_path(str(visual if visual.exists() and n>1 else P/'scene.xml'));ui_data=mujoco.MjData(ui_model)
 for i in range(ui_model.ngeom):
  if i!=floor and ui_model.geom_group[i]!=3:ui_model.geom_group[i]=5
 ui_model.geom_size[floor,:2]=20
 with mujoco.viewer.launch_passive(ui_model,ui_data,show_left_ui=True,show_right_ui=True) as viewer:
  viewer.cam.lookat[:]=[(cols-1)*spacing/2+.3,(rows-1)*spacing/2,.18]
  viewer.cam.distance=max(cols,rows)*2.5;viewer.cam.azimuth=135;viewer.cam.elevation=-50
  viewer.opt.geomgroup[5]=0;viewer.opt.geomgroup[3]=0;viewer.opt.sitegroup[:]=0
  viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0;viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
  while viewer.is_running():
   start=time.monotonic()
   if stream.exists():
    try:
     stamp=stream.stat().st_mtime
     if stamp!=last_stamp:
      with np.load(stream) as state:
       q=np.atleast_2d(state['qpos']);v=np.atleast_2d(state['qvel']);u=np.atleast_2d(state['ctrl']);f=np.atleast_2d(state['touch_force']);t=np.atleast_1d(state['time']);active=min(n,len(q))
       stage=str(state['stage']) if 'stage' in state else 'single warmstart environment'
       for i in range(active):
        data[i].qpos[:]=q[i];data[i].qvel[:]=v[i];data[i].ctrl[:]=u[i];data[i].time=float(t[i]);times[i]=t[i];force[i]=f[i];mujoco.mj_kinematics(m,data[i]);mujoco.mj_comPos(m,data[i]);mujoco.mj_camlight(m,data[i])
      # Native right-hand controls show ENV 00; playback remains policy-controlled.
      if active:
       ui_data.qpos[:]=data[0].qpos;ui_data.qvel[:]=data[0].qvel;ui_data.ctrl[:]=data[0].ctrl;ui_data.time=data[0].time
      last_stamp=stamp
     fresh=time.time()-stamp<2
    except (OSError,ValueError,KeyError):fresh=False
   viewer.user_scn.ngeom=0
   for i in range(n):
    offset=np.array([(i%cols)*spacing,(i//cols)*spacing,0.])
    if i<active:
     for k,sid in enumerate(site_ids):m.site_rgba[sid]=[min(1,force[i,k]/3),max(0,1-force[i,k]/3),0,.5]
     before=viewer.user_scn.ngeom
     mujoco.mjv_addGeoms(m,data[i],draw,mujoco.MjvPerturb(),mujoco.mjtCatBit.mjCAT_ALL,viewer.user_scn)
     for k in range(before,viewer.user_scn.ngeom):viewer.user_scn.geoms[k].pos[:]+=offset
    if viewer.user_scn.ngeom>=viewer.user_scn.maxgeom:raise RuntimeError('Viewer geometry budget exceeded')
    geom=viewer.user_scn.geoms[viewer.user_scn.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),offset+[.3,0,.78],np.eye(3).ravel(),np.array([1.,.85,.3,1.]))
    geom.label=f'ENV {i:02d} t={times[i]:.1f}s' if i<active else f'ENV {i:02d} WAITING'
    viewer.user_scn.ngeom+=1
   if viewer.user_scn.ngeom<viewer.user_scn.maxgeom:
    geom=viewer.user_scn.geoms[viewer.user_scn.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),np.array([.3,-.7,.9]),np.eye(3).ravel(),np.array([1.,1.,1.,1.]))
    geom.label=(('LIVE ' if fresh else 'WAITING ') + stage)[:90];viewer.user_scn.ngeom+=1
   sync_start=time.perf_counter();viewer.sync();sync_ms=(time.perf_counter()-sync_start)*1000
   if stream.parent.exists():
    unique=len(np.unique(np.round(np.stack([d.qpos for d in data[:active]]),8),axis=0)) if active else 0
    status=dict(pid=os.getpid(),wall_time=time.time(),fresh=fresh,active_envs=active,total_cells=n,stage=stage,times=times.tolist(),unique_states=unique,sync_ms=sync_ms,loop_ms=(time.monotonic()-start)*1000,camera=dict(azimuth=float(viewer.cam.azimuth),elevation=float(viewer.cam.elevation),distance=float(viewer.cam.distance)),display_mesh_faces=int(m.nmeshface))
    path=stream.with_name('grid_status.json');temp=path.with_suffix('.tmp');temp.write_text(json.dumps(status));os.replace(temp,path)
   time.sleep(max(0,.05-(time.monotonic()-start)))
if __name__=='__main__':main()
