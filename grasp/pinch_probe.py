import mujoco,numpy as np,json,time,os
from pathlib import Path
from build_scene import solve,R
P=Path(__file__).parent

def run(squeeze=.12,delta=(0,0,0),publish=False,realtime=False):
 m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);cfg=json.load(open(P/'config.json'))
 grasp=np.array(cfg['grasp_xyz'])+delta;home=solve(m,d,grasp+[0,0,.10],R);down=solve(m,d,grasp,R,home);up=solve(m,d,grasp+[0,0,.115],R,down)
 target=np.array(cfg['hand_target']);target[1]+=squeeze;target[2]+=squeeze;target=np.clip(target,m.actuator_ctrlrange[7:,0],m.actuator_ctrlrange[7:,1]);opened=target.copy();opened[1:3]=0.
 d.qpos[:6]=home;d.ctrl[:6]=home
 for i in range(7,m.nu):j=m.actuator_trnid[i,0];d.qpos[m.jnt_qposadr[j]]=opened[i-7]
 # Initialize passive mimic joints consistently.
 for _ in range(3):
  for i in range(m.neq):d.qpos[m.jnt_qposadr[m.eq_obj1id[i]]]=m.eq_data[i,1]*d.qpos[m.jnt_qposadr[m.eq_obj2id[i]]]
 mujoco.mj_forward(m,d);obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');og=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'object_geom');z0=.3009;hold=0;maxheight=0;contact_log=[]
 for k in range(5000):
  start=time.monotonic();t=k*.002
  d.ctrl[:6]=home+(down-home)*min(1,t/2) if t<2 else down if t<4 else down+(up-down)*min(1,(t-4)/2)
  d.ctrl[7:]=opened+(target-opened)*np.clip((t-2)/1.5,0,1)
  mujoco.mj_step(m,d);height=d.xpos[obj,2]-z0;maxheight=max(maxheight,height)
  touch=set();table=False;tablehand=False
  for c in d.contact:
   if og in (c.geom1,c.geom2):
    g=c.geom2 if c.geom1==og else c.geom1;name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
    if name.startswith('right_'):touch.add(name.split('_')[1])
    if m.geom_contype[g]==4:table=True
   elif 4 in [m.geom_contype[c.geom1],m.geom_contype[c.geom2]] and any(x in [2,8] for x in [m.geom_contype[c.geom1],m.geom_contype[c.geom2]]):tablehand=True
  vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,obj,vel,0)
  good=height>=.10 and len(touch)>=2 and not table and not tablehand and np.linalg.norm(vel[3:])<.08 and np.linalg.norm(vel[:3])<1
  hold=hold+.002 if good else 0
  if k%500==0:contact_log.append(dict(t=round(t,2),object=d.xpos[obj].round(4).tolist(),touch=sorted(touch),height=round(height,4),hold=hold,tablehand=tablehand))
  if publish and k%25==0:
   tmp=P/f'.probe_{os.getpid()}.npz';np.savez(tmp,qpos=d.qpos,qvel=d.qvel,ctrl=d.ctrl,time=d.time);os.replace(tmp,P/'live_state.npz')
  if realtime:time.sleep(max(0,.002-(time.monotonic()-start)))
 return dict(success=hold>=2.,squeeze=squeeze,delta=list(delta),final=d.xpos[obj].tolist(),max_height=maxheight,hold=hold,warnings=[int(w.number) for w in d.warning],trace=contact_log)
if __name__=='__main__':
 import sys
 if '--live' in sys.argv:print(json.dumps(run(publish=True,realtime=True)),flush=True)
 else:
  for squeeze in [0,.025,.05,.08,.12]:
   for z in [-.005,0,.005]:
    print(json.dumps(run(squeeze,delta=(0,0,z))),flush=True)
