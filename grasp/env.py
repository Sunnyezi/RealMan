import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,time,hashlib
import numpy as np,mujoco,gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from build_scene import solve,R
from vision import RGBD
P=Path(__file__).resolve().parent
class GraspEnv(gym.Env):
 metadata={'render_modes':['rgb_array'],'render_fps':20}
 def __init__(self,randomize=False,publish=False,render_mode=None):
  self.m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));self.d=mujoco.MjData(self.m);self.ikd=mujoco.MjData(self.m)
  self.cfg=json.load(open(P/'config.json'));self.randomize=randomize;self.publish=publish;self.render_mode=render_mode
  self.vision=RGBD(self.m);self.overview=None
  self.bid=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'right_base_link');self.ob=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'object');self.og=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'object_geom');self.oj=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,'object_free');self.oa=self.m.jnt_qposadr[self.oj]
  self.arm=np.array([mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_ACTUATOR,f'joint_{i}_position') for i in range(1,7)])
  self.hand=np.array([i for i in range(self.m.nu) if mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_ACTUATOR,i).startswith('right_')]);assert len(self.hand)==6
  self.robot_nq=int(self.oa);self.robot_nv=int(self.m.jnt_dofadr[self.oj]);self.action_space=spaces.Box(-1,1,(9,),np.float32)
  self.observation_space=spaces.Box(-np.inf,np.inf,(self.robot_nq+self.robot_nv+3+9+6+3+2+9+9,),np.float32)
  self.prev_action=np.zeros(9);self.next_vision=0;self.steps=0
 def _publish(self):
  if self.publish:
   path=P/f'.live_{os.getpid()}.npz';np.savez(path,qpos=self.d.qpos,qvel=self.d.qvel,ctrl=self.d.ctrl,time=self.d.time);os.replace(path,P/'live_state.npz')
 def _vision(self,force=False):
  if force or self.d.time+1e-8>=self.next_vision:
   self.vision.update(self.d);self.next_vision=float(self.d.time)+.1
 def _obs(self):
  # No object qpos/qvel, truth pose, contact IDs, or segmentation IDs enter observations.
  vel=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.bid,vel,0)
  return np.concatenate([self.d.qpos[:self.robot_nq],self.d.qvel[:self.robot_nv],self.d.xpos[self.bid],self.d.xmat[self.bid],vel,self.vision.last-self.d.xpos[self.bid],[float(self.vision.visible),min(10.,self.d.time-self.vision.last_seen)],self.prev_action,self.target_xyz,self.hand_target]).astype(np.float32)
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed);mujoco.mj_resetData(self.m,self.d)
  self.d.qpos[:6]=self.cfg['home_q'];self.d.ctrl[self.arm]=self.cfg['home_q'];self.d.ctrl[self.hand]=0
  self.target_xyz=np.array(self.cfg['home_xyz']);self.hand_target=np.array(self.cfg['hand_target']);self.hand_target[1:3]=0;self.d.ctrl[self.hand]=self.hand_target
  for i,u in enumerate(self.hand):self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[u,0]]]=self.hand_target[i]
  for _ in range(3):
   for i in range(self.m.neq):self.d.qpos[self.m.jnt_qposadr[self.m.eq_obj1id[i]]]=self.m.eq_data[i,1]*self.d.qpos[self.m.jnt_qposadr[self.m.eq_obj2id[i]]]
  self.hand_target=self.hand_target.copy();self.q_target=np.array(self.cfg['home_q']);self.prev_action=np.zeros(9)
  offset=self.np_random.uniform(-.05,.05,2) if self.randomize else np.zeros(2)
  self.d.qpos[self.oa:self.oa+3]=np.array(self.cfg['object_xyz'])+np.r_[offset,0.]
  for _ in range(100):mujoco.mj_step(self.m,self.d)
  self.d.time=0.;self.steps=0;self.hold=0.;self.ik_failures=0;self.lost=0;self.max_height=0.;self.ever_lifted=False
  self.z0=float(self.d.xpos[self.ob,2]);self.vision.last[:]=0;self.vision.last_seen=-10.;self.next_vision=0;self._vision(True);self._publish()
  return self._obs(),{}
 def _metrics(self):
  fingers=set();table_hit=False;object_table=False
  for c in self.d.contact:
   a,b=int(c.geom1),int(c.geom2)
   if self.og in (a,b):
    other=b if a==self.og else a;bid=int(self.m.geom_bodyid[other]);name=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_BODY,bid) or ''
    if name.startswith('right_'):fingers.add(name.split('_force_sensor')[0] if '_force_sensor' in name else name.rsplit('_',1)[0])
    if self.m.geom_contype[other]==4:object_table=True
   else:
    types=[int(self.m.geom_contype[a]),int(self.m.geom_contype[b])]
    if 4 in types and (2 in types or 8 in types):table_hit=True
  height=float(self.d.xpos[self.ob,2]-self.z0);self.max_height=max(self.max_height,height)
  vel=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.ob,vel,0)
  stable=height>=.10 and np.linalg.norm(vel[3:])<.08 and np.linalg.norm(vel[:3])<1. and len(fingers)>=2 and not object_table and not table_hit
  self.hold=self.hold+.05 if stable else 0.;self.ever_lifted|=height>.02
  dropped=bool(self.d.xpos[self.ob,2]<self.cfg['object_xyz'][2]-.04 or (self.ever_lifted and height<.005))
  warnings=int(sum(w.number for w in self.d.warning));success=self.hold>=2.-1e-8
  return dict(is_success=bool(success),height=height,hold_seconds=self.hold,hand_contacts=len(fingers),table_collision=table_hit,dropped=dropped,warnings=warnings,ik_failures=self.ik_failures,vision_lost_rate=self.lost/max(1,self.steps),max_height=self.max_height)
 def step(self,action):
  action=np.asarray(action,dtype=np.float64)
  if action.shape!=(9,) or not np.isfinite(action).all():raise ValueError('Action must be finite and have shape (9,)')
  action=np.clip(action,-1,1);candidate=np.clip(self.target_xyz+.005*action[:3],[.20,-.20,.30],[.65,.20,.60])
  try:
   q=solve(self.m,self.ikd,candidate,R,self.q_target);self.q_target=q;self.target_xyz=candidate
  except ValueError:self.ik_failures+=1
  self.hand_target=np.clip(self.hand_target+.06*action[3:],self.m.actuator_ctrlrange[self.hand,0],self.m.actuator_ctrlrange[self.hand,1]);self.d.ctrl[self.arm]=self.q_target;self.d.ctrl[self.hand]=self.hand_target
  oldaction=self.prev_action.copy();self.prev_action=action.copy()
  for _ in range(25):mujoco.mj_step(self.m,self.d)
  self.steps+=1;self._vision();self.lost+=not self.vision.visible;info=self._metrics()
  center=self.d.xpos[self.bid]+self.d.xmat[self.bid].reshape(3,3)@R.T@(np.array(self.cfg['object_xyz'])-np.array(self.cfg['grasp_xyz']));dist=np.linalg.norm(center-self.d.xpos[self.ob])
  reward=2*np.exp(-15*dist)+.25*min(info['hand_contacts'],3)+10*np.clip(info['height'],0,.12)+.5*(self.hold>0)+20*info['is_success']-.01*np.mean(action**2)-.02*np.mean((action-oldaction)**2)-.5*info['table_collision']-10*info['dropped']
  bad=bool(info['warnings'] or not np.isfinite(self.d.qpos).all());info['physics_failure']=bad;self._publish()
  return self._obs(),float(reward),bool(info['is_success'] or info['dropped'] or bad),bool(self.steps>=200),info
 def render(self):
  if self.overview is None:self.overview=mujoco.Renderer(self.m,480,640)
  opt=mujoco.MjvOption();opt.geomgroup[3]=0;self.overview.update_scene(self.d,camera='overview',scene_option=opt);return self.overview.render().copy()
 def close(self):
  self.vision.close()
  if self.overview:self.overview.close()
