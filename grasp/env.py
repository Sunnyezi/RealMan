import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,time,hashlib
import numpy as np,mujoco,gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from build_scene import solve,R
from vision import RGBD
from planner import Planner
P=Path(__file__).resolve().parent
class GraspEnv(gym.Env):
 metadata={'render_modes':['rgb_array'],'render_fps':20}
 def __init__(self,randomize=False,publish=False,render_mode=None):
  self.m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));self.d=mujoco.MjData(self.m);self.ikd=mujoco.MjData(self.m)
  self.cfg=json.load(open(P/'config.json'));self.randomize=randomize;self.publish=P/'live_state.npz' if publish is True else publish;self.render_mode=render_mode
  self.vision=RGBD(self.m);self.overview=None;self.planner=Planner(self.m,self.cfg)
  self.sensor_map=json.load(open(P/'sensor_map.json'));self.sensor_adrs=np.array([self.m.sensor_adr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_SENSOR,r['sensor'])] for r in self.sensor_map]);self.touch_force=np.zeros(17);self.touch_flags=np.zeros(17)
  self.bid=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'right_base_link');self.ob=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,'object');self.og=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'object_geom');self.oj=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,'object_free');self.oa=self.m.jnt_qposadr[self.oj]
  self.arm=np.array([mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_ACTUATOR,f'joint_{i}_position') for i in range(1,7)])
  self.hand=np.array([i for i in range(self.m.nu) if mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_ACTUATOR,i).startswith('right_')]);assert len(self.hand)==6
  self.robot_nq=int(self.oa);self.robot_nv=int(self.m.jnt_dofadr[self.oj]);self.action_space=spaces.Box(-1,1,(9,),np.float32)
  self.observation_space=spaces.Box(-np.inf,np.inf,(self.robot_nq+self.robot_nv+3+9+6+3+2+9+9+34+7,),np.float32)
  self.prev_action=np.zeros(9);self.next_vision=0;self.steps=0
 def _publish(self):
  if self.publish:
   path=self.publish.with_name(f'.live_{os.getpid()}.npz');np.savez(path,qpos=self.d.qpos,qvel=self.d.qvel,ctrl=self.d.ctrl,time=self.d.time,touch_force=self.touch_force,grasp_goal=self.grasp_goal,lift_goal=self.lift_goal);os.replace(path,self.publish)
 def _vision(self,force=False):
  if force or self.d.time+1e-8>=self.next_vision:
   self.vision.update(self.d);self.next_vision=float(self.d.time)+.1
 def _obs(self):
  # No object qpos/qvel, truth pose, contact IDs, or segmentation IDs enter observations.
  vel=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.bid,vel,0)
  return np.concatenate([self.d.qpos[:self.robot_nq],self.d.qvel[:self.robot_nv],self.d.xpos[self.bid],self.d.xmat[self.bid],vel,self.vision.last-self.d.xpos[self.bid],[float(self.vision.visible),min(10.,self.d.time-self.vision.last_seen)],self.prev_action,self.target_xyz,self.hand_target,self.touch_force/self.cfg['force_scale_N'],self.touch_flags,self.grasp_goal-self.d.xpos[self.bid],self.lift_goal-self.d.xpos[self.bid],[float(self.plan_valid)]]).astype(np.float32)
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed);mujoco.mj_resetData(self.m,self.d)
  self.d.qpos[:6]=self.cfg['home_q'];self.d.ctrl[self.arm]=self.cfg['home_q'];self.d.ctrl[self.hand]=0
  self.target_xyz=np.array(self.cfg['home_xyz']);self.hand_target=np.array(self.cfg['hand_target']);self.hand_target[1]=self.cfg['opened_thumb2'];self.hand_target[2]=0;self.d.ctrl[self.hand]=self.hand_target
  for i,u in enumerate(self.hand):self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[u,0]]]=self.hand_target[i]
  for _ in range(3):
   for i in range(self.m.neq):self.d.qpos[self.m.jnt_qposadr[self.m.eq_obj1id[i]]]=self.m.eq_data[i,1]*self.d.qpos[self.m.jnt_qposadr[self.m.eq_obj2id[i]]]
  self.hand_target=self.hand_target.copy();self.q_target=np.array(self.cfg['home_q']);self.prev_action=np.zeros(9)
  offset=self.np_random.uniform(-self.cfg['random_radius'],self.cfg['random_radius'],2) if self.randomize else np.zeros(2)
  self.d.qpos[self.oa:self.oa+3]=np.array(self.cfg['object_xyz'])+np.r_[offset,0.]
  if options and 'object_yaw_rad' in options:
   yaw = options['object_yaw_rad']
   self.d.qpos[self.oa+3:self.oa+7] = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
  for _ in range(100):mujoco.mj_step(self.m,self.d)
  self.d.time=0.;self.steps=0;self.hold=0.;self.ik_failures=0;self.lost=0;self.max_height=0.;self.ever_lifted=False
  self.z0=float(self.d.xpos[self.ob,2]);self.vision.last[:]=0;self.vision.last_seen=-10.;self.next_vision=0;self._vision(True)
  self.grasp_locked=False;self.plan_valid=False;self.plan_rejections=0;self.action_projections=0;self.ik_rejections=0;self.touch_force[:]=0;self.touch_flags[:]=0
  if not self.vision.visible:raise RuntimeError('Object not visible at reset')
  self._set_plan(self.vision.last)
  self._publish()
  return self._obs(),{}
 def _set_plan(self,estimate):
  plan=self.planner.plan(estimate,self.target_xyz,self.q_target)
  self.grasp_goal=plan['grasp'];self.lift_goal=plan['lift'];self.pregrasp_goal=plan['pre'];self.path_plan=plan;self.plan_valid=True
 def _update_touch(self,mean_force):
  self.touch_force=.4*np.clip(mean_force,0,30)+.6*self.touch_force
  self.touch_flags=(self.touch_force>self.cfg['contact_threshold_N']).astype(float)
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
  return dict(is_success=bool(success),height=height,hold_seconds=self.hold,hand_contacts=len(fingers),table_collision=table_hit,dropped=dropped,warnings=warnings,ik_failures=self.ik_failures,vision_lost_rate=self.lost/max(1,self.steps),max_height=self.max_height,ik_rejections=self.ik_rejections,action_projections=self.action_projections,plan_valid=bool(self.plan_valid),plan_rejections=self.plan_rejections,touch_force_N=self.touch_force.tolist(),touch_flags=self.touch_flags.tolist())
 def step(self,action):
  action=np.asarray(action,dtype=np.float64)
  if action.shape!=(9,) or not np.isfinite(action).all():raise ValueError('Action must be finite and have shape (9,)')
  action=np.clip(action,-1,1)
  if not self.grasp_locked and self.vision.visible and self.d.time-self.vision.last_seen<.15 and np.linalg.norm(self.vision.last-(self.grasp_goal-np.array(self.cfg['grasp_xyz'])+np.array(self.cfg['object_xyz'])))>.002:
   try:self._set_plan(self.vision.last)
   except ValueError:self.plan_rejections+=1
  low=self.grasp_goal-np.array([.025,.025,.004]);high=np.maximum(self.lift_goal,self.cfg['home_xyz'])+np.array([.025,.025,0.])
  desired=self.target_xyz+.005*action[:3];candidate=np.clip(desired,low,high)
  projected=not np.allclose(candidate,desired,atol=1e-9)
  accepted=False;rejected=False
  for scale in [1.,.5,.25,.125]:
   point=self.target_xyz+scale*(candidate-self.target_xyz)
   try:
    if self.planner.margin(point)<.003:raise ValueError('Reach margin')
    q=solve(self.m,self.ikd,point,R,self.q_target);self.q_target=q;self.target_xyz=point;accepted=True;projected|=scale<1;break
   except ValueError:rejected=True
  self.ik_rejections+=int(rejected);self.action_projections+=int(projected)
  if not accepted:self.ik_failures+=1
  self.hand_target=np.clip(self.hand_target+.06*action[3:],self.m.actuator_ctrlrange[self.hand,0],self.m.actuator_ctrlrange[self.hand,1]);self.d.ctrl[self.arm]=self.q_target;self.d.ctrl[self.hand]=self.hand_target
  self.grasp_locked |= self.hand_target[2]>.01
  oldaction=self.prev_action.copy();self.prev_action=action.copy()
  force_sum=np.zeros(17)
  for _ in range(25):
   mujoco.mj_step(self.m,self.d);force_sum+=self.d.sensordata[self.sensor_adrs]
  self._update_touch(force_sum/25)
  self.steps+=1;self._vision();self.lost+=not self.vision.visible;info=self._metrics()
  center=self.d.xpos[self.bid]+self.d.xmat[self.bid].reshape(3,3)@R.T@(np.array(self.cfg['object_xyz'])-np.array(self.cfg['grasp_xyz']));dist=np.linalg.norm(center-self.d.xpos[self.ob])
  reward=2*np.exp(-15*dist)+.25*min(info['hand_contacts'],3)+10*np.clip(info['height'],0,.12)+.5*(self.hold>0)+20*info['is_success']-.01*np.mean(action**2)-.02*np.mean((action-oldaction)**2)-.5*info['table_collision']-10*info['dropped']-.2*rejected-.05*projected-2*max(0,info['height']-.115)
  bad=bool(info['warnings'] or not np.isfinite(self.d.qpos).all());info['physics_failure']=bad;self._publish()
  return self._obs(),float(reward),bool(info['is_success'] or info['dropped'] or bad),bool(self.steps>=self.cfg['horizon']),info
 def render(self):
  if self.overview is None:self.overview=mujoco.Renderer(self.m,480,640)
  opt=mujoco.MjvOption();opt.geomgroup[3]=0;self.overview.update_scene(self.d,camera='overview',scene_option=opt);return self.overview.render().copy()
 def close(self):
  self.vision.close()
  if self.overview:self.overview.close()
