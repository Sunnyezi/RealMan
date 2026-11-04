from pathlib import Path
import numpy as np,mujoco
from build_scene import solve,R
class Planner:
 def __init__(self,m,cfg):
  self.m=m;self.d=mujoco.MjData(m);self.cfg=cfg;mujoco.mj_forward(m,self.d)
  ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{i}') for i in range(1,7)]
  self.shoulder=self.d.xanchor[ids[1]].copy();self.reach=np.linalg.norm(self.d.xanchor[ids[2]]-self.shoulder)+np.linalg.norm(self.d.xanchor[ids[4]]-self.d.xanchor[ids[2]])
  bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_base_link');self.offset=self.d.xmat[bid].reshape(3,3).T@(self.d.xpos[bid]-self.d.xanchor[ids[4]])
 def margin(self,xyz):return self.reach-np.linalg.norm(np.asarray(xyz)-R@self.offset-self.shoulder)
 def plan(self,object_estimate,start_xyz,start_q):
  grasp=np.asarray(object_estimate)+np.array(self.cfg['grasp_xyz'])-np.array(self.cfg['object_xyz']);pre=grasp+[0,0,.08];lift=grasp+[0,0,self.cfg['lift_distance']]
  q=np.array(start_q);prev=np.asarray(start_xyz);minimum=1.;waypoints=[]
  for endpoint in [pre,grasp,lift]:
   for alpha in np.linspace(.2,1,5):
    xyz=prev+(endpoint-prev)*alpha;margin=self.margin(xyz);minimum=min(minimum,margin)
    if margin<.004:raise ValueError(f'Insufficient reach margin {margin}')
    q=solve(self.m,self.d,xyz,R,q);self.d.qpos[:6]=q;mujoco.mj_forward(self.m,self.d)
    for c in self.d.contact:
     kinds={int(self.m.geom_contype[c.geom1]),int(self.m.geom_contype[c.geom2])}
     if kinds=={4,8} and c.dist<-.0005:raise ValueError('Arm/table collision along planned path')
    waypoints.append(dict(xyz=xyz.tolist(),q=q.tolist()))
   prev=endpoint
  return dict(grasp=grasp,lift=lift,pre=pre,minimum_margin=minimum,waypoints=waypoints)
