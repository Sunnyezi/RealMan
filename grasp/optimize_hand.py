from pathlib import Path
import numpy as np,mujoco,json
from scipy.optimize import least_squares
P=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m)
base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_base_link')
def forward(q,finger):
 d.qpos[:19]=0
 for name,x in [('right_thumb_1_joint',q[0]),('right_thumb_2_joint',q[1]),(f'right_{finger}_1_joint',q[2])]:
  j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);d.qpos[m.jnt_qposadr[j]]=x
 for _ in range(3):
  for i in range(m.neq):
   j1,j2=m.eq_obj1id[i],m.eq_obj2id[i];d.qpos[m.jnt_qposadr[j1]]=m.eq_data[i,0]+m.eq_data[i,1]*d.qpos[m.jnt_qposadr[j2]]
 mujoco.mj_forward(m,d)
def pad(name):
 b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);Q=d.xmat[base].reshape(3,3)
 return Q.T@(d.xpos[b]-d.xpos[base]),Q.T@d.xmat[b].reshape(3,3)[:,2]
rng=np.random.default_rng(1);out=[]
for finger in ['index','middle','ring']:
 for ts in [1,3,4]:
  for fs in [1,2,3]:
   def fun(q):
    forward(q,finger);a,na=pad(f'right_thumb_force_sensor_{ts}');b,nb=pad(f'right_{finger}_force_sensor_{fs}')
    return np.r_[b-a-.04*na,.02*(na+nb)]
   best=None
   for _ in range(6):
    res=least_squares(fun,rng.uniform([0,0,0],[1.1641,.5864,1.4381]),bounds=([0,0,0],[1.1641,.5864,1.4381]),max_nfev=70)
    loss=np.linalg.norm(fun(res.x))
    if best is None or loss<best[0]:best=(loss,res.x)
   forward(best[1],finger);a,na=pad(f'right_thumb_force_sensor_{ts}');b,nb=pad(f'right_{finger}_force_sensor_{fs}')
   out.append(dict(loss=best[0],finger=finger,thumb_sensor=ts,finger_sensor=fs,q=best[1].tolist(),a=a.tolist(),b=b.tolist(),na=na.tolist(),nb=nb.tolist(),center=((a+b)/2).tolist()))
out.sort(key=lambda x:x['loss']);(P/'hand_pose_candidates.json').write_text(json.dumps(out,indent=2));print(json.dumps(out[:4],indent=2))
