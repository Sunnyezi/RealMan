from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np,mujoco,json
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
P=Path(__file__).resolve().parent; ROOT=P.parent
POSE=json.load(open(Path(__file__).resolve().parent/'grasp_pose.json'))
R=np.array(POSE['rotation'])
def solve(m,d,target,rotation,q0=None):
 ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{i}') for i in range(1,7)]
 qa=m.jnt_qposadr[ids];bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_base_link')
 lo=m.jnt_range[ids,0]+.001;hi=m.jnt_range[ids,1]-.001
 def fun(q):
  d.qpos[qa]=q;mujoco.mj_forward(m,d)
  return np.r_[d.xpos[bid]-target,.25*Rotation.from_matrix(rotation@d.xmat[bid].reshape(3,3).T).as_rotvec()]
 rng=np.random.default_rng(9)
 seeds=[np.clip(q0,lo,hi)] if q0 is not None else [np.array([0.,.5,-1.,0.,1.,0.])]+[rng.uniform(lo,hi) for _ in range(30)]
 best=None
 for seed in seeds:
  opt=least_squares(fun,np.clip(seed,lo,hi),bounds=(lo,hi),max_nfev=180,xtol=1e-9,ftol=1e-9,gtol=1e-9)
  err=np.linalg.norm(fun(opt.x))
  if best is None or err<best[0]:best=(err,opt.x.copy())
  if err<1e-5:break
 if best[0]>.003: raise ValueError(f'IK residual {best[0]} for {target}')
 return best[1]
def build():
 r=E.parse(ROOT/'models/scene.xml').getroot();r.set('model','RGB-D grasp training')
 r.find('compiler').set('meshdir','../models/meshes')
 r.find('option').set('timestep','0.002')
 w=r.find('worldbody')
 for b in w.iter('body'):
  for g in list(b.findall('geom')):
   if g.get('group')!='1':b.remove(g);continue
   g.set('contype','0');g.set('conaffinity','0')
   name=b.get('name','')
   if name=='right_base_link':
    E.SubElement(b,'geom',name='palm_collision',type='box',pos='0 0.003 0.081',size='0.038 0.014 0.074',contype='2',conaffinity='5',group='3',rgba='0 1 0 0.2',friction='1.2 0.01 0.001',condim='4',density='0')
   elif name.startswith('right_'):
    c=E.SubElement(b,'geom',**dict(g.attrib));c.set('name',name+'_collision');c.set('group','3');c.set('contype','2');c.set('conaffinity','5');c.set('density','0');c.set('friction','1.2 0.01 0.001');c.set('condim','4');c.set('rgba','0 1 0 0.15')
   elif name.startswith('link_'):
    c=E.SubElement(b,'geom',**dict(g.attrib));c.set('name',name+'_collision');c.set('group','3');c.set('contype','8');c.set('conaffinity','5');c.set('density','0')
 floor=w.find('geom[@name="floor"]');floor.set('contype','4');floor.set('conaffinity','11')
 E.SubElement(w,'geom',name='table',type='box',pos='0.55 0 0.255',size='0.25 0.25 0.025',rgba='0.35 0.3 0.25 1',contype='4',conaffinity='11',friction='0.9 0.01 0.001')
 obj=E.SubElement(w,'body',name='object',pos='0.60 0 0.301');E.SubElement(obj,'freejoint',name='object_free')
 E.SubElement(obj,'geom',name='object_geom',type='box',size='0.02 0.02 0.02',mass='0.05',rgba='0.05 0.1 0.95 1',contype='1',conaffinity='14',friction='1.2 0.01 0.001',condim='4')
 # Camera extrinsics are synthetic; aim fixed in camera body at the nominal grasp region.
 cam_body=r.find('.//body[@name="camera_link"]')
 cam=E.SubElement(cam_body,'camera',name='rgbd',pos='0 0 0',fovy='65')
 E.SubElement(w,'camera',name='overview',pos='1.15 -1.1 1.0',xyaxes='0.72 0.69 0 -0.3 0.31 0.9')
 act=r.find('actuator')
 for a in act:
  if a.get('joint','').startswith('joint_'):a.set('gainprm','650');a.set('biasprm','0 -650 -45')
  elif a.get('joint','').startswith('right_'):a.set('gainprm','10');a.set('biasprm','0 -10 -0.25');a.set('forcerange','-1.5 1.5')
 (P/'scene.xml').write_bytes(E.tostring(r))
 m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m)
 q=solve(m,d,np.array(POSE['home_xyz']),R)
 d.qpos[:6]=q;mujoco.mj_forward(m,d)
 cb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'camera_link')
 direction=np.array([.60,0,.30])-d.xpos[cb];z=-direction/np.linalg.norm(direction);x=np.cross(np.array([0,0,1.]),z);x/=np.linalg.norm(x);y=np.cross(z,x)
 local=d.xmat[cb].reshape(3,3).T@np.column_stack([x,y,z]);xy=local[:,:2].T.ravel()
 cam.set('xyaxes',' '.join(map(str,xy)));E.indent(r);E.ElementTree(r).write(P/'scene.xml')
 config=dict(home_q=q.tolist(),home_xyz=POSE['home_xyz'],grasp_xyz=POSE['grasp_xyz'],hand_target=POSE['hand_target'],hand_center=POSE['hand_center'],object_xyz=[.60,0,.301],rotation=R.tolist(),physics_dt=.002,control_dt=.05,vision_dt=.1,horizon=200,seed=42)
 (P/'config.json').write_text(json.dumps(config,indent=2));print(config)
if __name__=='__main__':build()
