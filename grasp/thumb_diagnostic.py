import mujoco,numpy as np,json
from pathlib import Path
from build_scene import solve,R
P=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m)
x=.44;y=-.015;z=.33
home=solve(m,d,np.array([x,y,.44]),R);down=solve(m,d,np.array([x,y,z]),R,home);up=solve(m,d,np.array([x,y,z+.12]),R,down)
d.qpos[:6]=home;d.ctrl[:6]=home;mujoco.mj_forward(m,d)
rows=[]
for k in range(3500):
 t=k*.002;d.ctrl[:6]=home+(down-home)*min(1,t/2) if t<2 else down if t<3 else down+(up-down)*min(1,(t-3)/1.5)
 d.ctrl[7:]=m.actuator_ctrlrange[7:,1]*np.array([1.,1.,.6,.6,.6,.6])*min(1,t/1.5);mujoco.mj_step(m,d)
 if k in [1250,1450,1700,2300,3400]:
  contacts=[]
  for i,c in enumerate(d.contact):
   names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(v)) or '' for v in [c.geom1,c.geom2]]
   if 'object_geom' in names:
    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);contacts.append(dict(geoms=names,normal=c.frame[:3].tolist(),normal_force=float(force[0]),penetration=float(-c.dist)))
  joints=[]
  for name in ['right_thumb_1_joint','right_thumb_2_joint','right_thumb_3_joint','right_thumb_4_joint']:
   j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name);joints.append(dict(name=name,actual=float(d.qpos[m.jnt_qposadr[j]])))
  rows.append(dict(time=d.time,thumb_targets=d.ctrl[7:9].tolist(),thumb_joints=joints,contacts=contacts))
(P/'thumb_diagnostic.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
