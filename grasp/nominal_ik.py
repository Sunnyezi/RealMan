import json,numpy as np,mujoco
from env import P
from build_scene import solve,R
m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);cfg=json.load(open(P/'config.json'));geo=json.load(open(P/'checks/ik_workspace_geometry.json'));shoulder=np.array(geo['shoulder']);offset=np.array(geo['tool_offset']);rows=[]
for dz in [0,.1,.115,.13,.15]:
 target=np.array(cfg['grasp_xyz'])+[0,0,dz]
 try:
  q=solve(m,d,target,R,np.array(cfg['home_q']));warm=True;error=''
 except ValueError as e:warm=False;error=str(e)
 row=dict(lift_target_m=dz,target=target.tolist(),outside_reach_m=float(np.linalg.norm(target-R@offset-shoulder)-geo['reach']),warm_start_solved=warm,error=error)
 if not warm and row['outside_reach_m']<.004:
  try:solve(m,d,target,R);row['multi_start_solved']=True
  except ValueError:row['multi_start_solved']=False
 rows.append(row)
(P/'checks/ik_nominal_targets.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
