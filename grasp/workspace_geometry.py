import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,numpy as np,mujoco
from pathlib import Path
from env import P
m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
j=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint_{i}') for i in range(1,7)];bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_base_link')
shoulder=d.xanchor[j[1]].copy();reach=np.linalg.norm(d.xanchor[j[2]]-shoulder)+np.linalg.norm(d.xanchor[j[4]]-d.xanchor[j[2]])
offset=d.xmat[bid].reshape(3,3).T@(d.xpos[bid]-d.xanchor[j[4]])
cfg=json.load(open(P/'config.json'));R=np.array(cfg['rotation']);rows=[]
for seed in range(30000,30100):
 delta=np.r_[np.random.default_rng(seed).uniform(-.05,.05,2),0.];p=np.array(cfg['grasp_xyz'])+delta
 rows.append(dict(seed=seed,delta=delta.tolist(),grasp_excess=float(np.linalg.norm(p-R@offset-shoulder)-reach),lift10_excess=float(np.linalg.norm(p+[0,0,.10]-R@offset-shoulder)-reach),lift115_excess=float(np.linalg.norm(p+[0,0,.115]-R@offset-shoulder)-reach)))
print('shoulder',shoulder,'reach',reach,'offset',offset)
print('grasp / lift10 / lift115 outside exact bound:',*[sum(x[k]>.003 for x in rows) for k in ['grasp_excess','lift10_excess','lift115_excess']])
(P/'checks/ik_workspace_geometry.json').write_text(json.dumps(dict(shoulder=shoulder.tolist(),reach=reach,tool_offset=offset.tolist(),rows=rows),indent=2))
