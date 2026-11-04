import json,numpy as np,mujoco,time
from env import P
from build_scene import solve,R
m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));seen=set();cases=[]
for path in sorted((P/'checks').glob('ik_failed_requests_*.json')):
 for row in json.load(open(path)):
  key=tuple(np.round(row['target'],5))
  if row['proven_outside_tolerance'] or key in seen:continue
  seen.add(key);d=mujoco.MjData(m)
  try:q=solve(m,d,np.array(row['target']),R);success=True;error=''
  except ValueError as e:success=False;error=str(e)
  cases.append(dict(seed=row['seed'],step=row['step'],target=row['target'],excess=row['excess'],multiple_initial_guesses_solved=success,error=error))
(P/'checks/ik_restarts.json').write_text(json.dumps(cases,indent=2));print(json.dumps(cases,indent=2))
