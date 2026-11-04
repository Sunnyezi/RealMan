import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,numpy as np,mujoco,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import env as module
from env import GraspEnv,P
cfg=json.load(open(P/'config.json'));geo=json.load(open(P/'checks/ik_workspace_geometry.json'));R=np.array(cfg['rotation']);offset=np.array(geo['tool_offset']);shoulder=np.array(geo['shoulder']);reach=geo['reach'];original=module.solve
rows=[];context={}
def traced(m,d,target,rotation,q0=None):
 try:return original(m,d,target,rotation,q0)
 except ValueError as error:
  delta=float(np.linalg.norm(target-rotation@offset-shoulder)-reach)
  rows.append(dict(seed=context['seed'],step=context['step'],target=np.asarray(target).tolist(),q0=np.asarray(q0).tolist(),excess=delta,proven_outside_tolerance=bool(delta>.003*np.sqrt(1+(np.linalg.norm(offset)/.25)**2)+1e-6),error=str(error)))
  raise
module.solve=traced;torch.set_num_threads(1)
e=GraspEnv(randomize=True);run=P/'runs/random_stage_smoke';norm=VecNormalize.load(str(run/'vecnormalize.pkl'),DummyVecEnv([lambda:e]));norm.training=False;norm.norm_reward=False;model=PPO.load(str(run/'policy.zip'),device='cpu');episodes=[]
try:
 for seed in range(30000,30100):
  context['seed']=seed;start=len(rows);obs,_=e.reset(seed=seed)
  for step in range(200):
   context['step']=step;a,_=model.predict(norm.normalize_obs(obs[None,:]),deterministic=True);obs,_,term,trunc,info=e.step(a[0])
   if term or trunc:break
  failures=rows[start:];episodes.append(dict(seed=seed,failures=len(failures),proven=sum(r['proven_outside_tolerance'] for r in failures),is_success=info['is_success']))
  if (seed-29999)%10==0:print('Replayed',seed-29999,'episodes; IK failures',len(rows),flush=True)
finally:norm.close()
summary=dict(episodes=len(episodes),episodes_with_failures=sum(x['failures']>0 for x in episodes),failed_requests=len(rows),geometrically_proven_requests=sum(x['proven_outside_tolerance'] for x in rows),episodes_all_failed_requests_proven=sum(x['failures']>0 and x['proven']==x['failures'] for x in episodes))
(P/'checks/ik_failed_requests.json').write_text(json.dumps(rows,indent=2));(P/'checks/ik_replay_summary.json').write_text(json.dumps(dict(summary=summary,episodes=episodes),indent=2));print(json.dumps(summary),flush=True)
