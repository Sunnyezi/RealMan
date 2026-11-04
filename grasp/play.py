import os,time,json
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from env import GraspEnv,P
from train import fingerprint

def play(checkpoint,randomize=False,seed=20000):
 checkpoint=Path(checkpoint);run=checkpoint.parent;meta=json.load(open(run/'run.json'))
 if meta['fingerprint']!=fingerprint():raise RuntimeError('Environment fingerprint mismatch')
 torch.set_num_threads(1);e=GraspEnv(randomize=randomize,publish=True);vec=DummyVecEnv([lambda:e]);norm=checkpoint.with_name(checkpoint.stem+'_vecnormalize.pkl')
 if not norm.exists() and checkpoint.stem.startswith('checkpoint_'):norm=checkpoint.with_name(checkpoint.stem.replace('checkpoint_','checkpoint_vecnormalize_',1)+'.pkl')
 if not norm.exists():norm=run/'vecnormalize.pkl'
 v=VecNormalize.load(str(norm),vec);v.training=False;v.norm_reward=False;model=PPO.load(str(checkpoint),device='cpu');episode=0
 try:
  while True:
   pid=int((P/'live.pid').read_text());os.kill(pid,0)
   obs,_=e.reset(seed=seed+episode)
   for _ in range(e.cfg['horizon']):
    start=time.monotonic();a,_=model.predict(v.normalize_obs(obs[None,:]),deterministic=True);obs,_,term,trunc,info=e.step(a[0]);time.sleep(max(0,.05-(time.monotonic()-start)))
    if term or trunc:break
   episode+=1;(P/'policy_play_status.json').write_text(json.dumps(dict(episode=episode,checkpoint=str(checkpoint),seed=seed+episode-1,info=info),indent=2))
 except (ProcessLookupError,FileNotFoundError):pass
 finally:v.close()
