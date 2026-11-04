import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from env import GraspEnv,P
from train import fingerprint

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path);p.add_argument('--loop',action='store_true');p.add_argument('--episodes',type=int,default=100);p.add_argument('--seed',type=int,default=10000);p.add_argument('--viewer',action='store_true');p.add_argument('--randomize',action='store_true');args=p.parse_args()
 if args.loop:
  from play import play
  play(args.checkpoint,args.randomize,args.seed);return
 if args.episodes<=0:raise SystemExit('episodes must be positive')
 run=args.checkpoint.parent;meta=json.load(open(run/'run.json'))
 if meta['fingerprint']!=fingerprint():raise SystemExit('Environment differs from the trained environment')
 torch.set_num_threads(1);e=GraspEnv(randomize=args.randomize,publish=args.viewer);base=DummyVecEnv([lambda:e]);norm=args.checkpoint.with_name(args.checkpoint.stem+'_vecnormalize.pkl')
 if not norm.exists() and args.checkpoint.stem.startswith('checkpoint_'):norm=args.checkpoint.with_name(args.checkpoint.stem.replace('checkpoint_','checkpoint_vecnormalize_',1)+'.pkl')
 if not norm.exists():norm=run/'vecnormalize.pkl'
 env=VecNormalize.load(str(norm),base);env.training=False;env.norm_reward=False;model=PPO.load(str(args.checkpoint),device='cpu');results=[]
 try:
  for k in range(args.episodes):
   obs,_=e.reset(seed=args.seed+k);done=False;total=0;info={}
   while not done:
    start=time.monotonic();action,_=model.predict(env.normalize_obs(obs[None,:]),deterministic=True);obs,reward,terminated,truncated,info=e.step(action[0]);total+=reward;done=terminated or truncated
    if args.viewer:time.sleep(max(0,.05-(time.monotonic()-start)))
   row=dict(episode=k,seed=args.seed+k,reward=total,**info);results.append(row);print(json.dumps(row),flush=True)
 finally:env.close()
 summary=dict(episodes=len(results),success_rate=float(np.mean([x['is_success'] for x in results])),target=.8,randomize=args.randomize,results=results)
 output=args.output or run/('evaluation_random.json' if args.randomize else 'evaluation_fixed.json');output.write_text(json.dumps(summary,indent=2));print('Evaluation:',output,'success',summary['success_rate'])
if __name__=='__main__':main()
