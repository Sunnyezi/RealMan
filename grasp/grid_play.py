"""Replay independent policy environments in the shared MuJoCo batch viewer."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,time
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from shared_vec import SharedVecEnv
from compute import resolve_device
from protocol import fingerprint

def main():
 p=argparse.ArgumentParser();p.add_argument('--backend',choices=['shared','subproc'],default='subproc');p.add_argument('checkpoint',type=Path);p.add_argument('--envs',type=int,default=16);p.add_argument('--physics-threads',type=int,default=1);p.add_argument('--stream',type=Path);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=20000);args=p.parse_args()
 if not 1<=args.envs<=16:raise ValueError('Use 1-16 environments')
 checkpoint=args.checkpoint.resolve();meta=json.loads((checkpoint.parent/'run.json').read_text())
 if meta['fingerprint']!=fingerprint():raise RuntimeError('Environment fingerprint differs')
 stream=args.stream or checkpoint.parent/'live_state.npz';torch.set_num_threads(1)
 if args.backend=='shared':base=SharedVecEnv(args.envs,stream=stream,threads=args.physics_threads)
 else:
  from parallel_vec import ParallelGridVec
  base=ParallelGridVec(args.envs,stream=stream)
 base.stage='POLICY PLAYBACK (no learning)';base.seed(args.seed)
 norm=checkpoint.with_name(checkpoint.stem+'_vecnormalize.pkl')
 if not norm.exists() and checkpoint.stem.startswith('checkpoint_'):
  norm=checkpoint.with_name(checkpoint.stem.replace('checkpoint_','checkpoint_vecnormalize_',1)+'.pkl')
 if not norm.exists():norm=checkpoint.parent/'vecnormalize.pkl'
 env=VecNormalize.load(str(norm),base);env.training=False;env.norm_reward=False
 try:
  model=PPO.load(str(checkpoint),device=resolve_device(args.device));obs=env.reset();episodes=0;successes=0
  while True:
   start=time.monotonic();obs,_,done,infos=env.step(model.predict(obs,deterministic=True)[0])
   for finished,info in zip(done,infos):
    if finished:episodes+=1;successes+=int(info['is_success'])
   if any(done):print(json.dumps(dict(episodes=episodes,successes=successes,success_rate=successes/episodes)),flush=True)
   time.sleep(max(0,.05-(time.monotonic()-start)))
 finally:env.close()
if __name__=='__main__':main()
