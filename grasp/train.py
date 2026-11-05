import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from compute import resolve_device,compute_info
from env import P
from stable_baselines3.common.utils import get_schedule_fn
from protocol import VALIDATION_SEEDS,episode,fingerprint
from domain import load_profile,make_env

class Metrics(BaseCallback):
 def __init__(self,path):super().__init__();self.path=path;self.last=0;self.started=None;self.transitions=0;self.episodes=0;self.successes=0;self.warnings=0
 def _on_step(self):
  if self.started is None:self.started=time.monotonic()
  infos=self.locals['infos']
  self.transitions+=len(infos)
  self.warnings+=sum(int(info['warnings']) for info in infos)
  completed=[info for info,done in zip(infos,self.locals['dones']) if done]
  self.episodes+=len(completed)
  self.successes+=sum(bool(info['is_success']) for info in completed)
  if self.num_timesteps-self.last>=256:
   self.last=self.num_timesteps
   elapsed=time.monotonic()-self.started
   rate=self.transitions/max(elapsed,1e-9)
   self.logger.record('floor/transitions_per_second',rate)
   self.logger.record('floor/physics_warnings',self.warnings)
   with self.path.open('a') as f:
    f.write(json.dumps(dict(steps=self.num_timesteps,transitions=self.transitions,
                            wall_seconds=elapsed,transitions_per_second=rate,
                            completed_episodes=self.episodes,successes=self.successes,
                            physics_warnings=self.warnings))+'\n')
  return True

class SuccessCheckpoint(BaseCallback):
 def __init__(self,run,normalized_env,randomize,profile):
  super().__init__();self.run=run;self.normalized_env=normalized_env;self.randomize=randomize;self.profile=profile;self.best=(-1.,-float('inf'));self.last=0
 def assess(self,stage):
  e=make_env(self.profile,randomize=self.randomize)
  try:
   def action(obs):
    return self.model.predict(self.normalized_env.normalize_obs(obs[None,:]),deterministic=True)[0][0]
   rows=[episode(e,action,seed) for seed in VALIDATION_SEEDS]
  finally:e.close()
  score=(float(np.mean([row['is_success'] for row in rows])),
         float(np.mean([row['reward'] for row in rows])))
  report=dict(timesteps=self.num_timesteps,stage=stage,success_rate=score[0],
              mean_reward=score[1],episodes=len(rows),randomize=self.randomize,results=rows)
  (self.run/f'validation_{stage}.json').write_text(json.dumps(report,indent=2))
  if stage!='imitation' and score>self.best:
   self.best=score;self.model.save(str(self.run/'best_policy'));self.normalized_env.save(str(self.run/'best_policy_vecnormalize.pkl'))
   (self.run/'best_policy_metrics.json').write_text(json.dumps(report,indent=2))
  print('Validation',stage,'success',score[0],flush=True)
 def _on_training_start(self):
  if self.model.num_timesteps==0 and (self.run/'warmstart.zip').exists():self.assess('imitation')
 def _on_step(self):
  if self.num_timesteps-self.last>=10000:
   self.last=self.num_timesteps;self.assess(f'ppo_{self.num_timesteps}')
  return True
 def _on_training_end(self):self.assess('ppo_final')

def main():
 p=argparse.ArgumentParser();p.add_argument('--backend',choices=['shared','subproc'],default='subproc');p.add_argument('--physics-threads',type=int,default=1);p.add_argument('--device',default='cuda:0',help='Policy device: cuda:N (default cuda:0) or cpu');p.add_argument('--learning-rate',type=float);p.add_argument('--action-std',type=float);p.add_argument('--no-live',action='store_true');p.add_argument('--warmstart',action='store_true');p.add_argument('--steps',type=int,default=300000);p.add_argument('--envs',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--randomize',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--run',type=Path);p.add_argument('--domain-profile',type=Path);args=p.parse_args()
 device=resolve_device(args.device);print('Compute',compute_info(device),flush=True)
 if args.warmstart and args.resume:raise SystemExit('Choose warmstart or resume, not both')
 if args.steps<=0 or args.envs<=0:raise SystemExit('steps/envs must be positive')
 if args.learning_rate is not None and args.learning_rate<=0:raise SystemExit('learning-rate must be positive')
 if args.action_std is not None and not 0<args.action_std<=1:raise SystemExit('action-std must lie in (0,1]')
 gate=P/'checks/gate.json'
 if not gate.exists():raise SystemExit('Training blocked: run check.py; scripted grasp must pass 10/10 first.')
 g=json.load(open(gate))
 if not g.get('passed') or g.get('fingerprint')!=fingerprint():raise SystemExit('Training blocked: missing/stale grasp gate. Rerun check.py.')
 profile=load_profile(args.domain_profile) if args.domain_profile else None
 if args.backend=='shared' and profile:raise SystemExit('Per-environment model randomization requires --backend subproc')
 if args.physics_threads<1:raise SystemExit('physics-threads must be positive')
 torch.set_num_threads(1)
 run=args.run or P/'runs'/time.strftime('%Y%m%d_%H%M%S');run=run.resolve();run.mkdir(parents=True,exist_ok=False)
 stream=False if args.no_live else run/'live_state.npz'
 if args.backend=='shared':
  from shared_vec import SharedVecEnv
  base=SharedVecEnv(args.envs,randomize=args.randomize,stream=stream,threads=args.physics_threads,monitor_dir=run)
 else:
  from parallel_vec import ParallelGridVec
  base=ParallelGridVec(args.envs,randomize=args.randomize,stream=stream,profile=profile,monitor_dir=run)
 base.seed(args.seed)
 if args.resume:
  meta=json.load(open(args.resume.parent/'run.json'))
  if meta['fingerprint']!=fingerprint():raise SystemExit('Resume blocked: environment fingerprint changed')
  if meta.get('domain_profile')!=json.loads(json.dumps(profile)):
   raise SystemExit('Resume blocked: domain profile changed')
  norm=args.resume.with_name(args.resume.stem+'_vecnormalize.pkl')
  if not norm.exists() and args.resume.stem.startswith('checkpoint_'):norm=args.resume.with_name(args.resume.stem.replace('checkpoint_','checkpoint_vecnormalize_',1)+'.pkl')
  if not norm.exists():norm=args.resume.parent/'vecnormalize.pkl'
  env=VecNormalize.load(str(norm),base)
  model=PPO.load(str(args.resume),env=env,device=device,tensorboard_log=str(run/'tensorboard'))
 else:
  env=VecNormalize(base,norm_obs=True,norm_reward=False,clip_obs=10.)
  model=PPO('MlpPolicy',env,learning_rate=(1e-6 if args.warmstart else 3e-4),n_steps=256,batch_size=256,n_epochs=(5 if args.warmstart else 10),gamma=.99,gae_lambda=.95,ent_coef=(0. if args.warmstart else .005),target_kl=.01,policy_kwargs=dict(net_arch=[128,128]),seed=args.seed,device=device,verbose=1,tensorboard_log=str(run/'tensorboard'))
 if args.warmstart:
  from warmstart import warmstart
  warmstart(model,env,run,args.seed,publish=stream,profile=profile)
  env.training=False
 if args.learning_rate is not None:model.learning_rate=args.learning_rate;model.lr_schedule=get_schedule_fn(args.learning_rate)
 if args.action_std is not None:
  with torch.no_grad():model.policy.log_std.fill_(float(np.log(args.action_std)))
 manifest=dict(backend=args.backend,physics_threads=args.physics_threads,compute=compute_info(model.device),domain_profile=profile,learning_rate=float(model.lr_schedule(1.)),action_std=float(torch.exp(model.policy.log_std).mean().detach()),warmstart=args.warmstart,seed=args.seed,randomize=args.randomize,envs=args.envs,steps=args.steps,fingerprint=fingerprint(),config=json.load(open(P/'config.json')),resume=str(args.resume) if args.resume else None)
 (run/'run.json').write_text(json.dumps(manifest,indent=2));(run/'config.json').write_text((P/'config.json').read_text())
 cb=CallbackList([Metrics(run/'metrics.jsonl'),SuccessCheckpoint(run,env,args.randomize,profile),CheckpointCallback(save_freq=max(1,10000//args.envs),save_path=str(run),name_prefix='checkpoint',save_vecnormalize=True)])
 base.seed(args.seed*1000)
 try:model.learn(total_timesteps=args.steps,callback=cb,reset_num_timesteps=not bool(args.resume))
 finally:
  model.save(str(run/'policy'));env.save(str(run/'vecnormalize.pkl'));env.close()
 print('Saved',run,flush=True)
if __name__=='__main__':main()
