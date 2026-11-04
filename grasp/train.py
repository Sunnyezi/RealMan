import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
from env import GraspEnv,P
from stable_baselines3.common.utils import get_schedule_fn

def fingerprint():
 h=hashlib.sha256()
 for name in ['scene.xml','config.json','env.py','vision.py','check.py','pinch_probe.py']:
  h.update(name.encode());h.update((P/name).read_bytes())
 return h.hexdigest()
class Metrics(BaseCallback):
 def __init__(self,path):super().__init__();self.path=path;self.last=0
 def _on_step(self):
  if self.num_timesteps-self.last>=256:
   self.last=self.num_timesteps
   for key in ['height','max_height','hand_contacts','vision_lost_rate','ik_failures','warnings','is_success','dropped']:
    self.logger.record('grasp/'+key,np.mean([i.get(key,0) for i in self.locals['infos']]))
   with self.path.open('a') as f:f.write(json.dumps(dict(steps=self.num_timesteps,time=time.time(),infos=self.locals['infos']),default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
  return True

class SuccessCheckpoint(BaseCallback):
 def __init__(self,run,normalized_env,randomize):
  super().__init__();self.run=run;self.normalized_env=normalized_env;self.randomize=randomize;self.best=(-1.,-float('inf'));self.last=0
 def assess(self):
  e=GraspEnv(randomize=self.randomize);scores=[];returns=[]
  try:
   for seed in [71001,71002]:
    obs,_=e.reset(seed=seed);total=0.
    for _ in range(200):
     a,_=self.model.predict(self.normalized_env.normalize_obs(obs[None,:]),deterministic=True);obs,r,t,tr,info=e.step(a[0]);total+=r
     if t or tr:break
    scores.append(float(info['is_success']));returns.append(total)
  finally:e.close()
  score=(float(np.mean(scores)),float(np.mean(returns)))
  if score>self.best:
   self.best=score;self.model.save(str(self.run/'best_policy'));self.normalized_env.save(str(self.run/'best_policy_vecnormalize.pkl'))
   (self.run/'best_policy_metrics.json').write_text(json.dumps(dict(timesteps=self.num_timesteps,success_rate=score[0],mean_reward=score[1],episodes=2,seeds=[71001,71002],randomize=self.randomize),indent=2))
  print('Validation',self.num_timesteps,'success',score[0],flush=True)
 def _on_training_start(self):self.assess()
 def _on_step(self):
  if self.num_timesteps-self.last>=10000:self.last=self.num_timesteps;self.assess()
  return True
 def _on_training_end(self):self.assess()

def main():
 p=argparse.ArgumentParser();p.add_argument('--learning-rate',type=float);p.add_argument('--action-std',type=float);p.add_argument('--no-live',action='store_true');p.add_argument('--warmstart',action='store_true');p.add_argument('--steps',type=int,default=300000);p.add_argument('--envs',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--randomize',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--run',type=Path);args=p.parse_args()
 if args.steps<=0 or args.envs<=0:raise SystemExit('steps/envs must be positive')
 if args.learning_rate is not None and args.learning_rate<=0:raise SystemExit('learning-rate must be positive')
 if args.action_std is not None and not 0<args.action_std<=1:raise SystemExit('action-std must lie in (0,1]')
 gate=P/'checks/gate.json'
 if not gate.exists():raise SystemExit('Training blocked: run check.py; scripted grasp must pass 10/10 first.')
 g=json.load(open(gate))
 if not g.get('passed') or g.get('fingerprint')!=fingerprint():raise SystemExit('Training blocked: missing/stale grasp gate. Rerun check.py.')
 torch.set_num_threads(1)
 run=args.run or P/'runs'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True,exist_ok=True)
 factories=[lambda i=i:Monitor(GraspEnv(randomize=args.randomize,publish=(i==0 and not args.no_live)),filename=str(run/f'worker_{i}')) for i in range(args.envs)]
 base=SubprocVecEnv(factories,start_method='spawn');base.seed(args.seed)
 if args.resume:
  meta=json.load(open(args.resume.parent/'run.json'))
  if meta['fingerprint']!=fingerprint():raise SystemExit('Resume blocked: environment fingerprint changed')
  norm=args.resume.with_name(args.resume.stem+'_vecnormalize.pkl')
  if not norm.exists() and args.resume.stem.startswith('checkpoint_'):norm=args.resume.with_name(args.resume.stem.replace('checkpoint_','checkpoint_vecnormalize_',1)+'.pkl')
  if not norm.exists():norm=args.resume.parent/'vecnormalize.pkl'
  env=VecNormalize.load(str(norm),base)
  model=PPO.load(str(args.resume),env=env,device='cpu',tensorboard_log=str(run/'tensorboard'))
 else:
  env=VecNormalize(base,norm_obs=True,norm_reward=False,clip_obs=10.)
  model=PPO('MlpPolicy',env,learning_rate=(1e-6 if args.warmstart else 3e-4),n_steps=256,batch_size=256,n_epochs=(5 if args.warmstart else 10),gamma=.99,gae_lambda=.95,ent_coef=(0. if args.warmstart else .005),target_kl=.01,policy_kwargs=dict(net_arch=[128,128]),seed=args.seed,device='cpu',verbose=1,tensorboard_log=str(run/'tensorboard'))
 if args.warmstart and args.resume:raise SystemExit('Choose warmstart or resume, not both')
 if args.warmstart:
  from warmstart import warmstart
  warmstart(model,env,run)
  env.training=False
 if args.learning_rate is not None:model.learning_rate=args.learning_rate;model.lr_schedule=get_schedule_fn(args.learning_rate)
 if args.action_std is not None:
  with torch.no_grad():model.policy.log_std.fill_(float(np.log(args.action_std)))
 manifest=dict(learning_rate=float(model.lr_schedule(1.)),action_std=float(torch.exp(model.policy.log_std).mean().detach()),warmstart=args.warmstart,seed=args.seed,randomize=args.randomize,envs=args.envs,steps=args.steps,fingerprint=fingerprint(),config=json.load(open(P/'config.json')),resume=str(args.resume) if args.resume else None)
 (run/'run.json').write_text(json.dumps(manifest,indent=2));(run/'config.json').write_text((P/'config.json').read_text())
 cb=CallbackList([Metrics(run/'metrics.jsonl'),SuccessCheckpoint(run,env,args.randomize),CheckpointCallback(save_freq=max(1,10000//args.envs),save_path=str(run),name_prefix='checkpoint',save_vecnormalize=True)])
 try:model.learn(total_timesteps=args.steps,callback=cb,reset_num_timesteps=not bool(args.resume))
 finally:
  model.save(str(run/'policy'));env.save(str(run/'vecnormalize.pkl'));env.close()
 print('Saved',run,flush=True)
if __name__=='__main__':main()
