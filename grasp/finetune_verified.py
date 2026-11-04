import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from env import GraspEnv,P
from train import fingerprint,Metrics

def main():
 torch.set_num_threads(1);source=P/'runs/grasp_v5';run=P/'runs/grasp_verified';run.mkdir(parents=True,exist_ok=True)
 assert json.load(open(P/'checks/gate.json'))['fingerprint']==fingerprint()
 assert json.load(open(source/'run.json'))['config']==json.load(open(P/'config.json'))
 base=SubprocVecEnv([lambda i=i:Monitor(GraspEnv(publish=(i==0)),filename=str(run/f'worker_{i}')) for i in range(4)],start_method='spawn');base.seed(42)
 env=VecNormalize.load(str(source/'warmstart_vecnormalize.pkl'),base);env.training=False;env.norm_reward=False
 model=PPO.load(str(source/'warmstart.zip'),env=env,device='cpu',tensorboard_log=str(run/'tensorboard'));model.learning_rate=1e-6;model.lr_schedule=get_schedule_fn(1e-6);model.clip_range=get_schedule_fn(.05)
 manifest=dict(seed=42,randomize=False,envs=4,steps=1024,fingerprint=fingerprint(),config=json.load(open(P/'config.json')),initialization=str(source/'warmstart.zip'),migration='Reject camera housing in color segmentation; unchanged physics, 79-dimensional observation and 9-dimensional action. Fine-tune and evaluate using corrected pipeline.',source_fingerprint=json.load(open(source/'run.json'))['fingerprint'],ppo_learning_rate=1e-6)
 (run/'run.json').write_text(json.dumps(manifest,indent=2))
 try:model.learn(1024,callback=Metrics(run/'metrics.jsonl'),reset_num_timesteps=True)
 finally:model.save(str(run/'policy'));env.save(str(run/'vecnormalize.pkl'));env.close()
 print('Saved',run,flush=True)
if __name__=='__main__':main()
