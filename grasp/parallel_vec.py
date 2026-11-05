"""True process-parallel simulation, aggregated into one multi-environment viewer."""
from pathlib import Path
import os,time
import numpy as np,gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from domain import make_env

class GridSnapshot(gym.Wrapper):
 def grid_state(self):
  e=self.unwrapped
  return dict(qpos=e.d.qpos.copy(),qvel=e.d.qvel.copy(),ctrl=e.d.ctrl.copy(),time=float(e.d.time),touch_force=e.touch_force.copy())

class ParallelGridVec(SubprocVecEnv):
 def __init__(self,n,randomize=True,stream=False,profile=None,monitor_dir=None):
  if not 1<=n<=16:raise ValueError('Process backend is limited to 16 environments')
  self.stream=Path(stream) if stream else None;self.last_publish=0.;self.stage='parallel training batch'
  def factory(i):
   def create():
    e=GridSnapshot(make_env(profile,randomize=randomize,publish=False))
    return Monitor(e,filename=str(Path(monitor_dir)/f'worker_{i}') if monitor_dir else None)
   return create
  super().__init__([factory(i) for i in range(n)],start_method='spawn')
 def reset(self):
  result=super().reset();self.publish(force=True);return result
 def step_wait(self):
  result=super().step_wait();self.publish();return result
 def publish(self,force=False):
  if self.stream is None:return
  states=self.env_method('grid_state');self.stream.parent.mkdir(parents=True,exist_ok=True)
  temp=self.stream.with_name(f'.parallel_{os.getpid()}.npz')
  np.savez(temp,**{key:np.stack([s[key] for s in states]) for key in ('qpos','qvel','ctrl','time','touch_force')},
           env_ids=np.arange(self.num_envs),stage=self.stage,wall_time=time.time())
  os.replace(temp,self.stream);self.last_publish=time.monotonic()
