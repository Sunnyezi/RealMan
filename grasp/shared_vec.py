"""One immutable MjModel, independent MjData, threaded physics, one RGB-D renderer."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,time
import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import GraspEnv,P

class StagedEnv(gym.Wrapper):
 def __init__(self,env):super().__init__(env);self.pending=None
 def step(self,action):
  if self.pending is None:raise RuntimeError('Physics stage missing')
  context=self.pending;self.pending=None
  return self.env._finish(context)

class SharedVecEnv(DummyVecEnv):
 def __init__(self,n,randomize=True,stream=False,threads=1,monitor_dir=None):
  if n<1 or threads<1:raise ValueError('Positive environment/thread counts required')
  self.model=mujoco.MjModel.from_xml_path(str(P/'scene.xml'))
  self.renderer=mujoco.Renderer(self.model,128,128)
  self.raw=[];self.staged=[];self.stream=Path(stream) if stream else None;self.last_publish=0.;self.closed=False;self.stage='training batch'
  self.executor=ThreadPoolExecutor(max_workers=min(n,threads),thread_name_prefix='mujoco-physics')
  factories=[]
  try:
   for i in range(n):
    env=GraspEnv(randomize=randomize,model=self.model,renderer=self.renderer)
    stage=StagedEnv(env);self.raw.append(env);self.staged.append(stage)
    monitor=Monitor(stage,filename=str(Path(monitor_dir)/f'worker_{i}') if monitor_dir else None)
    factories.append(lambda monitor=monitor:monitor)
   super().__init__(factories)
  except BaseException:
   self.executor.shutdown(wait=True);self.renderer.close();raise
 def reset(self):
  result=super().reset();self.publish(force=True);return result
 def step_wait(self):
  # _advance has no OpenGL/model mutation; each task owns all of its MjData.
  contexts=list(self.executor.map(lambda pair:pair[0]._advance(pair[1]),zip(self.raw,self.actions)))
  for stage,context in zip(self.staged,contexts):stage.pending=context
  # Finish RGB-D, observations, rewards and per-env autoresets on the GL owner thread.
  result=super().step_wait();self.publish();return result
 def publish(self,force=False):
  if self.stream is None or (not force and time.monotonic()-self.last_publish<.1):return
  self.stream.parent.mkdir(parents=True,exist_ok=True)
  temp=self.stream.with_name(f'.grid_{os.getpid()}.npz')
  np.savez(temp,qpos=np.stack([e.d.qpos for e in self.raw]),qvel=np.stack([e.d.qvel for e in self.raw]),
           ctrl=np.stack([e.d.ctrl for e in self.raw]),time=np.array([e.d.time for e in self.raw]),
           touch_force=np.stack([e.touch_force for e in self.raw]),env_ids=np.arange(self.num_envs),
           stage=self.stage,wall_time=time.time())
  os.replace(temp,self.stream);self.last_publish=time.monotonic()
 def close(self):
  if self.closed:return
  self.closed=True
  self.executor.shutdown(wait=True,cancel_futures=True)
  try:super().close()
  finally:self.renderer.close()
