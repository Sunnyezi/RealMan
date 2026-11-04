"""Exercise saved PPO CUDA optimizer state, resume, imitation gradients and CPU inference."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json
from unittest.mock import patch
import numpy as np,torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'grasp'))
from compute import resolve_device,compute_info
from env import GraspEnv

def main():
 checkpoint=Path(sys.argv[1]).resolve()
 device=resolve_device('cuda:0')
 with patch('torch.cuda.is_available',return_value=False):
  try:resolve_device('cuda:0')
  except RuntimeError:pass
  else:raise AssertionError('CUDA must not silently fall back')
 assert resolve_device('cpu').type=='cpu'
 vec=DummyVecEnv([lambda:GraspEnv(randomize=True)])
 env=VecNormalize.load(str(checkpoint.with_name('vecnormalize.pkl')),vec)
 try:
  model=PPO.load(str(checkpoint),env=env,device=device)
  assert next(model.policy.parameters()).is_cuda and model._n_updates>0
  assert any(v.is_cuda for state in model.policy.optimizer.state.values() for k,v in state.items() if isinstance(v,torch.Tensor) and k!='step')
  obs=env.reset();cpu=PPO.load(str(checkpoint),device='cpu')
  assert np.allclose(model.predict(obs,deterministic=True)[0],cpu.predict(obs,deterministic=True)[0],atol=1e-5)
  # Same minibatch path as supervised warmstart, with CUDA-native indexing.
  x=torch.zeros((64,120),device=device);y=torch.ones((64,9),device=device)*.1
  before=model.policy.action_net.weight.detach().clone()
  indices=torch.randint(64,(32,),device=device)
  optimizer=torch.optim.Adam(model.policy.parameters(),lr=3e-4)
  loss=(model.policy.get_distribution(x[indices]).distribution.mean-y[indices]).square().mean()
  optimizer.zero_grad();loss.backward();optimizer.step()
  assert not torch.equal(before,model.policy.action_net.weight)
  # Resume a real rollout and PPO update from the saved CUDA checkpoint.
  model.n_steps=32
  from stable_baselines3.common.buffers import RolloutBuffer
  model.rollout_buffer=RolloutBuffer(32,model.observation_space,model.action_space,device=device,gae_lambda=model.gae_lambda,gamma=model.gamma,n_envs=1)
  model.batch_size=32
  initial=model.num_timesteps
  model.learn(32,reset_num_timesteps=False)
  assert model.num_timesteps==initial+32
  report=dict(compute=compute_info(device),cuda_parameters=True,cuda_optimizer=True,
              cpu_inference_parity=True,fail_fast=True,warmstart_gradient_update=True,
              checkpoint_resume_steps=32,initial_timesteps=initial,final_timesteps=model.num_timesteps)
  (ROOT/'docs/cuda_training_check.json').write_text(json.dumps(report,indent=2))
  print(json.dumps(report,indent=2))
 finally:env.close()

if __name__=='__main__':main()
