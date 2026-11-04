from env import GraspEnv,P
from build_scene import R
import numpy as np,json,time

def action(env):
 cfg=env.cfg;grasp=np.array(cfg['grasp_xyz']);up=grasp+[0,0,.115]
 target=np.array(cfg['hand_target']);target[1:3]+=.12
 if env.hand_target[2]<.01 and np.linalg.norm(env.target_xyz-grasp)>.003:
  xyz=grasp;ht=env.hand_target.copy()
 elif env.hand_target[2]<target[2]-.001 or env.hand_target[1]<target[1]-.001:
  xyz=grasp;ht=target
 else:xyz=up;ht=target
 delta=np.clip(xyz-env.target_xyz,-.0025,.0025)
 dh=np.clip(ht-env.hand_target,-.02,.02)
 return np.clip(np.r_[delta/.005,dh/.06],-1,1).astype(np.float32)
if __name__=='__main__':
 e=GraspEnv(publish=True);obs,_=e.reset(seed=42);rows=[]
 for _ in range(200):
  start=time.monotonic();obs,reward,term,trunc,info=e.step(action(e))
  if e.steps%20==0:print(e.steps,info,flush=True)
  if term or trunc:break
  # Let the visible viewer follow the same run.
  time.sleep(max(0,.05-(time.monotonic()-start)))
 print('result',info,flush=True);(P/'scripted_env_result.json').write_text(json.dumps(info,indent=2));e.close()
