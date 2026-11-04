import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,json,time
import numpy as np,mujoco
from stable_baselines3.common.env_checker import check_env
from env import GraspEnv,P
from pinch_probe import run
from train import fingerprint

def main():
 p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=10);p.add_argument('--live',action='store_true');args=p.parse_args()
 out=P/'checks';out.mkdir(exist_ok=True)
 env=GraspEnv();results={}
 try:
  check_env(env,warn=True);results['gymnasium_check']=True
  obs,_=env.reset(seed=42);seedpos=env.d.qpos.copy();obs2,_=env.reset(seed=42);assert np.allclose(obs,obs2) and np.allclose(seedpos,env.d.qpos);results['deterministic_reset']=True
  mujoco.mj_forward(env.m,env.d);clean=env._obs();env.d.qpos[env.oa]+=.04;mujoco.mj_forward(env.m,env.d);assert np.array_equal(clean,env._obs());results['no_object_state_leak']=True
  obs,_=env.reset(seed=42);truth=env.d.xpos[env.ob].copy();estimate,visible,age=env.vision.update(env.d);assert visible
  error=float(np.linalg.norm(estimate-truth));assert error<.01,error;results['initial_vision_error_m']=error
  last=env.vision.last.copy();t=env.vision.last_seen;old=env.m.geom_rgba[env.og].copy();env.m.geom_rgba[env.og]=[.5,.5,.5,1];env.d.time+=.1
  estimate,visible,age=env.vision.update(env.d);assert not visible and age>=.099 and np.array_equal(last,estimate) and env.vision.last_seen==t;env.m.geom_rgba[env.og]=old;results['occlusion_stale_flag']=True
  env.reset(seed=43);env.target_xyz=np.array([.65,.2,.6]);prev=env.q_target.copy();env.step(np.ones(9));assert env.ik_failures>0 and np.array_equal(prev,env.q_target);results['ik_failure_holds_target']=True
  env.reset(seed=44);z0=env.d.xpos[env.ob,2]
  for _ in range(1000):mujoco.mj_step(env.m,env.d)
  assert abs(env.d.xpos[env.ob,2]-z0)<.002 and not any(w.number for w in env.d.warning);results['table_settle_stable']=True
 finally:env.close()
 trials=[]
 for k in range(args.trials):
  row=run(squeeze=.12,publish=args.live,realtime=args.live);trials.append(row);print(f'Trial {k+1}: success={row["success"]} max_height={row["max_height"]:.4f} hold={row["hold"]:.3f}',flush=True)
 passed=args.trials>=10 and all(r['success'] and not any(r['warnings']) for r in trials)
 report=dict(passed=passed,fingerprint=fingerprint(),time=time.time(),checks=results,script_trials=trials);(out/'gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=passed,checks=results)),flush=True)
 if not passed:raise SystemExit('Grasp gate did not pass; PPO must remain blocked.')
if __name__=='__main__':main()
