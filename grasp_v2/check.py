import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,time
import numpy as np,mujoco
from stable_baselines3.common.env_checker import check_env
from env import GraspEnv,P
from probe_sensors import probe_regions
from validate_script import trial
from train import fingerprint

def main():
 p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=20);p.add_argument('--live',action='store_true');args=p.parse_args();out=P/'checks';out.mkdir(exist_ok=True);results={}
 e=GraspEnv(randomize=True)
 try:
  check_env(e,warn=True);results['gymnasium_check']=True
  a,_=e.reset(seed=42);b,_=e.reset(seed=42);assert np.allclose(a,b);assert a.shape==(120,);results['observation_dim']=120
  assert e.m.nsensor==17 and len(e.sensor_adrs)==17;assert np.max(e.touch_force)==0;results['no_contact_zero_force']=True
  mujoco.mj_forward(e.m,e.d);before=e._obs();e.d.qpos[e.oa]+=.03;mujoco.mj_forward(e.m,e.d);assert np.array_equal(before,e._obs());results['no_object_truth_leak']=True
  e.reset(seed=1);g1=e.grasp_goal.copy();est1=e.vision.last.copy();e.reset(seed=2);assert np.allclose(e.grasp_goal-g1,e.vision.last-est1);results['vision_drives_goal']=True
  margins=[]
  for dx in [-.05,.05]:
   for dy in [-.05,.05]:
    plan=e.planner.plan(np.array(e.cfg['object_xyz'])+[dx,dy,-.001],e.cfg['home_xyz'],e.cfg['home_q']);assert len(plan['waypoints'])==15;margins.append(plan['minimum_margin'])
  results['corner_trajectories_valid']=True;results['corner_minimum_margin_m']=min(margins)
  e.reset(seed=3);last=e.vision.last.copy();original=e.m.geom_rgba[e.og].copy();e.m.geom_rgba[e.og]=[.5,.5,.5,1];e.d.time+=.1;est,visible,age=e.vision.update(e.d);assert not visible and age>=.099 and np.array_equal(est,last);e.m.geom_rgba[e.og]=original;results['occlusion_stale']=True
  e.reset(seed=4)
  for _ in range(40):e.step(np.r_[0,0,1,np.zeros(6)])
  assert e.target_xyz[2]<=e.lift_goal[2]+1e-8;assert e.action_projections>0;results['upward_target_capped']=True
 finally:e.close()
 sensors=probe_regions();assert all(x['responded'] for x in sensors);results['sensor_regions_responding']=sum(x['responded'] for x in sensors);(out/'sensor_probe.json').write_text(json.dumps(sensors,indent=2))
 trials=[]
 for seed in range(args.trials):
  row=trial(seed,live=args.live);trials.append(row);print('Trial',seed,row['is_success'],'lift',round(row['max_height'],4),'IK rejected',row['ik_rejections'],flush=True)
 passed=args.trials>=10 and all(x['is_success'] and x['ik_failures']==0 and x['ik_rejections']==0 and not x['warnings'] for x in trials)
 report=dict(passed=passed,fingerprint=fingerprint(),time=time.time(),checks=results,script_trials=trials);(out/'gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=passed,checks=results)),flush=True)
 if not passed:raise SystemExit('Random visual-script validation failed; policy training blocked')
if __name__=='__main__':main()
