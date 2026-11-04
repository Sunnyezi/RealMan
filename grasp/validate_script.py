import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
from env import GraspEnv,P
from scripted import action
import argparse,json,numpy as np,time
from itertools import count

def trial(seed,randomize=True,live=False):
 e=GraspEnv(randomize=randomize,publish=live);obs,_=e.reset(seed=seed);initial_est=e.vision.last.copy();initial_truth=e.d.xpos[e.ob].copy();peak=np.zeros(17);minimum_margin=e.path_plan['minimum_margin'];goals=[e.grasp_goal.copy(),e.lift_goal.copy()]
 for _ in range(e.cfg['horizon']):
  start=time.monotonic();obs,reward,t,tr,info=e.step(action(e));peak=np.maximum(peak,e.touch_force)
  if live:time.sleep(max(0,.05-(time.monotonic()-start)))
  if t or tr:break
 row=dict(seed=seed,estimate=initial_est.tolist(),truth=initial_truth.tolist(),vision_error=float(np.linalg.norm(initial_est-initial_truth)),grasp_goal=goals[0].tolist(),lift_goal=goals[1].tolist(),minimum_margin=minimum_margin,touch_peak_N=peak.tolist(),**info);e.close();return row
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--live-loop',action='store_true',help='Loop scripted grasps and stream frames to the viewer')
 args=parser.parse_args()
 if args.live_loop:
  for seed in count(10000):
   row=trial(seed,live=True)
   print(json.dumps(row),flush=True)
 rows=[]
 for seed in range(10):
  try:r=trial(seed);rows.append(r);print(json.dumps(r),flush=True)
  except Exception as ex:print('FAILED',seed,str(ex),flush=True);raise
 (P/'random_script_check.json').write_text(json.dumps(rows,indent=2))
