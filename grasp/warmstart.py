from pathlib import Path
import numpy as np,torch,json,copy
from env import GraspEnv,P
from scripted import action

def warmstart(model,vecenv,run,episodes=8):
 optimizer=torch.optim.Adam(model.policy.parameters(),lr=3e-4,eps=1e-5)
 e=GraspEnv(randomize=True,publish=True);xs=[];ys=[];reports=[];best=None;best_score=-1;rng=np.random.default_rng(77)
 try:
  for round in range(4):
   successes=0
   for k in range(episodes):
    obs,_=e.reset(seed=500+round*episodes+k)
    for _ in range(e.cfg['horizon']):
     expert=action(e);xs.append(obs.copy());ys.append(expert.copy())
     if round==0 or (round==1 and rng.random()<.5):a=expert
     else:a=model.predict(vecenv.normalize_obs(obs[None,:]),deterministic=True)[0][0]
     obs,_,term,trunc,info=e.step(a)
     if term or trunc:break
    successes+=bool(info['is_success'])
   if round>=2 and successes>best_score:
    best_score=successes;best=copy.deepcopy(model.policy.state_dict());print('Retained actor with',successes,'/',episodes,'successful policy rollouts',flush=True)
   x=np.array(xs);y=np.array(ys)
   if round==0:
    vecenv.obs_rms.update(x);vecenv.obs_rms.var=np.maximum(vecenv.obs_rms.var,.0025)
   xt=torch.as_tensor(vecenv.normalize_obs(x),device=model.device);yt=torch.as_tensor(y,device=model.device)
   for i in range(4500):
    ids=rng.integers(0,len(x),256);mean=model.policy.get_distribution(xt[ids]).distribution.mean;loss=((mean-yt[ids])**2).mean()
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);optimizer.step()
   report=dict(round=round,successes=successes,episodes=episodes,samples=len(x),mse=float(loss.detach()));reports.append(report);print('Warmstart:',report,flush=True)
 finally:e.close()
 if best is not None:model.policy.load_state_dict(best)
 with torch.no_grad():model.policy.log_std.fill_(-4.)
 (run/'warmstart.json').write_text(json.dumps(reports,indent=2));model.save(str(run/'warmstart'));vecenv.save(str(run/'warmstart_vecnormalize.pkl'))
