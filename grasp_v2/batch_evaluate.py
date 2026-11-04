from pathlib import Path
import subprocess,concurrent.futures,json,argparse,os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
P=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--seed',type=int,default=10000);a=p.parse_args();cp=Path(a.checkpoint).resolve();run=cp.parent
starts=[a.seed+i*25 for i in range(4)]
def test(seed):
 out=run/f'eval_part_{seed}.json'
 with (run/f'eval_part_{seed}.log').open('w') as f:
  subprocess.run(['/home/szy2025/.conda/envs/realman-mujoco/bin/python',str(P/'evaluate.py'),str(cp),'--randomize','--episodes','25','--seed',str(seed),'--output',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Finished seeds',seed,seed+24,flush=True)
 return json.load(open(out))['results']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:parts=list(ex.map(test,starts))
r=[x for part in parts for x in part];out=dict(episodes=100,success_rate=sum(x['is_success'] for x in r)/100,target=.8,randomize=True,results=r);(run/'evaluation_random.json').write_text(json.dumps(out,indent=2));print('Random success',out['success_rate'],flush=True)
