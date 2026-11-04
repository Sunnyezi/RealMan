from pathlib import Path
import subprocess,concurrent.futures,json
P=Path(__file__).parent
starts=[30000,30025,30050,30075]
def run(start):
 with (P/f'checks/ik_part_{start}.log').open('w') as f:subprocess.run(['/home/szy2025/.conda/envs/realman-mujoco/bin/python',str(P/'replay_ik_part.py'),str(start),str(start+25)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Completed',start,start+24,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,starts))
rows=[];episodes=[]
for start in starts:
 rows+=json.load(open(P/f'checks/ik_failed_requests_{start}.json'));episodes+=json.load(open(P/f'checks/ik_replay_summary_{start}.json'))['episodes']
summary=dict(episodes=len(episodes),episodes_with_failures=sum(x['failures']>0 for x in episodes),failed_requests=len(rows),geometrically_proven_requests=sum(x['proven_outside_tolerance'] for x in rows),episodes_all_failed_requests_proven=sum(x['failures']>0 and x['proven']==x['failures'] for x in episodes))
(P/'checks/ik_failed_requests.json').write_text(json.dumps(rows,indent=2));(P/'checks/ik_replay_summary.json').write_text(json.dumps(dict(summary=summary,episodes=episodes),indent=2));print(json.dumps(summary),flush=True)
