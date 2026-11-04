import json,urllib.request,concurrent.futures,pathlib,subprocess
base=pathlib.Path('/home/szy2025/Desktop/RealMan/sources')
jobs=[]
for tree,repo,prefix,out in [('tree.json','RealManRobot/rm_models','RM65/urdf/RM65-6FB-V/','arm'),('ftp_tree.json','AlexArutiunian/tactile-rh56dftp','urdf/inspire_hand/','hand')]:
 for x in json.load(open(base/tree)).get('tree',[]):
  p=x['path']
  if x['type']=='blob' and p.startswith(prefix) and (p.endswith('.urdf') or '/meshes/' in p and ('right_' in p or out=='arm')):
   jobs.append((repo,p,base/out/p[len(prefix):]))
def get(j):
 repo,p,d=j; d.parent.mkdir(parents=True,exist_ok=True)
 for i in range(3):
  try: subprocess.run(['curl','-fsSL','--max-time','90','https://raw.githubusercontent.com/'+repo+'/main/'+p,'-o',str(d)],check=True); return str(d)
  except Exception:
   if i==2: raise
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
 for p in ex.map(get,jobs): print(p,flush=True)

