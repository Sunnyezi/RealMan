from pathlib import Path
import json,subprocess,concurrent.futures,struct
P=Path('/home/szy2025/Desktop/RealMan'); prefix='RM65/urdf/RM65-6F/'
tree=json.load(open(P/'sources/tree.json'))['tree']
jobs=[x['path'] for x in tree if x['type']=='blob' and x['path'].startswith(prefix) and (x['path'].endswith('.urdf') or '/meshes/' in x['path'])]
def get(p):
 d=P/'sources/arm_rm65_6f'/p[len(prefix):];d.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['curl','-fLsS','--retry','2','--max-time','480','https://raw.githubusercontent.com/RealManRobot/rm_models/main/'+p,'-o',str(d)],check=True)
 if d.suffix.lower()=='.stl':
  b=d.read_bytes();assert len(b)==84+50*struct.unpack('<I',b[80:84])[0],str(d)
 print(d,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(get,jobs))
