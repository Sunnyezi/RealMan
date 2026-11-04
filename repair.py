import pathlib,struct,subprocess,concurrent.futures
P=pathlib.Path('/home/szy2025/Desktop/RealMan/sources')
jobs=[]
for part,repo,prefix in [('arm','RealManRobot/rm_models','RM65/urdf/RM65-6FB-V/meshes/'),('hand','AlexArutiunian/tactile-rh56dftp','urdf/inspire_hand/meshes/')]:
 for p in (P/part/'meshes').glob('*.STL'):
  data=p.read_bytes(); expected=84+50*struct.unpack('<I',data[80:84])[0] if len(data)>=84 else 0
  if len(data)!=expected: jobs.append((p,repo,prefix,expected)); print('repair',p.name,len(data),expected,flush=True)
def fix(j):
 p,repo,prefix,expected=j
 subprocess.run(['curl','-fLsS','--max-time','480','-C','-','https://raw.githubusercontent.com/'+repo+'/main/'+prefix+p.name,'-o',str(p)],check=True)
 assert p.stat().st_size==expected,(p,p.stat().st_size,expected)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex: list(ex.map(fix,jobs))
