from pathlib import Path
import json,subprocess,base64,hashlib
P=Path('/home/szy2025/Desktop/RealMan')
t=json.load(open(P/'sources/tree.json'))['tree'];prefix='RM65/urdf/RM65-6F/'
x=next(x for x in t if x['path']==prefix+'meshes/link_6.STL')
subprocess.run(['curl','-fsSL','--max-time','50','https://api.github.com/repos/RealManRobot/rm_models/git/blobs/'+x['sha'],'-o',str(P/'sources/link6_blob.json')],check=True)
b=base64.b64decode(json.load(open(P/'sources/link6_blob.json'))['content'])
assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha']
(P/'sources/arm_rm65_6f/meshes/link_6.verified').write_bytes(b)
print('verified',len(b))
