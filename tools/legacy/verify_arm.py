from pathlib import Path
import hashlib,json
P=Path('/home/szy2025/Desktop/RealMan');prefix='RM65/urdf/RM65-6F/'
for x in json.load(open(P/'sources/tree.json'))['tree']:
 if x['type']!='blob' or not x['path'].startswith(prefix):continue
 f=P/'sources/arm_rm65_6f'/x['path'][len(prefix):]
 if not f.is_file():continue
 b=f.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha'],str(f)
 print('Verified official blob:',f.name)
