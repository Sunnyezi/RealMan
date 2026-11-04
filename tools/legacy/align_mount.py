from pathlib import Path
import shutil,datetime,subprocess,xml.etree.ElementTree as E,json,hashlib
import trimesh,numpy as np
P=Path('/home/szy2025/Desktop/RealMan');B=P/'backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_before_flush_mount');B.mkdir(parents=True)
for f in ['build_model.py','README.md','validation.json','motion_validation.json']:shutil.copy2(P/f,B/f)
shutil.copytree(P/'models',B/'models')
hand_hash={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'sources/hand').rglob('*') if f.is_file()}
s=(P/'build_model.py').read_text();assert "xyz='0 0 0.02'" in s
s=s.replace("xyz='0 0 0.02'","xyz='0 0 0'").replace('RM65 V + RH56DFTP right (provisional mount)','RM65-6FB-V + RH56DFTP right (flush mount)')
(P/'build_model.py').write_text(s)
subprocess.run(['/home/szy2025/.conda/envs/realman-mujoco/bin/python',str(P/'build_model.py')],check=True)
a=E.parse(B/'models/scene.xml').getroot();b=E.parse(P/'models/scene.xml').getroot()
pa=a.find('.//body[@name="right_base_link"]');pb=b.find('.//body[@name="right_base_link"]')
print('Palm position before/after:',pa.get('pos'),pb.get('pos'))
pa.attrib.pop('pos',None);pb.attrib.pop('pos',None);assert E.tostring(pa)==E.tostring(pb),'Hand geometry or physics changed'
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hand_hash.items())
zarm=trimesh.load(P/'sources/arm/meshes/link_6.STL').bounds[1,2];zhand=trimesh.load(P/'sources/hand/meshes/right_base_link.STL').bounds[0,2]
assert abs(zarm-zhand)<1e-6
subprocess.run(['/home/szy2025/.conda/envs/realman-mujoco/bin/python',str(P/'check_motion.py')],stdout=open(P/'motion_validation.json','w'),check=True)
s=(P/'README.md').read_text().replace('provisional 20 mm Z offset and zero rotation','zero translation and zero rotation (mounting faces aligned)').replace('Check actual adapter and palm orientation against hardware drawings before real use.','Mesh mounting faces coincide at local Z=0; no added adapter geometry.')
(P/'README.md').write_text(s)
json.dump(dict(mount_translation=[0,0,0],mount_rpy=[0,0,0],mount_face_gap_m=float(zhand-zarm),hand_unchanged_except_mount=True,backup=str(B)),open(P/'mount_validation.json','w'),indent=2)
print('Mount faces aligned; gap in meters:',zhand-zarm)
