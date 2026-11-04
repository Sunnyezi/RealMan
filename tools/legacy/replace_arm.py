from pathlib import Path
import shutil,hashlib,json,datetime,subprocess,xml.etree.ElementTree as E
P=Path('/home/szy2025/Desktop/RealMan')
backup=P/'backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_rm65_6f');backup.mkdir(parents=True)
for name in ['build_model.py','README.md','validation.json','motion_validation.json']:
 if (P/name).exists():shutil.copy2(P/name,backup/name)
shutil.copytree(P/'models',backup/'models')
def hashes():
 return {str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (P/'sources/hand').rglob('*') if x.is_file()}
before=hashes()
s=(P/'build_model.py').read_text();s=s.replace("sources/arm/urdf/RM65-6FB-V.urdf","sources/arm_rm65_6f/urdf/RM65-6F.urdf").replace("a,P/'sources/arm'","a,P/'sources/arm_rm65_6f'").replace('RM65 V + RH56DFTP right','RM65-6F + RH56DFTP right')
(P/'build_model.py').write_text(s)
subprocess.run(['/home/szy2025/.conda/envs/realman-mujoco/bin/python',str(P/'build_model.py')],check=True)
assert hashes()==before,'Hand source changed'
old=E.parse(backup/'models/scene.xml').getroot();new=E.parse(P/'models/scene.xml').getroot()
for tag in ['body','joint','geom','site','position']:
 a={x.get('name'):E.tostring(x) for x in old.iter(tag) if (x.get('name') or '').startswith('right_')}
 b={x.get('name'):E.tostring(x) for x in new.iter(tag) if (x.get('name') or '').startswith('right_')}
 assert a==b,('Hand compiled elements changed',tag)
assert E.tostring(old.find('equality'))==E.tostring(new.find('equality'))
json.dump({'hand_sources_unchanged':True,'hand_compiled_elements_unchanged':True,'backup':str(backup),'arm_urdf':'sources/arm_rm65_6f/urdf/RM65-6F.urdf'},open(P/'replacement_validation.json','w'),indent=2)
s=(P/'README.md').read_text();s=s.replace('RM65 V +','RM65-6F +').replace('RM65/urdf/RM65-6FB-V','RM65/urdf/RM65-6F').replace('  Requested RM65-6F-V exact model was not found; 6FB-V is explicitly a provisional substitute.','  Arm replaced with the user-selected official RM65-6F URDF.').replace('6 arm, 1 camera rotation, and 6 hand','6 arm and 6 hand')
(P/'README.md').write_text(s)
print('Hand sources, compiled subtree, actuators and mimic constraints unchanged. Backup:',backup)
