from pathlib import Path
import shutil,datetime,xml.etree.ElementTree as E,json,subprocess
P=Path('/home/szy2025/Desktop/RealMan');old=P/'backups/20261104_155613_rm65_6f'
backup=P/'backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_before_restore');backup.mkdir(parents=True)
for name in ['build_model.py','README.md','validation.json','motion_validation.json','replacement_validation.json']:
 if (P/name).exists():shutil.copy2(P/name,backup/name)
shutil.copytree(P/'models',backup/'models')
for name in ['build_model.py','README.md','validation.json','motion_validation.json']:
 if (old/name).exists():shutil.copy2(old/name,P/name)
shutil.copytree(old/'models',P/'models',dirs_exist_ok=True)
a=E.parse(backup/'models/scene.xml').getroot();b=E.parse(P/'models/scene.xml').getroot()
for tag in ['body','joint','geom','site','position']:
 left={x.get('name'):E.tostring(x) for x in a.iter(tag) if (x.get('name') or '').startswith('right_')}
 right={x.get('name'):E.tostring(x) for x in b.iter(tag) if (x.get('name') or '').startswith('right_')}
 assert left==right,tag
assert E.tostring(a.find('equality'))==E.tostring(b.find('equality'))
subprocess.run(['/home/szy2025/.conda/envs/realman-mujoco/bin/python',str(P/'check_motion.py')],stdout=open(P/'motion_validation.json','w'),check=True)
print('Restored RM65-6FB-V; hand unchanged; motion validation passed. Backup:',backup)
