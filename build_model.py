from pathlib import Path
import xml.etree.ElementTree as E
import shutil,json
import mujoco,numpy as np
P=Path(__file__).resolve().parent
LIMITS=json.loads((P/'physical_limits.json').read_text())
(P/'models/meshes').mkdir(parents=True,exist_ok=True)
a=E.parse(P/'sources/arm/urdf/RM65-6FB-V.urdf').getroot()
h=E.parse(P/'sources/hand/FTP_right_hand.urdf').getroot()
for root,src,prefix in [(a,P/'sources/arm','arm_'),(h,P/'sources/hand','hand_')]:
 for x in list(root):
  if x.tag not in ('link','joint'): root.remove(x)
 for mesh in root.iter('mesh'):
  name=Path(mesh.get('filename')).name
  dest=prefix+name
  shutil.copy2(src/'meshes'/name,P/'models/meshes'/dest)
  mesh.set('filename',dest)
# Mount the palm frame directly; source wrist stub belongs to another assembly.
for x in list(h):
 if x.get('name') in ('right_wrist_yaw_link','right_base_joint'): h.remove(x)
for x in h: a.append(x)
j=E.SubElement(a,'joint',name='hand_mount',type='fixed')
E.SubElement(j,'parent',link='link_6'); E.SubElement(j,'child',link='right_base_link')
E.SubElement(j,'origin',xyz='0 0 0',rpy='0 0 0')
c=E.SubElement(a,'mujoco'); E.SubElement(c,'compiler',meshdir='meshes',discardvisual='false',fusestatic='false',balanceinertia='true')
E.ElementTree(a).write(P/'models/combined.urdf')
m=mujoco.MjModel.from_xml_path(str(P/'models/combined.urdf'))
mujoco.mj_saveLastXML(str(P/'models/robot.xml'),m)
r=E.parse(P/'models/robot.xml').getroot(); r.set('model','RM65-6FB-V + RH56DFTP right (flush mount)')
r.find('compiler').set('meshdir','meshes')
E.SubElement(r,'option',timestep='0.002',integrator='implicitfast')
default=E.SubElement(r,'default'); E.SubElement(default,'joint',damping='1',armature='0.01')
w=r.find('worldbody')
E.SubElement(w,'geom',name='floor',type='plane',size='2 2 0.1',rgba='0.22 0.25 0.3 1')
E.SubElement(w,'light',pos='0 -2 3',dir='0 0 -1',diffuse='0.8 0.8 0.8')
# Adjacent CAD shells overlap; disable robot self collision for this initial scene.
for g in w.iter('geom'):
 if g.get('name')!='floor': g.set('contype','0'); g.set('conaffinity','0')
eq=E.SubElement(r,'equality'); mimic={}
for joint in a.findall('joint'):
 mm=joint.find('mimic')
 if mm is not None:
  mimic[joint.get('name')]=mm.get('joint')
  E.SubElement(eq,'joint',joint1=joint.get('name'),joint2=mm.get('joint'),polycoef=f"{mm.get('offset','0')} {mm.get('multiplier','1')} 0 0 0",solref='0.01 1')
act=E.SubElement(r,'actuator'); names=[]
for joint in a.findall('joint'):
 name=joint.get('name')
 if joint.get('type') not in ('revolute','continuous','prismatic') or name in mimic: continue
 lim=joint.find('limit'); lo=lim.get('lower','-3.14'); hi=lim.get('upper','3.14'); hand=name.startswith('right_')
 torque = LIMITS['hand']['simulation_torque_cap_Nm'] if hand else LIMITS['arm']['joint_torque_limit_Nm'].get(name,80.)
 # Camera servo remains a synthetic holding actuator, not a calibrated motor.
 E.SubElement(act,'position',name=name+'_position',joint=name,kp='15' if hand else '250',kv='1' if hand else '25',ctrlrange=lo+' '+hi,forcerange=f'{-torque:g} {torque:g}')
 names.append(name)
E.indent(r); E.ElementTree(r).write(P/'models/scene.xml')
m=mujoco.MjModel.from_xml_path(str(P/'models/scene.xml')); d=mujoco.MjData(m)
# Gravity compensation supports slider inspection without a calibrated controller.
for b in range(1,m.nbody): m.body_gravcomp[b]=1
mujoco.mj_saveLastXML(str(P/'models/scene.xml'),m)
for _ in range(5000): mujoco.mj_step(m,d)
assert np.isfinite(d.qpos).all()
report=dict(mujoco=mujoco.__version__,nq=m.nq,nv=m.nv,nu=m.nu,actuators=names,time=d.time,warnings=[int(x.number) for x in d.warning],max_abs_qpos=float(np.max(np.abs(d.qpos))))
report_dir = P / 'docs' / 'model-validation'
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / 'validation.json').write_text(json.dumps(report, indent=2))
print(report)
