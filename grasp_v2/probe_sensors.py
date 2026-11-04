import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import xml.etree.ElementTree as E,json,numpy as np,mujoco
from env import P

def probe_regions():
 r=E.parse(P/'scene.xml').getroot();g=r.find('.//geom[@name="object_geom"]');g.set('type','sphere');g.set('size','.003');m=mujoco.MjModel.from_xml_string(E.tostring(r).decode(),assets={str(p):p.read_bytes() for p in []}) if False else None
 # Preserve mesh resolution relative to the source directory.
 path=P/'checks/sensor_probe.xml';path.parent.mkdir(exist_ok=True);r.find('compiler').set('meshdir','../../models/meshes');E.ElementTree(r).write(path)
 m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);cfg=json.load(open(P/'config.json'));mapping=json.load(open(P/'sensor_map.json'));oj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free');oa=m.jnt_qposadr[oj];rows=[]
 for entry in mapping:
  mujoco.mj_resetData(m,d);d.qpos[:6]=cfg['home_q'];mujoco.mj_forward(m,d);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,entry['site']);sensor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SENSOR,entry['sensor']);center=d.site_xpos[site].copy();normal=d.site_xmat[site].reshape(3,3)[:,2]
  d.qpos[oa:oa+3]=center+normal*.001;mujoco.mj_forward(m,d);force=float(d.sensordata[m.sensor_adr[sensor]]);rows.append(dict(region=entry['region'],force_N=force,responded=force>1e-4))
 return rows
if __name__=='__main__':
 rows=probe_regions();(P/'checks/sensor_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2));assert all(x['responded'] for x in rows)
