from pathlib import Path
import mujoco,numpy as np,json
P=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(P/'models/scene.xml'));d=mujoco.MjData(m)
for k in range(5000):
 for i in range(6,m.nu):
  lo,hi=m.actuator_ctrlrange[i];d.ctrl[i]=lo+0.3*(hi-lo)*(0.5-0.5*np.cos(k*m.opt.timestep))
 mujoco.mj_step(m,d)
 assert np.isfinite(d.qpos).all()
assert not any(x.number for x in d.warning)
print(json.dumps(dict(time=d.time,nu=m.nu,nq=m.nq,warnings=[int(x.number) for x in d.warning],qpos=d.qpos.tolist()),indent=2))
