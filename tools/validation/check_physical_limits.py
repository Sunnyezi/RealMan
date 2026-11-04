from pathlib import Path
import json,sys,ast
import mujoco,numpy as np
root=Path(__file__).resolve().parents[2]
limits=json.loads((root/'physical_limits.json').read_text())
for rel in ['build_model.py','grasp/build_scene.py']:
    ast.parse((root/rel).read_text())
report={'models':{},'hand_calibrated':False}
for rel in ['models/scene.xml','grasp/scene.xml']:
    m=mujoco.MjModel.from_xml_path(str(root/rel));d=mujoco.MjData(m)
    rows=[]
    for i in range(m.nu):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(m.actuator_trnid[i,0]))
        if name in limits['arm']['joint_torque_limit_Nm']: cap=limits['arm']['joint_torque_limit_Nm'][name]
        elif name.startswith('right_'): cap=1.5
        else:continue
        assert m.actuator_forcelimited[i]
        assert np.allclose(m.actuator_forcerange[i],[-cap,cap])
        assert np.allclose(m.actuator_gear[i],[1,0,0,0,0,0])
        rows.append({'joint':name,'cap_Nm':cap})
    # Force large position error and verify saturation in the compiled dynamics.
    for sign in [-1,1]:
        d.ctrl[:]=sign*100
        mujoco.mj_forward(m,d)
        assert np.all(np.abs(d.actuator_force)<=np.max(np.abs(m.actuator_forcerange),axis=1)+1e-8)
    report['models'][rel]=rows
sys.path.insert(0,str(root/'grasp'))
from protocol import fingerprint
report['fingerprint']=fingerprint()
p=root/'grasp/checks/gate.json'
if p.exists():
    old=json.loads(p.read_text())
    report['old_gate_stale']=old.get('fingerprint')!=fingerprint()
(root/'docs/physical_limits_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
