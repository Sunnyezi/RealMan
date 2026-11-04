import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json,time
import numpy as np,mujoco
from PIL import Image
P=Path(__file__).parent
class RGBD:
 def __init__(self,m):
  self.m=m;self.renderer=mujoco.Renderer(m,128,128);self.cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'rgbd');self.last=np.zeros(3);self.last_seen=-100.;self.visible=False;self.rgb=None;self.depth=None
 def update(self,d):
  opt=mujoco.MjvOption();opt.geomgroup[3]=0
  self.renderer.disable_depth_rendering();self.renderer.update_scene(d,camera=self.cid,scene_option=opt);self.rgb=self.renderer.render().copy()
  self.renderer.enable_depth_rendering();self.depth=self.renderer.render().copy();self.renderer.disable_depth_rendering()
  rgb=self.rgb.astype(float);mask=(rgb[:,:,2]>35)&(rgb[:,:,2]>3.*rgb[:,:,0])&(rgb[:,:,2]>3.*rgb[:,:,1])&np.isfinite(self.depth)&(self.depth>.01)&(self.depth<2)
  self.visible=bool(mask.sum()>=4)
  if self.visible:
   v,u=np.where(mask);z=self.depth[v,u];f=64/np.tan(np.deg2rad(self.m.cam_fovy[self.cid])/2)
   pts=np.column_stack([(u-63.5)*z/f,-(v-63.5)*z/f,-z]);world=pts@d.cam_xmat[self.cid].reshape(3,3).T+d.cam_xpos[self.cid]
   low,high=np.quantile(world,[.02,.98],axis=0);center=(low+high)/2
   # Known 4 cm, initially axis-aligned cube: correct single-face depth bias.
   for k in range(3):
    if high[k]-low[k]<.025:center[k]+=.02*np.sign(center[k]-d.cam_xpos[self.cid,k])
   self.last=center;self.last_seen=float(d.time)
  return self.last.copy(),self.visible,min(10.,float(d.time)-self.last_seen)
 def close(self):self.renderer.close()
if __name__=='__main__':
 m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);cfg=json.load(open(P/'config.json'));d.qpos[:6]=cfg['home_q'];d.ctrl[:6]=cfg['home_q'];mujoco.mj_forward(m,d)
 v=RGBD(m);print('estimate',v.update(d),'true',cfg['object_xyz']);Image.fromarray(v.rgb).save(P/'rgbd_initial.png');np.save(P/'depth_initial.npy',v.depth);v.close()
