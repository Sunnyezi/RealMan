"""Cached overview meshes for software-rendered VNC; never modifies training assets."""
from pathlib import Path
import hashlib,json
import xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent

def ensure_model():
 source=P/'scene.xml';tree=ET.parse(source);compiler=tree.find('compiler')
 meshdir=(P/compiler.get('meshdir','')).resolve();cache=P.parent/'work/view_lod';cache.mkdir(parents=True,exist_ok=True)
 digest=hashlib.sha256(source.read_bytes());digest.update(b'display-grid-v1')
 for mesh in tree.findall('./asset/mesh'):
  if mesh.get('file'):
   f=meshdir/mesh.get('file');st=f.stat();digest.update(f'{f}:{st.st_size}:{st.st_mtime_ns}'.encode())
 signature=digest.hexdigest();target=cache/'scene.xml';report_file=cache/'report.json'
 if target.exists() and report_file.exists():
  try:
   if json.loads(report_file.read_text()).get('signature')==signature:return target
  except (OSError,ValueError):pass
 import trimesh
 report=[]
 for i,mesh in enumerate(tree.findall('./asset/mesh')):
  if not mesh.get('file'):continue
  shape=trimesh.load(meshdir/mesh.get('file'),force='mesh');before=len(shape.faces)
  count=min(before,max(120,min(1600,int(before*.06))))
  simple=shape.simplify_quadric_decimation(face_count=count) if count<before else shape
  path=cache/f'{i:03d}.obj';simple.export(path);mesh.set('file',str(path))
  report.append(dict(name=mesh.get('name'),before=before,after=len(simple.faces)))
 asset=tree.find('asset')
 ET.SubElement(asset,'texture',name='viewer_floor_grid',type='2d',builtin='flat',rgb1='0.17 0.19 0.22',mark='edge',markrgb='0.40 0.43 0.48',width='128',height='128')
 ET.SubElement(asset,'material',name='viewer_floor_grid_mat',texture='viewer_floor_grid',texrepeat='2 2',texuniform='true',reflectance='0',specular='0.05',shininess='0.1')
 floor=tree.find(".//geom[@name='floor']");floor.set('material','viewer_floor_grid_mat');floor.set('rgba','1 1 1 1')
 compiler.set('meshdir',str(meshdir));tree.write(target)
 report_file.write_text(json.dumps(dict(signature=signature,original_faces=sum(m['before'] for m in report),display_faces=sum(m['after'] for m in report),meshes=report),indent=2))
 return target

if __name__=='__main__':print(ensure_model())
