"""Create an editable first-pass mechanical glTF skin for Lemegeton.
Usage: python rig_lemegeton.py [source.glb] [output.glb]
No Blender required. Review/correct vertex assignments in Blender before animation.
"""
import json,struct,sys
from pathlib import Path
import numpy as np

src=Path(sys.argv[1] if len(sys.argv)>1 else '/mnt/data/LEMEGETON_MASTER_SOURCE.glb')
dst=Path(sys.argv[2] if len(sys.argv)>2 else '/mnt/data/LEMEGETON_RIGGED_DRAFT.glb')
with src.open('rb') as f:
 magic,ver,total=struct.unpack('<4sII',f.read(12))
 assert magic==b'glTF' and ver==2
 jl,jt=struct.unpack('<I4s',f.read(8)); data=json.loads(f.read(jl))
 bl,bt=struct.unpack('<I4s',f.read(8)); buf=bytearray(f.read(bl))
assert len(data['meshes'])==1 and not data.get('skins')
acc=data['accessors'][0]; view=data['bufferViews'][acc['bufferView']]
assert acc['componentType']==5123 and view['byteStride']==8
n=acc['count']
pos0=np.ndarray((n,3),dtype='<u2',buffer=buf,offset=view.get('byteOffset',0)+acc.get('byteOffset',0),strides=(8,2))
nd=data['nodes'][0]
scale=np.array(nd['scale'],np.float32);offset=np.array(nd['translation'],np.float32)
pos=pos0.astype(np.float32)*scale+offset
x,y,z=pos.T

# Names, parent index, joint position in glTF Y-up space.
bones=[
 ('ROOT',-1,(0,-.56,0)),
 ('Torso',0,(0,-.34,0)),
 ('Head',1,(0,.01,0)),
 ('Antenna',2,(0,.77,0)),
 ('Shoulder.L',1,( -.48,-.18,0)),
 ('UpperArm.L',4,(-.57,-.32,0)),
 ('Forearm.L',5,(-.66,-.48,0)),
 ('Hand.L',6,(-.70,-.66,0)),
 ('Shoulder.R',1,( .48,-.18,0)),
 ('UpperArm.R',8,(.57,-.32,0)),
 ('Forearm.R',9,(.66,-.48,0)),
 ('Hand.R',10,(.70,-.66,0)),
 ('Hip.L',1,(-.23,-.59,0)),
 ('Leg.L',12,(-.23,-.72,0)),
 ('Foot.L',13,(-.23,-.87,0)),
 ('Hip.R',1,(.23,-.59,0)),
 ('Leg.R',15,(.23,-.72,0)),
 ('Foot.R',16,(.23,-.87,0)),
]
# Side L/R are viewed from the front here; Blender may label these opposite
# depending on the mesh's intended front axis.
# Region weights: hard assignment suitable for a mechanical/robot model.
# Since the source is fused, edge cleanup and manual weight paint are expected.
group=np.ones(n,dtype=np.uint8)
head=(y>-.055)&(np.abs(x)<.59)
group[head]=2
group[(y>.745)&(np.abs(x)<.16)]=3
for sign, shoulder,upper,fore,hand,hip,leg,foot in [(-1,4,5,6,7,12,13,14),(1,8,9,10,11,15,16,17)]:
 s=(x*sign)
 # Arm components lie outside keyboard's approximate silhouette.
 arm=(s>.39)&(y<-.055)&(y>-.79)
 group[arm & (y>-.295)]=shoulder
 group[arm & (y<=-.295)&(y>-.44)]=upper
 group[arm & (y<=-.44)&(y>-.60)]=fore
 group[arm & (y<=-.60)]=hand
 # Boot legs below the keyboard.
 low=(s>.075)&(s<.4)&(y<-.58)
 group[low & (y>-.65)]=hip
 group[low & (y<=-.65)&(y>-.78)]=leg
 group[low & (y<=-.78)]=foot

# Bake original node transform into mesh positions; the skin and joints share space.
# Float vertex positions improve interoperability with skinning.
nd.pop('scale',None);nd.pop('translation',None)
views=data['bufferViews'];accessors=data['accessors']
def attach(array,component_type,type_name,normalized=False,target=None,minmax=False):
 raw=array.tobytes(order='C'); padding=(-len(buf))%4
 buf.extend(b'\0'*padding);o=len(buf);buf.extend(raw)
 view={'buffer':0,'byteOffset':o,'byteLength':len(raw)}
 if target is not None: view['target']=target
 vi=len(views);views.append(view)
 count=len(array)
 a={'bufferView':vi,'componentType':component_type,'count':count,'type':type_name}
 if normalized:a['normalized']=True
 if minmax:a['min']=array.min(0).tolist();a['max']=array.max(0).tolist()
 ai=len(accessors);accessors.append(a)
 return ai
pos_id=attach(pos.astype('<f4'),5126,'VEC3',target=34962,minmax=True)
joints=np.zeros((n,4),dtype='<u2');joints[:,0]=group
weights=np.zeros((n,4),dtype='u1');weights[:,0]=255
joint_id=attach(joints,5123,'VEC4',target=34962)
weight_id=attach(weights,5121,'VEC4',normalized=True,target=34962)
prim=data['meshes'][0]['primitives'][0]
prim['attributes']['POSITION']=pos_id
prim['attributes']['JOINTS_0']=joint_id
prim['attributes']['WEIGHTS_0']=weight_id

# Parent joints under the scene root. Local joint translations keep requested
# world-space bind locations while preserving a real bone hierarchy.
base=len(data['nodes']);world=np.array([v[2] for v in bones],np.float32)
for i,(name,parent,at) in enumerate(bones):
 local=world[i]-(world[parent] if parent>=0 else 0)
 bone={'name':name,'translation':local.tolist()}
 data['nodes'].append(bone)
for i,(name,parent,at) in enumerate(bones):
 if parent>=0:data['nodes'][base+parent].setdefault('children',[]).append(base+i)
data['nodes'][1].setdefault('children',[]).append(base)
# Inverse bind matrices are column-major (glTF convention).
ibm=np.tile(np.eye(4,dtype=np.float32),(len(bones),1,1))
ibm[:,0,3]=-world[:,0];ibm[:,1,3]=-world[:,1];ibm[:,2,3]=-world[:,2]
ibm_id=attach(ibm.transpose(0,2,1).copy().astype('<f4').reshape(len(bones),16),5126,'MAT4')
data['skins']=[{'name':'Lemegeton_Mechanical_Rig','joints':list(range(base,base+len(bones))),'skeleton':base,'inverseBindMatrices':ibm_id}]
data['nodes'][0]['skin']=0
# Original positions accessor can remain, but is not used.
data['buffers'][0]['byteLength']=len(buf)
data['asset'].setdefault('generator','Lemegeton Rig Draft Script')
js=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode('utf-8')
js+=b' '*((-len(js))%4)
buf.extend(b'\0'*((-len(buf))%4))
with dst.open('wb') as f:
 f.write(struct.pack('<4sII',b'glTF',2,12+8+len(js)+8+len(buf)))
 f.write(struct.pack('<I4s',len(js),b'JSON'));f.write(js)
 f.write(struct.pack('<I4s',len(buf),b'BIN\0'));f.write(buf)
print(f'{dst} | vertices={n}, joints={len(bones)}, bytes={dst.stat().st_size}')
for idx,(name,_,_) in enumerate(bones):print(f'{name:16}: {(group==idx).sum()} vertices')
