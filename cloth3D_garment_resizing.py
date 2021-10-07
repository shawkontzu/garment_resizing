import bpy
import numpy as np
from struct import pack, unpack
import scipy.io as sio
import pickle
import mathutils
from scipy import sparse


#load cloth3D garment info
def load_garment_info(filename):
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    del data['__globals__']
    del data['__header__']
    del data['__version__']
    return check_keys(data)

def check_keys(dict):
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = todict(dict[key])
    return dict

def todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        elif isinstance(elem, np.ndarray) and np.any([isinstance(item, sio.matlab.mio5_params.mat_struct) for item in elem]):
            dict[strg] = [None] * len(elem)
            for i,item in enumerate(elem):
                if isinstance(item, sio.matlab.mio5_params.mat_struct):
                    dict[strg][i] = todict(item)
                else:
                    dict[strg][i] = item
        else:
            dict[strg] = elem
    return dict

#load cloth3D garment obj
def load_garment_mesh(garment_obj_dir, x_offset):
    garment_import_obj = bpy.ops.import_scene.obj(filepath=garment_obj_dir)
    garment_obj = bpy.context.selected_objects[0]  
    garment_obj.rotation_euler = (0,0,0)
    garment_obj.location += mathutils.Vector((x_offset,0,0))
    return garment_obj

def load_garment_laplacian(garment_laplacian_dir):
    laplacian_matrix = sparse.load_npz(garment_laplacian_dir)
    return laplacian_matrix.toarray()

#load smpl parameters
def load_smpl_params(params_dir):
    with open(params_dir, 'rb') as fp:
        smpl_params = pickle.load(fp)
        
    return smpl_params

#load smpl model
def load_smpl_model(smpl_model_dir):
    with open(smpl_model_dir, 'rb') as fp:
        smpl_model = pickle.load(fp, encoding="latin")
    return smpl_model
 

#load smpl object
def load_smpl_mesh(smpl_obj_dir, x_offset):
    smpl_import_obj = bpy.ops.import_scene.obj(filepath=smpl_obj_dir)
    smpl_obj = bpy.context.selected_objects[0]
    smpl_obj.location += mathutils.Vector((x_offset, 0, 0))
    return smpl_obj


#find nearest neighbor vertex on smpl
def find_nearest_neighbors(garment_obj, smpl_obj):
    # for each vertex in garment mesh, find nearest vertex in smpl obj
    # return the index of the nearest point in smpl obj
    
    # create a kd tree for smpl obj
    smpl_mesh = smpl_obj.data
    smpl_mesh_size = len(smpl_mesh.vertices)
    kd = mathutils.kdtree.KDTree(smpl_mesh_size)
    for i,v in enumerate(smpl_mesh.vertices):
        kd.insert(v.co, i)
        
    kd.balance()
    
    #find closest point on smpl object to each vertex on garment obj
    garment_mesh = garment_obj.data
    nearest_neighbors = []
    for i,v in enumerate(garment_mesh.vertices):
        co, index, dist = kd.find(v.co)
        # i: ith vertex
        # index: index of neighbor vertex
        # dist: distance to neighbor vertex
        nearest_neighbors.append([i, index, dist])
    return nearest_neighbors

def find_garment_shapedirs(smpl_shapedirs, nearest_neighbors):
    garment_shapedirs = []
    for i in range(len(nearest_neighbors)):
        index = nearest_neighbors[i][1]
        garment_shapedirs.append(smpl_shapedirs[index])
    garment_shapedirs = np.stack(garment_shapedirs, axis=0)
    
    return garment_shapedirs

##laplacian smoothing
#def laplacian_smoothing(shapedirs, laplacian_matrix):
#    N = shapedirs.shape[0]
#    if laplacian_matrix.shape[0] != N:
#        print("wrong dimensions")
#        exit(0)
#    new_shapedirs = np.zeros((N, 3, 10))
#    for i in range(N):
#        for j in range(N):
#            if (laplacian_matrix[i][j] != 0): # vertex i, j are connected by edge
#                for x in range(3):
#                    for y in range(10):
#                        new_shapedirs[i][x][y] += laplacian_matrix[i][j] * new_shapedirs[j][x][y]
#    return new_shapedirs


def laplacian_smoothing(shapedirs, laplacian_matrix):
    N = shapedirs.shape[0]
    laplacian_matrix = laplacian_matrix[:N, :N]
    # reshape shapedirs NX3X10 -> NX30
    new_shapedirs = np.reshape(shapedirs, (-1, shapedirs.shape[1]*shapedirs.shape[2]))
    print(new_shapedirs.shape)
#    smoothed_shapedirs = np.matmul(laplacian_matrix, new_shapedirs)
    smoothed_shapedirs = laplacian_matrix.dot(new_shapedirs)
    smoothed_shapedirs = np.reshape(smoothed_shapedirs, (-1, 3, 10))
    return smoothed_shapedirs
                


def garment_resize(garment_obj, garment_shapedirs, betas):
    displacement = np.matmul(garment_shapedirs, betas)
    print(displacement.shape)
    garment_mesh = garment_obj.data
    template_garment_obj = garment_obj.copy()
    template_garment_obj.data = garment_obj.data.copy()
    template_garment_mesh = template_garment_obj.data
    for i, v in enumerate(garment_mesh.vertices):
        template_garment_mesh.vertices[i].co[0] = garment_mesh.vertices[i].co[0] + displacement[i][0]
        template_garment_mesh.vertices[i].co[1] = garment_mesh.vertices[i].co[1] + displacement[i][1]
        template_garment_mesh.vertices[i].co[2] = garment_mesh.vertices[i].co[2] + displacement[i][2]
    bpy.context.collection.objects.link(template_garment_obj)
    template_garment_obj.location += mathutils.Vector((3,0,0))
    return template_garment_mesh

def reverse_garment_resize(garment_obj, garment_shapedirs, betas):
#    displacement = np.matmul(garment_shapedirs, betas)
    displacement = garment_shapedirs.dot(betas)
    print("displacement shape: ", displacement.shape)
    garment_mesh = garment_obj.data
    template_garment_obj = garment_obj.copy()
#    template_garment_obj.data = garment_obj.data.copy()
#    template_garment_mesh = template_garment_obj.data
    for i, v in enumerate(garment_mesh.vertices):
        template_garment_obj.data.vertices[i].co[0] = garment_mesh.vertices[i].co[0] - displacement[i][0]
        template_garment_obj.data.vertices[i].co[1] = garment_mesh.vertices[i].co[1] - displacement[i][1]
        template_garment_obj.data.vertices[i].co[2] = garment_mesh.vertices[i].co[2] - displacement[i][2]
    bpy.context.collection.objects.link(template_garment_obj)
    template_garment_obj.location += mathutils.Vector((2,0,0))
    return template_garment_obj.data


if __name__ == "__main__":
    # cloth3D garment object path
    gament_object_path = ""
    # cloth3D garment info.mat path
    garment_info_path = ""
    # cloth3D laplacian file path
    garment_laplacian_path = ""
    # smpl object generated using default pose and betas from garment info.mat file
    shaped_smpl_object_path = ""
    # smpl object generated using default pose and default betas, template garment should fit a template smpl object
    template_smpl_object_path = ""
    # smpl model path where smpl shapedirs is included
    smpl_model_path = ""
    
    # load garment object from cloth3D data
    shaped_garment_obj = load_garment_mesh(garment_object_path, -2)
    copy_garment_obj = load_garment_mesh(garment_object_path, -2)
    print("garment number of vertices: ", len(shaped_garment_obj.data.vertices))
    
    
    # load cloth3D garment meta data
    shaped_garment_info = load_garment_info(garment_info_path)
    shaped_garment_laplacian = load_garment_laplacian(garment_laplacian_path)
    print("laplacian shape: ", shaped_garment_laplacian.shape)
    
    # smpl object generated using cloth3d garment meta data
    shaped_smpl_obj = load_smpl_mesh(shaped_smpl_object_path, -2)
    # smpl template object
    template_smpl_obj = load_smpl_mesh(template_smpl_object_path, 0)
    
    
    # smpl base model
    smpl_model = load_smpl_model(smpl_model_path)
    smpl_shapedirs = smpl_model['shapedirs']
    
    # find nearest garment object's nearest neighbor vertices on smpl object
    nearest_neighbors = find_nearest_neighbors(shaped_garment_obj, shaped_smpl_obj)
    garment_shapedirs = find_garment_shapedirs(smpl_shapedirs, nearest_neighbors)
    for i in range(0):
        print("smoothing {} iteration".format(i))
        garment_shapedirs = laplacian_smoothing(garment_shapedirs, shaped_garment_laplacian)
    betas = shaped_garment_info["shape"]
    print("betas: ", betas)
    
    # resize the shaped garment back to template garment
    template_garment_mesh = reverse_garment_resize(copy_garment_obj, garment_shapedirs, betas)
    template_garment_obj = bpy.data.objects.new('template_garment', template_garment_mesh)
    bpy.data.objects.remove(copy_garment_obj)
