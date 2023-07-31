import numpy as np
import os
import json


def make_cylinder(height, radius, i_index=101, j_index=21):
    k_index = j_index
    coords = np.zeros([i_index, j_index, 3])

    x = np.linspace(-height, 0, i_index)
    y = np.cos(np.linspace(0, 2*np.pi,j_index))*radius
    z = np.sin(np.linspace(0, 2*np.pi,k_index))*radius

    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
    _, z_mesh = np.meshgrid(x, z, indexing='ij')
    coords[:, :,  0] = x_mesh
    coords[:, :,  1] = y_mesh
    coords[:, :,  2] = z_mesh
    return coords, x_mesh

def make_core():
    i_index = 101
    j_index = 21

    coords, x_mesh = make_cylinder(core_height, core_radius, i_index, j_index)

    radius_adj = core_radius_of_x(x_mesh)
    coords[:,:,1] = coords[:,:,1]*radius_adj
    coords[:,:,2] = coords[:,:,2]*radius_adj
    coords[:,:,0] = -coords[:,:,0]
    
    return coords

def core_radius_of_x(x):
    nose_start = -core_nose_height
    rho = (core_radius**2 + nose_start**2)/ (2*core_radius)

    return np.piecewise(x, [x < nose_start, x >= nose_start], 
            [1 , lambda x: (np.sqrt(rho**2 - (-nose_start + x)**2) + core_radius - rho)/core_radius])
            #[1 , lambda x: (nose_start - x)**2])

def make_srb():
    i_index = 101
    j_index = 21
    
    coords, x_mesh = make_cylinder(srb_height, srb_radius, i_index, j_index)

    radius_adj = srb_cone_nose(x_mesh)
    coords[:,:,1] = coords[:,:,1]*radius_adj
    coords[:,:,2] = coords[:,:,2]*radius_adj
    coords[:,:,0] = -coords[:,:,0]

    return coords



def srb_cone_nose(x):
    nose_start = -srb_nose_height
    return np.piecewise(x, [x < nose_start, x >= nose_start], [1 , lambda x: x/nose_start])





















