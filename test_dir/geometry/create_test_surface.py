import os
import numpy as np
from scipy.interpolate import interp1d
from . import gen_triangle_mesh

def create_sample_surfaces(part_name='rocket',grid_type='smooth',
                           field_type='smooth', orientation='xy', save=False):
    config = 'test'
    flag_basis = 'sample'

    variables = ['cp', 'cfx', 'cfy', 'cfz']

    mesh = gen_triangle_mesh.Mesh(part_name, grid_type)
    x = mesh.x_mesh.flatten()
    y = mesh.y_mesh.flatten()
    z = mesh.z_mesh.flatten()
    n_points = mesh.size

    if part_name == 'rocket' or part_name == 'cylinder':
        hi_rf = interp1d(mesh.x_mesh[:,0],mesh.radius_adj[:,0]*mesh.core_radius)
        surface, lineload = generate_rocket(field_type, n_points, variables, x, y, z, hi_rf)
    elif part_name == 'flat':
        surface, lineload = generate_flat_plate(field_type, n_points, variables, x, y, z)

    if orientation == 'xz':
        surface[:,1], surface[:,2] = surface[:,2].copy(), surface[:,1].copy()
    elif orientation == 'yz':
        surface[:,0], surface[:,1], surface[:,2] = surface[:,2].copy(), surface[:,0].copy(), surface[:,1].copy()
    node_to_elem = mesh.connectivity
    data_by_node = surface
    
    os.makedirs('lineloads',exist_ok=True)
    if part_name == 'flat' and save:
        save_exact_lineload(os.path.join('lineloads', '_'.join([part_name,field_type,orientation])),lineload,variables,orientation)

    return node_to_elem, data_by_node


def generate_rocket(field_type, n_points, variables, x, y, z, hi_rf):
    surface = np.zeros((n_points, 7))
    surface[:, 0] = x
    surface[:, 1] = y
    surface[:, 2] = z
    
    hi_x = np.linspace(x.min(),x.max(),1001)
    lineload = np.zeros((len(hi_x),8,len(variables)))
    lineload[:,0,:] = np.tile(hi_x[None].T,(1,len(variables)))
    for n, var in enumerate(variables):
        n_index = 3 + n
        if 'smooth' in field_type:
            surface[:, n_index] = (n+1)*(x.max() - x)/(x.max() + n)
            # lineload[:, n_index, n] = 2*np.pi*hi_rf(hi_x) * n*(hi_x.max()-hi_x)/(hi_x.max()+n)
        elif 'bumpy' in field_type:
            surface[:, n_index] = (n+1)*np.cos((n+1)*np.pi*x/10)*np.sin((n+1)*np.pi*y/10)
        elif 'index' in field_type:
            surface[:, n_index] = np.arange(surface.shape[0])
    return surface, lineload

def generate_flat_plate(field_type, n_points, variables, x, y, z):
    surface = np.zeros((n_points, 7))
    surface[:, 0] = x
    surface[:, 1] = y
    surface[:, 2] = 0
    
    hi_x = np.linspace(x.min(),x.max(),1001)
    hi_y = np.linspace(y.min(),y.max(),1001)
    lineload = np.zeros((len(hi_x),8,len(variables),2))
    lineload[:,0,:,0] = np.tile(hi_x[None].T,(1,len(variables)))
    lineload[:,0,:,1] = np.tile(hi_y[None].T,(1,len(variables)))
    
    if 'step' in field_type:
        surface[ (x<=0) & (y<=0), 3] = 1
        lineload[hi_x<=0,3,0,0] = -1
        lineload[hi_x<=0,4,0,0] = 0.5
        lineload[hi_x<=0,5,0,0] = hi_x[np.where(hi_x<=0)[0]]
        lineload[hi_y<=0,3,0,1] = -1
        lineload[hi_y<=0,5,0,1] = -0.5
        lineload[hi_y<=0,4,0,1] = -hi_y[np.where(hi_y<=0)[0]]
    elif 'wavy' in field_type:
        surface[:, 3] = np.cos(x*3*np.pi)*(np.sin(y*3*np.pi)+1/5)
        #negative signs because force acts in "negative" direction
        lineload[:,3,0,0] = -2/5*np.cos(3*np.pi*hi_x)
        lineload[:,4,0,0] = -2/(3*np.pi)*np.cos(3*np.pi*hi_x)
        lineload[:,5,0,0] = 2/5*np.cos(3*np.pi*hi_x)*hi_x
    elif 'random' in field_type:
        for n, var in enumerate(variables):
            n_index = 3 + n
            np.random.seed(n_index)
            surface[:, n_index] = (x.max() - x)/(x.max() + n)
    return surface, lineload

def save_exact_lineload(filename,lineload,variables,orientation):
    if orientation == 'xy':
        modload = lineload.copy()
    elif orientation == 'xz':
        modload = lineload[:,[0,1,3,2,4,6,5],:,:].copy()
        modload[:,1:4,:,:] *= -1
    elif orientation == 'yz':
        modload = lineload[:,[0,3,1,2,6,4,5],:,:].copy()
    
    var = 'cp'
    i = 0
    for j in range(len(orientation)):    
        header = 'VARIABLES= '+orientation[j].upper()+'       Cx            Cy            Cz           Cmx           Cmy           Cmz       PROFILE'
        header += '\nZONE'
    #for i,var in enumerate(variables):
        np.savetxt(filename+'_exact_'+var+'_'+orientation[j]+'_LL.dat', modload[:,:,i,j], header=header, comments='')
        #if i == 0:
        #    full = lineload[:,:,i]
        #else:
        #    full[:,3:] += lineload[:,3:,i]
    #np.savetxt(filename+'_exact_all_LL.dat', full, header=header, comments='')
    return

if __name__ == '__main__':
    create_sample_surfaces()
