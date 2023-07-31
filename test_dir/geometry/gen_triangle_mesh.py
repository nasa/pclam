import numpy as np
from scipy.spatial import Delaunay


class Mesh(object):
    def __init__(self, part_type, case, i=101, j=21):
        self.case = case
        if part_type == 'rocket':
            self.make_core(i,j)
            self.get_triangular_connectivity()
            self.y_mesh = self.y_mesh_for_connectivity
            self.y_mesh = np.cos(self.y_mesh_for_connectivity*np.pi) * self.radius_adj
            self.z_mesh = np.sin(self.z_mesh *np.pi) * self.radius_adj
            self.size = self.x_mesh.size
            self.clean_up_connectivity(i,j)

        if part_type == 'cylinder':
            self.make_core(i,j)
            self.get_triangular_connectivity()
            self.y_mesh = self.y_mesh_for_connectivity
            self.y_mesh = np.cos(self.y_mesh_for_connectivity*np.pi) 
            self.z_mesh = np.sin(self.z_mesh *np.pi) 
            self.size = self.x_mesh.size
        if part_type == 'flat':
            self.make_plate(i,j)
            self.get_triangular_connectivity()
            self.y_mesh = self.y_mesh_for_connectivity
            self.size = self.x_mesh.size


    def get_triangular_connectivity(self):
        coordinates = np.asarray([self.x_mesh.flatten(), self.y_mesh_for_connectivity.flatten()]).T
        tri = Delaunay(coordinates)
        self.connectivity = tri.simplices

    def clean_up_connectivity(self, i, j):
        # by "pinching" the nose of the rocket we have j points in the same spot causing a lot of bad cells
        # going to replace all instances of j in connectivity with 0
        # going to remove any instances where a cell has the 0 node twice
        self.connectivity = np.where(self.connectivity < j,  0, self.connectivity)
        repeaters = ((self.connectivity[:,0] == self.connectivity[:,1]) |
                     (self.connectivity[:,1] == self.connectivity[:,2]) |
                     (self.connectivity[:,0] == self.connectivity[:,2]))
        self.connectivity = self.connectivity[~repeaters]

        return
    
    def make_core(self, i_index, j_index):

        core_height = 27
        self.core_radius = 3
        self.make_cylinder(core_height, self.core_radius, i_index, j_index)

        self.radius_adj = self.core_radius_of_x()
        if 'bumpy' in self.case:
            indices = (self.radius_adj[:,0]==1) & (np.mod(np.arange(i_index),11) == 0) 
            self.radius_adj[indices,:] *= 1.05
        
        return 
    
    def make_plate(self, i_index=101, j_index=21):
        k_index = j_index

        x = np.linspace(-1, 1, i_index)
        y = np.linspace(-1, 1, j_index)
        if 'cubic' in self.case:
            x = .8*x**3+.2*x
            y = .8*y**3+.2*y
        elif 'step' in self.case:
            x = np.insert(x, int(i_index/2), 2/(i_index-1)/10)
            y = np.insert(y, int(j_index/2), 2/(j_index-1)/10)
        z = 0

        self.x_mesh, self.y_mesh_for_connectivity = np.meshgrid(x, y, indexing='ij')
        #self.x_mesh, self.y_mesh_for_connectivity = np.meshgrid(x, y_for_connectivity, indexing='ij')
        _, self.z_mesh = np.meshgrid(x, z, indexing='ij')
        return 

    def make_cylinder(self, height, radius, i_index=101, j_index=21):
        k_index = j_index

        x = np.linspace(0, height, i_index)
        if 'bumpy' in self.case:
            x = x + np.sin(x)**2
        y = np.cos(np.linspace(0, 2*np.pi,j_index))*radius
        y_for_connectivity = np.linspace(-1, 1, j_index)
        #z = np.sin(np.linspace(0, 2*np.pi,k_index))*radius
        z= np.linspace(-1, 1, j_index)

        self.x_mesh, self.y_mesh = np.meshgrid(x, y, indexing='ij')
        self.x_mesh, self.y_mesh_for_connectivity = np.meshgrid(x, y_for_connectivity, indexing='ij')
        _, self.z_mesh = np.meshgrid(x, z, indexing='ij')
        return 


    def core_radius_of_x(self):
        x = self.x_mesh
        nose_start = 4
        rho = (self.core_radius**2 + nose_start**2)/ (2*self.core_radius)

        return np.piecewise(x, [x > nose_start, x <= nose_start], 
                [1 , lambda x: (np.sqrt(rho**2 - (-nose_start + x)**2) + self.core_radius - rho)/self.core_radius])



