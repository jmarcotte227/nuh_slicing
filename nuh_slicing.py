import numpy as np
import stl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class nuhSlicer:
    def __init__(self, model_file, transform = None):
        '''
        Initialize the slicer class with 
        model_file: the filepath to an stl to be sliced.
        transform: optional input, if provided will transform the stl on load
        '''
        # load model
        self.mod_mesh = stl.mesh.Mesh.from_file(model_file)
        # identify the base of the stl

        # shift base such that mean of the points is in the center
        if transform is not None:
            self.mod_mesh.transform(transform)
        # Extract Vertices
        self.vertices = np.reshape(self.mod_mesh.vectors,(-1,3))

    def vis_mesh(self):
        '''
        plots the mesh in matplotlib 3d plot
        '''
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.mod_mesh.vectors))
        axes.set_aspect('equal')
        plt.show()

    def eval_mesh(self):
        '''
        prints the mass properties of the imported mesh.
        This is good for verifying the mesh was read correctly.
        '''

        volume, cog, inertia = self.mod_mesh.get_mass_properties()
        print("Volume: ", volume)
        print("COG: ", cog)
        print("Inertia Tensor: ", inertia)

    def plane_cut(self, plane, vis=False, threshold = 3):
        '''
        Cuts the STL with a plane and extracts the points that are within the distance threshold of the plane.
        plane: R^4 vector representing the normal vector (x,y,z) and the distance (r) along the normal.
            Represents the equation ax+by+cz+d=0.
        '''
        # check which points in the list of verticies satisfy the plane equation
        plane_dist = self.vertices.dot(plane[0:3])-plane[3]
        idxs = np.where(np.logical_and(plane_dist<=threshold, plane_dist>=-threshold))
        in_plane = self.vertices[idxs]
        print(in_plane.shape)

        if vis:
            # initialize mesh for plane
            xx, yy = np.meshgrid(range(-30,30), range(-30, 30))
            z = (-plane[0]*xx-plane[1]*yy+plane[3])/plane[2]
            fig = plt.figure()
            axes = fig.add_subplot(projection='3d')
            axes.plot_surface(xx,yy,z, alpha=0.2)
            axes.scatter(in_plane[:,0], in_plane[:,1], in_plane[:,2], c='r')
            # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.mod_mesh.vectors))
            axes.set_aspect('equal')
            plt.show()

    def _intersect_line_plane(self,line,plane, epsilon = 1e-6):
        '''
        Calculates the intersection point of a line segment and a plane.
        If the intersection isn't between the two points on the line, then retursn None
        line: np array (2,3). Row: point, Col: x,y,z coords
        plane: plane of the form ax+by+cz+d=0
        Math taken from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
        '''
        u =line[1,:]-line[0,:]
        dot = u.dot(plane[:3])

        if abs(dot) > epsilon:
            p_co = plane[:3]*(-plane[3]/plane[:3].dot(plane[:3]))
            w = line[0,:]-p_co
            fac = -plane[:3].dot(w)/dot
            print(fac)
            u = u*fac
            return line[0,:]+u

        return None

if __name__=='__main__':
    stl_file = 'models/funnel_tube_solid.stl'

    # transform to shift test part
    tf = np.array([
        [1, 0, 0, 100],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    # define the parameters of the plane
    normal = np.array([0, 0, 1])
    normal = normal/np.linalg.norm(normal)
    dist = -0.5
    plane = np.zeros(4)
    plane[:3] = normal
    plane[3] = dist
    
    slicer = nuhSlicer(stl_file, tf)
    # slicer.plane_cut(plane, vis=True)
    line = np.array([[1,1,1],[-1, -1, -1]])
    point = slicer._intersect_line_plane(line, plane)
    
    xx, yy = np.meshgrid(range(-2,2), range(-2, 2))
    z = (-plane[0]*xx-plane[1]*yy-plane[3])/plane[2]
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.plot_surface(xx,yy,z, alpha=0.2)
    axes.plot3D(line[:,0], line[:,1], line[:,2])
    axes.scatter(point[0], point[1], point[2], c='r')
    plt.show()


    # plotting to test

