import numpy as np
from numpy.linalg import norm
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

    def single_ra_slice(self, rot_axis, nom_angle = np.deg2rad(1.5), vis=False):
        '''
        Slices about an axis, using bounds on h_min and h_max. Returns set of curves for each layer.
        Uses a bisection search to determine angular slice increase
        rot_axis : 2x3 numpy array. axis about which the slicing plane should rotate. 
            2 points to define this line.
        '''
        # define the base slice as the x,y plane
        base_slice = _plane_cut(np.array([0,0,1,0],vis=True))
        return True


    def _plane_cut(self, plane, vis=False):
        '''
        Cuts the STL with a plane and extracts the points that are within the 
        distance threshold of the plane.
        plane: R^4 vector representing the normal vector (x,y,z) and the distance (r) 
            along the normal. Represents the equation ax+by+cz+d=0.
        '''
        # loop through all faces in .stl file
        in_plane_pts = []
        for i in self.mod_mesh.vectors:
            pt, lam = self._intersect_line_plane(i[[0,1],:], plane)
            if (lam is not None) and (0<=lam<=1):
                in_plane_pts.append(pt)

        in_plane_pts = np.array(in_plane_pts)

        # normalize distance between points
        # TODO: how should we determine closed vs. open part?
        # TODO: how should we pick the starting point?
        in_plane_pts = self._normalize_point_distance(in_plane_pts)

        if vis:
            # initialize mesh for plane
            xx, yy = np.meshgrid(range(-30,30), range(-30, 30))
            z = (-plane[0]*xx-plane[1]*yy-plane[3])/plane[2]
            fig = plt.figure()
            axes = fig.add_subplot(projection='3d')
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.mod_mesh.vectors))
            axes.plot_surface(xx,yy,z, alpha=0.2)
            axes.scatter(in_plane_pts[:,0], in_plane_pts[:,1], in_plane_pts[:,2], c='r')
            axes.set_aspect('equal')
            plt.show()

        return in_plane_pts

    def _intersect_line_plane(self,line,plane, epsilon = 1e-6):
        '''
        Calculates the intersection point of a line segment and a plane.
        If the intersection isn't between the two points on the line, then retursn None
        line: np array (2,3). Row: point, Col: x,y,z coords
        plane: plane of the form ax+by+cz+d=0
        returns: intersection point, lambda
        if lambda not in (0,1), line segment does not intersect
        Math taken from https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
        '''
        u =line[1,:]-line[0,:]
        dot = u.dot(plane[:3])

        if abs(dot) > epsilon:
            p_co = plane[:3]*(-plane[3]/plane[:3].dot(plane[:3]))
            w = line[0,:]-p_co
            fac = -plane[:3].dot(w)/dot
            u = u*fac
            return line[0,:]+u, fac

        return None, None

    def _normalize_point_distance(self, pts, pts_dist=3, start_pt=None, closed=True):
        '''
        Normalize the distance between points in a list of points. Iterates through distances 
        to points, going first to the next closest point.
        Assumes closed section.

        pts: list of points at irregular distances
        pt_dist: distance between points (mm)
        start_pt: Point to start with. Must be in list of pts. If None, choose a random point.
        closed: flag denoting if section is closed or open
        '''
        # choose random point if one not specified
        if start_pt is None:
            start_pt = pts[np.random.choice(pts.shape[0])]
        # deletes point where distance is minimum, must be the start point
        pts = np.delete(pts, np.argmin(norm(pts-start_pt)), axis=0)
        norm_pts = []
        # iterate through all points, finding the closest to the current point
        # interpolate to find points at distances
        curr_pt = start_pt
        norm_pts.append(start_pt)
        while pts.shape[0]>0:
            # find the next point as the closest point to the current point
            next_pt = pts[np.argmin(norm(pts-curr_pt, axis=1))]
            # remove from pts list to remove possibility of finding it when searching for next point
            pts = np.delete(pts, np.argmin(norm(pts-next_pt,axis=1)), axis=0)
            # find distance and interpolate until distance distance to next point 
            #   is less than pts_dist
            dist_to_next = norm(curr_pt-next_pt)
            while dist_to_next>pts_dist:
                # travel along line between points at a distance of pts_dist
                fac = pts_dist/dist_to_next
                curr_pt = curr_pt+(next_pt-curr_pt)*fac
                norm_pts.append(curr_pt)
                dist_to_next = norm(curr_pt-next_pt)
        # finish traversing to start point, twice points dist to account for start and end of bead
        dist_to_next = norm(curr_pt-start_pt)
        while dist_to_next>(pts_dist*2):
            fac = pts_dist/dist_to_next
            curr_pt = curr_pt+(start_pt-curr_pt)*fac
            norm_pts.append(curr_pt)
            dist_to_next = norm(curr_pt-start_pt)

        return np.array(norm_pts)

    def _get_centerline(self):
        centerline=None
        return centerline

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
    normal = np.array([1, 0, 1])
    normal = normal/np.linalg.norm(normal)
    dist = -20
    plane = np.zeros(4)
    plane[:3] = normal
    plane[3] = dist
    
    slicer = nuhSlicer(stl_file, tf)
    slicer._plane_cut(plane, vis=True)

