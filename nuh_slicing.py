import numpy as np
from numpy.linalg import norm
import stl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import general_robotics_toolbox as rox
import ezdxf
from scipy.spatial import Voronoi
import trimesh

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

        # Initialize space for curve_sliced data
        self.curve_sliced = {}
        
        # initialize centerline
        self.cl = None

    def vis_mesh(self):
        '''
        plots the mesh in matplotlib 3d plot
        '''
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.mod_mesh.vectors,
            # facecolor=(0,0,1,0.2),
            linewidth=0.5,
            edgecolor=(0,0,0)
            ))
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

    def single_ra_slice(self, rot_dist = 100, nom_angle = np.deg2rad(1.5), vis=False):
        '''
        Slices about an axis parallel to the y axis, using bounds on h_min and h_max. 
        Returns set of curves for each layer.
        rot_axis : 2x3 numpy array. axis about which the slicing plane should rotate. 
            2 points to define this line.
        '''
        # define the base slice as the x,y plane
        x_y_plane = np.array([0,0,1,0])
        base_slice = self._plane_cut(x_y_plane)
        layer_num = 0
        rot_angle = 0

        pos_curve_sliced = {
                layer_num: base_slice
                }
        self.curve_sliced = {}

        while rot_angle<(np.deg2rad(90)-nom_angle): # TODO: Come up with a better condition here
            # calculate plane equation
            # direction vector of the axis
            # TODO: generalize to arbitrary axis. Will make things a lot more complicate
            # rotate normal of the plane about the origin
            rot_angle += nom_angle
            layer_num += 1
            R = rox.rot([0,1,0], rot_angle) # might be able to use this to generalize
            n = R@x_y_plane[0:3]
            slice_plane = np.zeros(4)
            slice_plane[0:3] = n
            slice_plane[3] = -np.dot(slice_plane[0:3], np.array([100, 0, 0]))
            pos_curve_sliced[layer_num] = self._plane_cut(slice_plane)

        # loop through and calculate direction to closest point in the previous layer
        self._find_dir_vec(pos_curve_sliced)

    def calc_centerline(self, vis=False):
        points_flat = np.reshape(self.mod_mesh.points, (-1,3))
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        axes.scatter(points_flat[:,0], points_flat[:,1], points_flat[:,2])
        axes.set_aspect('equal')
        plt.show()
        vor = Voronoi(points_flat)


    def centerline_slice(self, centerline,  vis=False):
        '''
        Slices at fixed distance along the centerline of the part. Currently requires a
        centerline with a known equation.
        Centerline: nx3 numpy array with each point at a fixed distance along the centerline.
        '''
        x_y_plane = np.array([0,0,1,0])
        base_slice = self._plane_cut(x_y_plane)
        pos_curve_sliced = {
                0: base_slice
                }
        self.curve_sliced = {}
        # already sliced first point
        for idx, pt in enumerate(centerline):
            if (idx != 0) and (idx != len(centerline)-1):
                # calculate normal using previous and next point
                normal = (centerline[idx+1]-centerline[idx-1])/norm(centerline[idx+1]-centerline[idx-1])
                # calculate plane using normal and current centerline point
                d = - np.dot(normal, pt)
                slice_plane = np.zeros(4)
                slice_plane[:3] = normal
                slice_plane[3] = d
                pos_curve_sliced[idx] = self._plane_cut(slice_plane)
        # approximate normal of last point as previous point and current point
        normal = (centerline[-1]-centerline[-2])/norm(centerline[-1]-centerline[-2])
        d = - np.dot(normal, centerline[-1])
        slice_plane = np.zeros(4)
        slice_plane[:3] = normal
        slice_plane[3] = d
        pos_curve_sliced[len(centerline)-1] = self._plane_cut(slice_plane)

        self._find_dir_vec(pos_curve_sliced)

    def load_centerline(self, filepath):
        '''
        Loads the centerline data from a .dxf file and formats it to the required shape
        filepath: centerline dxf filepath
        '''
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        arc = msp[0]
        # TODO: Need to figure out if this import is worth it




    def _find_dir_vec(self, pos_curve_sliced):
        for key, pos_curve in pos_curve_sliced.items():
            curve = np.zeros((pos_curve.shape[0], 6))
            curve[:,0:3] = pos_curve
            # if first layer, the direction is straight down
            if key == 0:
                curve[:,3:] = np.array([0,0,-1])
            else:
                pos_curve_prev = pos_curve_sliced[key-1]
                for idx, point in enumerate(pos_curve):
                    # find the 5 closest points in the previous layer
                    min_pt_idx = np.argsort(norm(point - pos_curve_prev, axis=1))[:1]
                    #average position of the 5 closest points
                    avg_point_prev = np.mean(pos_curve_prev[min_pt_idx], axis=0)
                    # draw a unit vector in the direction of the point
                    curve[idx,3:] = (avg_point_prev-point)/norm(avg_point_prev-point)
            self.curve_sliced[key] = curve

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
            xx, yy = np.meshgrid(range(-30,60), range(-30, 30))
            z = (-plane[0]*xx-plane[1]*yy-plane[3])/plane[2]
            fig = plt.figure()
            axes = fig.add_subplot(projection='3d')
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(self.mod_mesh.vectors))
            axes.plot_surface(xx,yy,z, alpha=0.2)
            axes.scatter(in_plane_pts[:,0], in_plane_pts[:,1], in_plane_pts[:,2], c='r')
            axes.set_aspect('equal')
            plt.show()

        return in_plane_pts

    def _calc_min_height(self):
        idx=0
        return idx
    def _calc_max_height(self):
        idx=0
        return idx
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

    def _normalize_point_distance(self, pts, pts_dist=1, start_pt=None, closed=True):
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

    def vis_curvesliced(self, dir_vec = True):
        if not self.curve_sliced:
            print("model not sliced yet")
            return

        # initialize mesh for plane
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')
        for key, curve in self.curve_sliced.items():
            axes.scatter(curve[:,0], curve[:,1], curve[:,2], c='r')
            if dir_vec:
                len_fact = 2
                axes.quiver(curve[:,0], 
                            curve[:,1],
                            curve[:,2],
                            curve[:,3]*len_fact,
                            curve[:,4]*len_fact,
                            curve[:,5]*len_fact)
        axes.plot3D(self.cl[:,0], self.cl[:,1], self.cl[:,2], c='b')
        axes.set_aspect('equal')
        plt.show()

def pt_line_projection(line, pt):
    '''
    Projects a single point onto a line.
    line: 2x3 matrix containing 3d points on either end of a line
    pt: 3 vector of the point to be projected
    '''
    p_0 = line[0]
    v = line[1]-line[0]
    proj_val = v@v.T/(v.T@v)

    # orthogonal projection to be implemented
    # https://math.stackexchange.com/questions/62633/orthogonal-projection-of-a-point-onto-a-line

if __name__=='__main__':
    stl_file = 'models/funnel_tube_solid.stl'
    cl_file = 'models/funnel_tube_cl.dxf'

    # transform to shift test part
    tf = np.array([
        [1, 0, 0, 100],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])

    # define the parameters of the plane
    slicer = nuhSlicer(stl_file, tf)
    slicer.calc_centerline()
    # slicer.vis_mesh()
    # slicer.load_centerline(cl_file)

    # define the centerline as circle of radius 100 for 100 slices
    num_points = 25
    cl = np.zeros((num_points,3))
    cl_len = (2*np.pi/4)*100 # len of centerline in mm
    ang_per_seg = -np.deg2rad(90/(num_points-1))+0.001
    for i in range(num_points):
        base_pt = np.array([-100, 0, 0])
        cl[i, 0] = base_pt[0]*np.cos(ang_per_seg*i)- np.sin(ang_per_seg*i)*base_pt[2]
        cl[i,2] = base_pt[0]*np.sin(ang_per_seg*i) + np.cos(ang_per_seg*i)*base_pt[2]
    # translatae to same as stl
    cl[:,0] = cl[:,0]+100
    slicer.cl = cl
    slicer.centerline_slice(cl)
    slicer.vis_curvesliced(dir_vec = True)

