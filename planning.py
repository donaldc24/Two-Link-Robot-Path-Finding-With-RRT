import random
import numpy as np
from graph import Tree
from edge import EdgeStraight
from geometry import get_euclidean_distance
import math
from shapely.geometry import Polygon


##############################################################################
# Classes for creating an edge
##############################################################################
class EdgeCreator:
    def make_edge(self, s1, s2):
        """Return an Edge object beginning at state s1 and ending at state s2"""
        raise NotImplementedError


class StraightEdgeCreator(EdgeCreator):
    def __init__(self, step_size):
        self.step_size = step_size

    def make_edge(self, s1, s2):
        return EdgeStraight(s1, s2, self.step_size)


##############################################################################
# Classes for computing distance between 2 points
##############################################################################
class DistanceComputator:
    def get_distance(self, s1, s2):
        """Return the distance between s1 and s2"""
        raise NotImplementedError


class EuclideanDistanceComputator(DistanceComputator):
    def get_distance(self, s1, s2):
        """Return the Euclidean distance between s1 and s2"""
        return get_euclidean_distance(s1, s2)


##############################################################################
# Planning algorithms
##############################################################################
def rrt(
    cspace,
    qI,
    qG,
    edge_creator,
    distance_computator,
    O,
    W,
    L,
    D,
    # collision_checker,
    pG=0.1,
    numIt=500,
    tol=1e-3,
):
    """RRT with obstacles

    @type cspace: a list of tuples (smin, smax) indicating that the C-space
        is given by the product of the tuples.
    @type qI: a tuple (x, y) indicating the initial configuration.
    @type qG: a typle (x, y) indicating the goal configuation
        (can be None if rrt is only used to explore the C-space).
    @type edge_creator: an EdgeCreator object that includes the make_edge(s1, s2) function,
        which returns an Edge object beginning at state s1 and ending at state s2.
    @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
        function, which returns the distance between s1 and s2.
    @type collision_checker: a CollisionChecker object that includes the is_in_collision(s)
        function, which returns whether the state s is in collision.
    @type pG: a float indicating the probability of choosing the goal configuration.
    @type numIt: an integer indicating the maximum number of iterations.
    @type tol: a float, indicating the tolerance on the euclidean distance when checking whether
        2 states are the same

    @return (G, root, goal) where G is the tree, root is the id of the root vertex
        and goal is the id of the goal vertex (if one exists in the tree; otherwise goal will be None).
    """
    G = Tree()
    root = G.add_vertex(np.array(qI))
    for i in range(numIt):
        use_goal = qG is not None and random.uniform(0, 1) <= pG
        if use_goal:
            alpha = np.array(qG)
        else:
            alpha = sample(cspace)
        vn = G.get_nearest(alpha, distance_computator, tol)
        qn = G.get_vertex_state(vn)
        (qs, edge) = stopping_configuration(
            qn, alpha, edge_creator, O, W, L, D, tol
        )
        if qs is None or edge is None:
            continue
        dist = get_euclidean_distance(qn, qs)
        if dist > tol:
            vs = G.add_vertex(qs)
            G.add_edge(vn, vs, edge)
            if use_goal and get_euclidean_distance(qs, qG) < tol:
                return (G, root, vs)

    return (G, root, None)

def sample(cspace):
    """Return a sample configuration of the C-space based on uniform random sampling"""
    sample = [random.uniform(cspace_comp[0], cspace_comp[1]) for cspace_comp in cspace]
    return np.array(sample)

def get_link_positions(config, W, L, D):
    """Compute the positions of the links and the joints of a 2D kinematic chain A_1, ..., A_m

    @type config: a list [theta_1, ..., theta_m] where theta_1 represents the angle between A_1 and the x-axis,
        and for each i such that 1 < i <= m, \theta_i represents the angle between A_i and A_{i-1}.
    @type W: float, representing the width of each link
    @type L: float, representing the length of each link
    @type D: float, the distance between the two points of attachment on each link

    @return: a tuple (joint_positions, link_vertices) where
        * joint_positions is a list [p_1, ..., p_{m+1}] where p_i is the position [x,y] of the joint between A_i and A_{i-1}
        * link_vertices is a list [V_1, ..., V_m] where V_i is the list of [x,y] positions of vertices of A_i
    """

    if len(config) == 0:
        return ([], [])

    joint_positions = [np.array([0, 0, 1])]
    link_vertices = []

    link_vertices_body = [
        np.array([-(L - D) / 2, -W / 2, 1]),
        np.array([D + (L - D) / 2, -W / 2, 1]),
        np.array([D + (L - D) / 2, W / 2, 1]),
        np.array([-(L - D) / 2, W / 2, 1]),
    ]
    joint_body = np.array([D, 0, 1])
    trans_mat = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    for i in range(len(config)):
        a = D if i > 0 else 0
        trans_mat = np.matmul(trans_mat, get_trans_mat(config[i], a))
        joint = np.matmul(trans_mat, joint_body)
        vertices = [
            np.matmul(trans_mat, link_vertex) for link_vertex in link_vertices_body
        ]
        joint_positions.append(joint)
        link_vertices.append(vertices)

    return (joint_positions, link_vertices)

def get_trans_mat(theta, a):
    """Return the homogeneous transformation matrix"""
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), a],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
    )

def is_in_collision(vex, O, W, L, D):
    #Gets the robots configuration
    # initRobotPos = list()
    # initRobotPos.append((vex[0] * math.pi) / 180)
    # initRobotPos.append((vex[1] * math.pi) / 180)
    (joint_positions, link_vertices) = get_link_positions(vex, W, L, D)

    # Makes polygons for the robots 2 links
    p1Vert = list()
    for i in link_vertices[0]:
        p1coords = (i[0], i[1])
        p1Vert.append(p1coords)
    p1 = Polygon([p1Vert[0], p1Vert[1], p1Vert[2], p1Vert[3]])
    p2Vert = list()
    for i in link_vertices[1]:
        p2coords = (i[0], i[1])
        p2Vert.append(p2coords)
    p2 = Polygon([p2Vert[0], p2Vert[1], p2Vert[2], p2Vert[3]])

    # Creates polygons for each obstacle and checks for collision with robot
    for i in O:
        vertices = list()
        for j in i:
            coords = (j[0], j[1])
            vertices.append(coords)
        p3 = Polygon([vertices[0], vertices[1], vertices[2], vertices[3]])
        if p1.intersects(p3) or p2.intersects(p3):
            return True
    return False

def stopping_configuration(s1, s2, edge_creator, O, W, L, D, tol):
    """Return (s, edge) where s is the point along the edge from s1 to s2 that is closest to s2 and
    is not in collision with the obstacles and edge is the edge from s to s1"""

    edge = edge_creator.make_edge(s1, s2)

    if edge.get_length() < tol:
        return (s1, edge)

    curr_ind = 0
    prev_state = None
    curr_state = edge.get_discretized_state(curr_ind)

    while curr_state is not None:
        if is_in_collision(curr_state, O, W, L, D):
            if curr_ind == 0:
                return (None, None)
            elif curr_ind == 1:
                return (s1, None)
            split_t = (curr_ind - 1) * edge.get_step_size() / edge.get_length()
            (edge1, _) = edge.split(split_t)
            return (prev_state, edge1)
        curr_ind = curr_ind + 1
        prev_state = curr_state
        curr_state = edge.get_discretized_state(curr_ind)

    return (s2, edge)