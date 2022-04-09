import sys, argparse
import matplotlib.pyplot as plt
from planning import (
    rrt,
    StraightEdgeCreator,
    EuclideanDistanceComputator
)
from draw_cspace import draw
import json, sys, argparse, math

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="C-Space and C-Space Obstacles:")
    parser.add_argument(
        "desc",
        metavar="problem_description_path",
        type=str,
        help="path to the problem description file containing the obstacles, width of link, length of link, distance between, the initial cell, and the goal region",
    )
    parser.add_argument(
        "--out",
        metavar="output_path",
        type=str,
        required=False,
        default="",
        dest="out",
        help="path to the output file",
    )

    args = parser.parse_args(sys.argv[1:])

    return args

def parse_desc(desc):
    """Parse problem description json file to get the problem description"""
    with open(desc) as desc:
        data = json.load(desc)

    O = data["O"]
    W = data["W"]
    L = data["L"]

    D = data["D"]
    xI = tuple(data["xI"])
    XG = tuple(data["xG"])
    U = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    return (O, U, W, L, D, xI, XG)

def main_rrt(
    cspace, qI, qG, edge_creator, distance_computator, O, W, L, D
):
    """Task 1 (Exploring the C-space using RRT) and Task 2 (Solve the planning problem using RRT)"""
    fig, ax3 = plt.subplots()

    # Task 2: Include obstacles and goal
    title3 = "RRT planning"
    (G3, root3, goal3) = rrt(
        cspace=cspace,
        qI=qI,
        qG=qG,
        edge_creator=edge_creator,
        distance_computator=distance_computator,
        O=O,
        W=W,
        L=L,
        D=D,
        # collision_checker=collision_checker,
    )
    path = []
    if goal3 is not None:
        path = G3.get_vertex_path(root3, goal3)
    return path, G3
    # draw(ax3, cspace, [], qI, qG, G3, path, title3)

    # plt.show()

if __name__ == "__main__":
    args = parse_args()
    (O, U, W, L, D, qI, qG) = parse_desc(args.desc)
    cspace = [(0, 2*math.pi), (0, 2*math.pi)]

    edge_creator = StraightEdgeCreator(0.1)
    distance_computator = EuclideanDistanceComputator()

    path, G = main_rrt(
        cspace,
        qI,
        qG,
        edge_creator,
        distance_computator,
        O,
        W,
        L,
        D,
    )
    vertices = G.get_vertices()
    allVert = []
    edges = []
    for v in vertices:
        allVert.append({"id": v, "config": G.get_vertex_state(v).tolist()})
    for k in dict.keys(G.edges):
        edges.append(list(k))

    result = {"vertices": allVert, "edges": edges, "path": path}
    with open(args.out, "w") as outfile:
        json.dump(result, outfile)
