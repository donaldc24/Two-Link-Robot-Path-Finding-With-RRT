import math
import numpy as np


def get_nearest_point_on_line(s1, s2, p, tol=1e-3):
    """Compute the nearest point on a line described by s1 and s2 to p

    Note that a point on the line can be parametrized by
        s(t) = s1 + t(s2 - s1).
    s(t) is on the line segment between s1 and s2 iff t \in [0, 1].

    The point on the line closest to p is s(t*) where
        t* = <p-s1, s2-s1> / ||s2 - s1||^2

    @return (s*, t*) where s* = s(t*)
    """
    ls = s2 - s1  # The line segment from s1 to s2
    len_ls2 = np.dot(ls, ls)  # the squared length of ls

    # If the line segment is too short, just return 0
    if len_ls2 < tol:
        return (s1, 0)

    tstar = np.dot(p - s1, ls) / len_ls2
    if tstar <= tol:
        return (s1, 0)
    if tstar >= 1 - tol:
        return (s2, 1)

    return (s1 + tstar * ls, tstar)


def get_euclidean_distance(s1, s2):
    """Compute the norm ||s2 - s1||"""
    ls = s2 - s1
    return math.sqrt(np.dot(ls, ls))


def is_inside_circle(c, r, p):
    """Return whether point p is inside a circle with radius r, centered at c"""
    return (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 <= r ** 2
