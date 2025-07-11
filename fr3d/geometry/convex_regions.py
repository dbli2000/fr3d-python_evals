import numpy as np


def totheleft(a, b):
    """Determines if vector `a` is to the left of vector `b`.

    Both `a` and `b` are 2D vectors assumed to originate from the same point.
    The "left" side is determined by considering the vectors in a 2D plane
    and using the sign of the z-component of the cross product `b x a`.
    A non-negative z-component means `a` is to the left of or collinear with `b`.

    Parameters
    ----------
    a : numpy.ndarray
        A 2D vector (e.g., [x, y]).
    b : numpy.ndarray
        A 2D vector (e.g., [x, y]).

    Returns
    -------
    bool
        True if vector `a` is to the left of or collinear with vector `b`, False otherwise.
    """

    a = np.append(a, 0)
    b = np.append(b, 0)
    cross = np.cross(b, a)
    return cross[2] >= 0


def ptinlefthalf(P1, P2, P3):
    """Checks if point P3 lies in the left half-plane defined by the directed line segment from P1 to P2.

    Parameters
    ----------
    P1 : array_like
        Coordinates of the first point defining the directed edge (e.g., [x1, y1]).
    P2 : array_like
        Coordinates of the second point defining the directed edge (e.g., [x2, y2]).
    P3 : array_like
        Coordinates of the point to test (e.g., [x3, y3]).

    Returns
    -------
    bool
        True if P3 is to the left of or on the directed line P1->P2, False otherwise.
    """
    V1 = np.array([P2[0]-P1[0], P2[1]-P1[1]])  # Vector from P1 to P2
    V2 = np.array([P3[0]-P1[0], P3[1]-P1[1]])  # Vector from P1 to P3
    return totheleft(V2, V1)


def testcounterclockwiseconvex(P):
    """Suppose that you have a sequence of points in the plane,
    P = p1, p2, p3, ... , pN.  These are supposed to go counterclockwise and
    each new point should be to the left of the line defined by the previous
    two.
    I don't want to assume that pN = p1, so the shape will be closed by
    following p1, p2, p3, ... , pN, p1.  We need a test to make sure that each
    point lies to the left of the line defined by the previous two.
    It's important to check all the way to the line defined by pN and p1 and
    make sure that p2 is to the left of that.

    Parameters
    ----------
    P : list of array_like
        A list of 2D points (e.g., [[x1, y1], [x2, y2], ...]).

    Returns
    -------
    bool
        True if the polygon defined by P is convex and its vertices are in
        counterclockwise order, False otherwise.
    """
    l = len(P)
    if l < 3: # A polygon must have at least 3 vertices
        return False

    for i in range(l):
        # Define points p_i, p_{i+1}, p_{i+2} with wrap-around using modulo
        p_prev = P[i]
        p_curr = P[(i + 1) % l]
        p_next = P[(i + 2) % l]

        # Check if p_next is to the left of the directed edge p_prev -> p_curr
        if not ptinlefthalf(p_prev, p_curr, p_next):
            return False

    return True


def counterclockwiseinside(polygon_points, test_point):
    """Given a sequence of 2D points `polygon_points` that form a convex polygon
    ordered counterclockwise, and a 2D `test_point`, this function returns True
    if the `test_point` is inside the polygon, and False otherwise.

    The polygon is closed by connecting the last point back to the first.
    Assumes `polygon_points` has at least 3 points and forms a convex polygon
    with counterclockwise winding. For robustness, one might call
    `testcounterclockwiseconvex` on `polygon_points` before using this function.

    Parameters
    ----------
    polygon_points : list of array_like
        A list of 2D points defining the vertices of the convex polygon in
        counterclockwise order.
    test_point : array_like
        The 2D point to test for inclusion.

    Returns
    -------
    bool
        True if `test_point` is inside `polygon_points`, False otherwise.
    """
    num_points = len(polygon_points)
    if num_points < 3:
        # Not a polygon, or degenerate. Behavior might need to be defined.
        # For now, consider point outside for non-polygons.
        return False

    for i in range(num_points):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % num_points]  # Next point, wraps around for the last edge

        # If test_point is not to the left of the edge (p1, p2), it's outside.
        if not ptinlefthalf(p1, p2, test_point):
            return False

    # If the point is to the left of all edges, it's inside.
    return True
