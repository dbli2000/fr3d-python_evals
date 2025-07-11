import numpy


def RMSD(set1, set2):
    """This function calculates the root mean square difference of the
    atomic positions of set1 with set2.  In other words, this function
    calculates the root mean squared differences between two sets of a
    n 3-dimensional coordinates.  The calculation is explained here:
    wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions
    """
    assert len(set1) == len(set2)
    L = len(set1)
    assert L > 0
    distance = numpy.sqrt(numpy.sum(numpy.power(set1 - set2, 2))/L)
    return distance


def sumsquarederror(set1, set2, weights=None):
    """This function calculates the summed squared difference of the
    atomic positions of set1 with set2.  In other words, this function
    calculates the root mean squared differences between two sets of a
    n 3-dimensional coordinates.  The calculation is explained here:
    wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions
    """
    assert len(set1) == len(set2)
    L = len(set1)
    assert L > 0

    squared_differences = numpy.power(set1 - set2, 2) # Shape (L, 3) for coordinates

    if weights is None:
        # Unweighted sum of squared errors for each coordinate, then sum over all coordinates
        return numpy.sum(squared_differences)
    else:
        weights = numpy.asarray(weights) # Ensure weights is a numpy array
        if weights.ndim == 1:
            assert len(weights) == L, "Length of 1D weights must match the number of points."
            # Sum of squared differences for each point: (dx_i^2 + dy_i^2 + dz_i^2)
            sum_sq_diff_per_point = numpy.sum(squared_differences, axis=1) # Shape (L,)
            # Apply weights to the sum of squared differences for each point
            weighted_sum_sq_diff = numpy.sum(weights * sum_sq_diff_per_point)
        elif weights.ndim == 2:
            assert weights.shape == (L, 3), "Shape of 2D weights must match set1 and set2."
            # Apply weights component-wise
            weighted_sum_sq_diff = numpy.sum(weights * squared_differences)
        else:
            raise ValueError("Weights must be None, a 1D array (per-point), or a 2D array (per-coordinate).")

        return weighted_sum_sq_diff
