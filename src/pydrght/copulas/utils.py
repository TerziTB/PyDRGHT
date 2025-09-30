import numpy as np

def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.

    Parameters
    ----------
    matrix : array-like, shape (n, n)
        Symmetric square matrix to check.

    Returns
    -------
    bool
        True if the matrix is positive definite, False otherwise.

    Notes
    -----
    Uses Cholesky decomposition to test positive definiteness.
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def nearest_positive_definite(matrix):
    """
    Find the nearest positive definite matrix to the input matrix.

    Parameters
    ----------
    matrix : array-like, shape (n, n)
        Symmetric square matrix which may not be positive definite.

    Returns
    -------
    np.ndarray, shape (n, n)
        Positive definite matrix closest to the input matrix in Frobenius norm.

    Notes
    -----
    Algorithm:
    1. Compute the symmetric part of the matrix.
    2. Use singular value decomposition (SVD) to construct a positive semidefinite matrix.
    3. Iteratively adjust eigenvalues if the resulting matrix is not positive definite.

    References
    ----------
    - Higham, N. J. (1988). 
    *Computing a nearest symmetric positive semidefinite matrix*. 
    Linear Algebra and its Applications, 103, 103–118.
    """
    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    spacing = np.spacing(np.linalg.norm(matrix))
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(A3):
        min_eig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-min_eig * k**2 + spacing)
        k += 1
    return A3

def bracket_1d(nll_fun, near_bnd: float, far_start: float) -> tuple[float, float]:
    """
    Bracket the minimizer of a one-parameter negative log-likelihood function.

    Parameters:
    - nll_fun: Function for computing negative log-likelihood.
    - near_bnd: A point known to be a lower/upper bound for the minimizer.
    - far_start: The first trial point to test to see if it's an upper/lower bound.

    Returns:
    - near_bnd: Updated lower/upper bound for the minimizer.
    - far_bnd: Desired upper/lower bound for the minimizer.
    """
    bound = far_start
    upper_lim = 1e12  # arbitrary finite limit for search
    old_nll = nll_fun(bound)
    old_bound = bound

    while abs(bound) <= upper_lim:
        bound = 2 * bound  # assumes lower start is < 0, upper is > 0
        nll = nll_fun(bound)

        if nll > old_nll:
            # The neg loglikelihood increased, we're on the far side of the minimum,
            # so the current point is the desired far bound.
            far_bnd = bound
            break
        else:
            # The neg loglikelihood continued to decrease, so the previous point
            # is on the near side of the minimum, update the near bound.
            near_bnd = old_bound

        old_nll = nll
        old_bound = bound

    if abs(bound) > upper_lim:
        far_bnd = float('nan')

    return near_bnd, far_bnd
