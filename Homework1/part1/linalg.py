import numpy as np


def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    When inputs are 2-D array, `np.matmul()` and `np.dot()` have same result, 
    you can also use `np.matmul()`.

    notice that `np.dot()` and `np.matmul()` need `a` with shape (x, n), `b` with shape `(n, x)
    so you need to transpose `a`, you can use syntax `a.T`.


    Args:
        a: numpy array of shape (n, x)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """

    # Input validation
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")

    # Calculate dot product
    out = np.dot(a.T, b)
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a^Tb) x (Ma), `a^T` is transpose of `a`, 
    (a^Tb) is matrix multiplication of a^T and b,
    (Ma) is matrix multiplication of M and a.

    You can use `np.matmul()` to do matrix multiplication.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (n, 1).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """

    # Input validation 
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional.")
    if a.shape[1] != 1 or b.shape[1] != 1:
        raise ValueError("Input arrays must have shape (n, 1).")
    if M.shape[1] != a.shape[0]:
        raise ValueError("Input arrays must have compatible shapes.")

    # Calculate (a^Tb) x (Ma)
    out = np.dot(a.T, b) * np.dot(M, a)
    return out


def eigen_decomp(M):
    """Implement eigenvalue decomposition.

    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, m)

    Returns:
        w: numpy array of shape (m,) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """

    # Input validation 
    if M.shape[0] != M.shape[1]:
        raise ValueError("Input matrix must be square.")
    if M.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    # Calculate eigenvalue decomposition
    w, v = np.linalg.eig(M)
    return w, v


def euclidean_distance_native(u, v):
    """Computes the Euclidean distance between two vectors, represented as Python
    lists.

    Args:
        u (List[float]): A vector, represented as a list of floats.
        v (List[float]): A vector, represented as a list of floats.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    # assert isinstance(u, list)
    # assert isinstance(v, list)
    # assert len(u) == len(v)

    if not isinstance(u, list):
        raise ValueError("Input u must be a list.")
    if not isinstance(v, list):
        raise ValueError("Input v must be a list.")
    if len(u) != len(v):
        raise ValueError("Input lists must have the same length.")

    # Compute the distance!
    # Notes:
    #  1) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    # distance = np.sum(np.square(np.array(u) - np.array(v)))
    # distance = np.sqrt(distance)

    distance = np.linalg.norm(np.array(u) - np.array(v))
    return distance


def euclidean_distance_numpy(u, v):
    """Computes the Euclidean distance between two vectors, represented as NumPy
    arrays.

    Args:
        u (np.ndarray): A vector, represented as a NumPy array.
        v (np.ndarray): A vector, represented as a NumPy array.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    # assert isinstance(u, np.ndarray)
    # assert isinstance(v, np.ndarray)
    # assert u.shape == v.shape

    if not isinstance(u, np.ndarray):
        raise ValueError("Input u must be a NumPy array.")
    if not isinstance(v, np.ndarray):
        raise ValueError("Input v must be a NumPy array.")
    if u.shape != v.shape:
        raise ValueError("Input arrays must have the same shape.")
        
    # if u.ndim != 1 or v.ndim != 1:
    #     raise ValueError("Input arrays must be 1-dimensional.")

    # Compute the distance!
    # Note:
    #  1) You shouldn't need any loops
    #  2) Some functions you can Google that might be useful:
    #         np.sqrt(), np.sum()
    #  3) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    return np.linalg.norm(u - v)


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """

    # Input validation 
    if M.shape[0] != M.shape[1]:
        raise ValueError("Input matrix must be square.")
    if M.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    eigenvalues_all, eigenvectors_all = eigen_decomp(M)

    indices = np.argsort(np.abs(eigenvalues_all))[::-1] 
    top_indices = indices[:k]

    eigenvalues = [eigenvalues_all[i] for i in top_indices]
    eigenvectors = [eigenvectors_all[:, i] for i in top_indices]

    return eigenvalues, eigenvectors
