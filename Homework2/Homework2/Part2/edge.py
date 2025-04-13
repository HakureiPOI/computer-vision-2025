import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    for i in range(Hi):
        for j in range(Wi):
            region = padded[i:i + Hk, j:j + Wk]
            out[i, j] = np.sum(region * kernel)
    return out



def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= (2 * np.pi * sigma**2)
    # kernel /= np.sum(kernel)  # 归一化
    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    # sobel_x = np.array([[-1, 0, 1],
    #                     [-2, 0, 2],
    #                     [-1, 0, 1]])

    # 没看题，懒得换变量名了
    sobel_x = np.array([[0.5, 0, -0.5]])
    sobel_x = np.flip(sobel_x)
    out = conv(img, sobel_x)

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    # sobel_y = np.array([[-1, -2, -1],
    #                     [ 0,  0,  0],
    #                     [ 1,  2,  1]])
    
    sobel_y = np.array([[0.5], [0], [-0.5]])
    sobel_y = np.flip(sobel_y)
    out = conv(img, sobel_y)

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """

    dx = partial_x(img)
    dy = partial_y(img)
    G = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx) * 180 / np.pi) % 360  # 角度制 0~360
    return G, theta

def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    angle = np.rad2deg(theta) % 180 # 转换成 [0, 180) 的角度范围

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            direction = angle[y, x]

            # 初始化邻居值
            neighbor1 = neighbor2 = 0

            # 四舍五入方向角度
            if (0 <= direction < 22.5) or (157.5 <= direction < 180):
                neighbor1 = G[y, x - 1]
                neighbor2 = G[y, x + 1]
            elif 22.5 <= direction < 67.5:
                neighbor1 = G[y - 1, x + 1]
                neighbor2 = G[y + 1, x - 1]
            elif 67.5 <= direction < 112.5:
                neighbor1 = G[y - 1, x]
                neighbor2 = G[y + 1, x]
            elif 112.5 <= direction < 157.5:
                neighbor1 = G[y - 1, x - 1]
                neighbor2 = G[y + 1, x + 1]

            # 非极大值抑制
            if G[y, x] >= neighbor1 and G[y, x] >= neighbor2:
                out[y, x] = G[y, x]
            else:
                out[y, x] = 0

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    # 双阈值限制
    strong_edges = img > high
    weak_edges = (img >= low) & (img <= high)
    return strong_edges, weak_edges



def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    from collections import deque
    H, W = strong_edges.shape
    visited = np.copy(strong_edges)
    edges = np.copy(strong_edges)

    queue = deque(np.argwhere(strong_edges))

    while queue:
        y, x = queue.popleft()
        for i, j in get_neighbors(y, x, H, W):
            if weak_edges[i, j] and not visited[i, j]:
                visited[i, j] = True
                edges[i, j] = True
                queue.append((i, j))

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """

    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)

    G, theta = gradient(smoothed)

    nms = non_maximum_suppression(G, theta)

    strong, weak = double_thresholding(nms, high, low)

    edge = link_edges(strong, weak)

    return edge