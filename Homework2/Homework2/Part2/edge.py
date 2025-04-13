import numpy as np

def conv(image, kernel):
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    pad_h, pad_w = Hk // 2, Wk // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    for i in range(Hi):
        for j in range(Wi):
            region = padded[i:i + Hk, j:j + Wk]
            out[i, j] = np.sum(region * kernel)
    return out

def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= (2 * np.pi * sigma**2)
    # kernel /= np.sum(kernel) 给的对比例没有归一化
    return kernel

def partial_x(img):
    kernel = np.array([[-0.5, 0, 0.5]])
    return conv(img, kernel)

def partial_y(img):
    kernel = np.array([[-0.5], [0], [0.5]])
    return conv(img, kernel)

def gradient(img):
    dx = partial_x(img)
    dy = partial_y(img)
    G = np.sqrt(dx**2 + dy**2)
    theta = (np.arctan2(dy, dx) * 180 / np.pi) % 360
    return G, theta

def non_maximum_suppression(G, theta):
    H, W = G.shape
    out = np.zeros((H, W))
    angle = theta % 180

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            direction = angle[y, x]
            q = r = 0

            if (0 <= direction < 22.5) or (157.5 <= direction < 180):
                q = G[y, x + 1]
                r = G[y, x - 1]
            elif 22.5 <= direction < 67.5:
                q = G[y - 1, x + 1]
                r = G[y + 1, x - 1]
            elif 67.5 <= direction < 112.5:
                q = G[y - 1, x]
                r = G[y + 1, x]
            elif 112.5 <= direction < 157.5:
                q = G[y - 1, x - 1]
                r = G[y + 1, x + 1]

            if G[y, x] >= q and G[y, x] >= r:
                out[y, x] = G[y, x]
    return out

def double_thresholding(img, high, low):
    strong = img > high
    weak = (img >= low) & (img <= high)
    return strong, weak

def get_neighbors(y, x, H, W):
    neighbors = []
    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            if (0 <= i < H) and (0 <= j < W) and (i != y or j != x):
                neighbors.append((i, j))
    return neighbors

def link_edges(strong_edges, weak_edges):
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
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong, weak = double_thresholding(nms, high, low)
    edge = link_edges(strong, weak)
    return edge
