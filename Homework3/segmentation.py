import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color

### Clustering Methods for 1-D points
def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    print(features.shape)
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        # scipy.spatial.distance.cdist 可以计算两个集合中所有向量之间的成对距离
        dists = cdist(features, centers)  # shape : (N, k)
        new_assignments = np.argmin(dists, axis=1)

        # 收敛了就提前终止
        if np.all(assignments == new_assignments):
            print(f'Converged after {n} iterations')
            break
        assignments = new_assignments

        # 重新计算每个簇的中心
        for i in range(k):
            points_in_cluster = features[assignments == i]
            if len(points_in_cluster) > 0:
                centers[i] = np.mean(points_in_cluster, axis=0)

    return assignments

### Clustering Methods for colorful image
def kmeans_color(img, k, num_iters=500):
    H, W, C = img.shape
    features = img.reshape(-1, C).astype(np.float32)
    assignments = kmeans(features, k, num_iters)
    return assignments.reshape(H, W)


# 找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01  # 收敛阈值
    shift = np.array([1])  # 初始位移值
    data_point = data[:, idx]  # 当前点
    dataT = data.T  # 所有点 shape: (N, D)
    data_pointT = data_point.T.reshape(1, -1)  # 当前点 shape: (1, D)

    while np.linalg.norm(shift) > t:
        # 计算当前点与所有点之间的欧几里得距离
        dists = np.linalg.norm(dataT - data_pointT, axis=1)

        # 找出在半径 r 内的点
        in_radius = dataT[dists < r]

        if len(in_radius) == 0:
            break

        # 计算这些点的均值（Mean Shift vector）
        new_point = np.mean(in_radius, axis=0, keepdims=True)

        # 更新 shift 并移动当前点
        shift = new_point - data_pointT
        data_pointT = new_point

    return data_pointT.T 



# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    # labels = np.zeros(len(data.T), dtype=np.int32)
    # peaks = []  # 已识别的聚类中心（peaks）
    # label_no = 1  # 当前聚类编号

    # # 处理第一个点
    # peak = findpeak(data, 0, r)
    # peakT = peak.T  # shape: (1, D)
    # peaks.append(peakT)
    # labels[0] = label_no

    # # 遍历每个点，寻找其聚类中心
    # for idx in range(1, len(data.T)):
    #     peak = findpeak(data, idx, r)
    #     peakT = peak.T  # shape: (1, D)

    #     assigned = False
    #     # 遍历已有聚类中心，看是否可以归入某类
    #     for i, existing_peak in enumerate(peaks):
    #         if np.linalg.norm(peakT - existing_peak) < r / 2:
    #             labels[idx] = i + 1
    #             assigned = True
    #             break

    #     # 如果未归入任何已有聚类，则新建聚类
    #     if not assigned:
    #         peaks.append(peakT)
    #         label_no += 1
    #         labels[idx] = label_no

    from sklearn.cluster import MeanShift

    # sklearn 的 MeanShift 期望输入为 (N, D)，所以需要转置
    data_T = data.T

    ms = MeanShift(bandwidth=r, bin_seeding=True)
    ms.fit(data_T)

    labels = ms.labels_
    peaks = ms.cluster_centers_.T  # 转回 (D, num_clusters)

    return labels, peaks

# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))


    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    res_img=color.lab2rgb(segmented_image)
    res_img=color.rgb2gray(res_img)
    return res_img


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    assert mask_gt.shape == mask.shape, "Masks must have the same shape"
    
    correct = (mask_gt == mask).sum()
    total = mask_gt.size
    acc = correct / total

    return acc

