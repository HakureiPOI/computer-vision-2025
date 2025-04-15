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

    N, D = features.shape
    assert N >= k, 'Number of clusters cannot be greater than number of points'
    
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for _ in range(num_iters):
        dists = cdist(features, centers)
        new_assignments = np.argmin(dists, axis=1)
        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments
        for i in range(k):
            if np.any(assignments == i):
                centers[i] = features[assignments == i].mean(axis=0)

    return assignments

### Clustering Methods for colorful image
def kmeans_color(img, k, num_iters=500):
    H, W, C = img.shape
    features = img.reshape(-1, C).astype(np.float32)
    assignments = kmeans(features, k, num_iters)
    return assignments.reshape(H, W)


# 找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01
    data_point = data[:, idx].reshape(3, 1)
    while True:
        dists = np.linalg.norm(data - data_point, axis=0)
        neighbors = data[:, dists < r]
        new_point = np.mean(neighbors, axis=1).reshape(3, 1)
        shift = np.linalg.norm(new_point - data_point)
        if shift < t:
            break
        data_point = new_point
    return data_point



# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = []
    label_no = 1
    peak = findpeak(data, 0, r)
    peaks.append(peak.flatten())
    labels[0] = label_no

    for idx in range(1, len(data.T)):
        current_peak = findpeak(data, idx, r)
        found = False
        for i, p in enumerate(peaks):
            if np.linalg.norm(current_peak.flatten() - p) < r / 2:
                labels[idx] = i + 1
                found = True
                break
        if not found:
            peaks.append(current_peak.flatten())
            label_no += 1
            labels[idx] = label_no
    return labels, np.array(peaks).T



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
    
    accuracy = np.mean(mask_gt == mask)
    return accuracy
