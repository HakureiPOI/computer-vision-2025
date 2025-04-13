import numpy as np

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    kh, kw = Hk // 2, Wk // 2
    kernel_flipped = np.flip(kernel)  

    for m in range(Hi):
        for n in range(Wi):
            val = 0
            for i in range(Hk):
                for j in range(Wk):
                    ii = m - kh + i
                    jj = n - kw + j
                    if 0 <= ii < Hi and 0 <= jj < Wi:
                        val += image[ii, jj] * kernel_flipped[i, j]
            out[m, n] = val

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    return out

def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    kh, kw = Hk // 2, Wk // 2
    kernel_flipped = np.flip(kernel)  
    image_padded = zero_pad(image, kh, kw)
    out = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            window = image_padded[m:m + Hk, n:n + Wk]
            out[m, n] = np.sum(window * kernel_flipped)

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    # 交叉相关不翻转模板
    out = conv_fast(f, g)
    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    g_mean = np.mean(g)
    g_zero_mean = g - g_mean
    out = cross_correlation(f, g_zero_mean)
    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    gh, gw = Hg // 2, Wg // 2
    out = np.zeros((Hf, Wf))

    g_mean = np.mean(g)
    g_std = np.std(g)
    g_norm = (g - g_mean) / (g_std + 1e-8)

    f_padded = zero_pad(f, gh, gw)

    for i in range(Hf):
        for j in range(Wf):
            window = f_padded[i:i + Hg, j:j + Wg]
            w_mean = np.mean(window)
            w_std = np.std(window)
            if w_std > 1e-8:
                window_norm = (window - w_mean) / w_std
                out[i, j] = np.sum(window_norm * g_norm)
            else:
                out[i, j] = 0  # 标准差过小，视为无信息

    return out