import numpy as np

def get_rgb_projected(img_r, img_g=None, img_b=None, axis=2, fun=np.max):
    img_r = img_r / np.max(img_r)

    if img_g is None:
        img_g = np.zeros(img_r.shape)
    else:
        img_g = img_g / np.max(img_g)

    if img_b is None:
        img_b = np.zeros(img_r.shape)
    else:
        img_b = img_b / np.max(img_b)

    return np.apply_along_axis(fun, axis, np.stack((img_r, img_g, img_b), 3))