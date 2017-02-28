from PIL import Image
import numpy as np


def get_interval(x1, x2, steps):
    intervals = []
    dx = x2 - x1
    intervals.append(x1)
    b = steps - 1
    i = 1
    while i <= b:
        a = x1 + (i / b) * dx
        intervals.append(a)
        i += 1
    return intervals


def read_image_stack(path):
    """
    :param path: path to .tif file
    :return: stack of layers
    """
    image_file = Image.open(path)
    image_list = list()
    n = 0
    while True:
        (w, h) = image_file.size
        image_list.append(np.array(image_file.getdata()).reshape(h, w))
        n += 1
        try:
            image_file.seek(n)
        except:
            break

    return np.dstack(image_list)


def distance_mat(pks1, pks2, links_to_ignore=[], pixelsize=np.array([1,1,1], dtype=float)):
    res = np.zeros((max(pks1.shape[0], pks2.shape[0]), max(pks1.shape[0], pks2.shape[0])))
    for i in range(pks1.shape[0]):
        for j in range(pks2.shape[0]):
            if [i,j] in links_to_ignore or [j,i] in links_to_ignore:
                res[i, j] = np.inf
            else:
                res[i, j] = np.sqrt(np.sum((pks1[i] * pixelsize - pks2[j] * pixelsize) ** 2))
    return res


def main():
    print(np.array([[0, 0, 0], [1, 1, 1]]))
    print(distance_mat(np.array([[0, 0, 0]]), np.array([[1, 1, 1], [1, 1, 1]])))


if __name__ == '__main__':
    main()
