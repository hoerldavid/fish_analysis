import numpy as np
import matplotlib.pyplot as plt

class nd_gauss():
    def __init__(self, d):
        self.d = d

    def eval(self, x, *params):
        b = params[0]
        x0 = params[1]
        A = params[2]

        exponent = (-1/2) * (x - x0) @ (np.linalg.inv(A)) @ (x - x0).transpose()
        return b * np.exp(exponent)


def main():
    arr = np.zeros([30,30])

    b = 1.0
    x0 = np.array([15, 15], dtype=np.float)
    x1 = np.array([14,14])
    x0 = np.transpose(x0)
    A = np.array([5, 0, 0, 5],dtype=np.float).reshape([2, 2])

    it = np.nditer(arr, flags=['f_index'], op_flags=['readwrite'])
    while not it.finished:
        x = np.unravel_index(it.index, arr.shape)

        arr[x] = nd_gauss(2).eval(np.array(x, dtype=np.float).transpose(), b, x0, A)
        print(nd_gauss(2).eval(np.array(x).transpose(), b, x0, A))
        it.iternext()

    print(A)
    print(x0)
    print(-x1 @ A @ x1 + (x0 @ x1))

    plt.imshow(arr)
    plt.show()


if __name__ == '__main__':
    main()