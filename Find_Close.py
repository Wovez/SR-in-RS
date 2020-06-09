"""
求二维数组中与(Lon, Lat)最接近的数
"""

import numpy as np


def find_close(arr1, arr2, Lon, Lat):
    shape = arr1.shape
    f_lon = arr1 - Lon
    f_lon_abs = abs(f_lon)
    f_lat = arr2 - Lat
    f_lat_abs = abs(f_lat)
    f = f_lon_abs + f_lat_abs
    temp = f.argmin()
    row = temp // shape[1]
    col = temp - row * shape[1]
    idx = [row, col]
    return idx


if __name__ == '__main__':
    a1 = np.mat([[1, 2, 3, 4], [5, 6, 7, 8, 9]])
    a2 = np.mat([[8, 7, 6, 5], [4, 3, 2, 1]])
    print(find_close(a1, a2, 3, 4))
