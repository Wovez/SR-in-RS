"""
MERSI影像重采样
（可去除Bowtie/双眼皮效应，可适用于任何包含经纬度数据的卫星）

<1>输入矩阵大小为k×m×n，其中k为输入通道个数，且倒数第二个通道为
Latitude，倒数第一个通道为Lontitude，前面每个通道对应一组数据，每
个通道数据的大小为m×n；
（例：src[0]为400nm通道数据，src[1]为600nm通道数据，src[2]为
Latitude，src[3]为Lontitude）

<2>只限北纬、东经地区，其他地区请自行更改rows、cols、_lon、_lat的
计算方式；

<3>插值方法为反距离权重插值，其他插值方法请自行更改interpolation函数。

注意：分辨率尽量与原始分辨率相近，以防结果中包含空值。
<----------------------------------------------------------------------------------->
1.1更新：提高了稳定性
v1.1 王剑 2020.3.31
"""
import numpy as np
import time


def GEO_PRO(src, xRes=250., yRes=250.):
    """
    :param src: 输入数据
    :param xRes: x方向分辨率，即每列间隔，默认为250m，根据自己的经纬度数据类型决定
    :param yRes: y方向分辨率，即每行间隔，默认为250m，根据自己的经纬度数据类型决定
    :return: 不包含坐标信息的数据及坐标计算方式
    Null为0，如需更改请在40行后进行赋值
    """
    shape = src.shape
    b = shape[0]
    lat_max = np.max(src[b-2]) + yRes
    lat_min = np.min(src[b-2]) - yRes
    lon_max = np.max(src[b-1]) + xRes
    lon_min = np.min(src[b-1]) - xRes
    outype = [lat_max, -1 * yRes, lon_min, xRes]        # 输出的坐标计算方式
    rows = int((lat_max - lat_min) / yRes)
    cols = int((lon_max - lon_min) / xRes)
    output = np.zeros((b-2, rows, cols), dtype=np.double)
    judge = np.zeros((rows, cols), dtype=np.int)
    _M1 = np.zeros((b, 4, 4), dtype=np.double)
    print('Start to project')
    start = time.time()
    for i in range(1, shape[1]-2):
        print(i, '/', shape[1]-3)
        for j in range(1, shape[2]-2):
            _row = int((lat_max - src[b-2][i][j]) / yRes)
            _col = int((src[b-1][i][j] - lon_min) / xRes)
            _M1 = src[:, (i-1):(i+3), (j-1):(j+3)]
            _latM = np.max(_M1[b - 2])
            _latm = np.min(_M1[b - 2])
            _lonM = np.max(_M1[b - 1])
            _lonm = np.min(_M1[b - 1])
            if judge[_row-1][_col-1] != 1:
                _lat = lat_max - (_row-1) * yRes
                _lon = lon_min + (_col-1) * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row-1][_col-1] = interpolation(_lon, _lat, _M1, k)
                    judge[_row-1][_col-1] = 1
            if judge[_row-1][_col] != 1:
                _lat = lat_max - (_row-1) * yRes
                _lon = lon_min + _col * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row-1][_col] = interpolation(_lon, _lat, _M1, k)
                    judge[_row-1][_col] = 1
            if judge[_row-1][_col+1] != 1:
                _lat = lat_max - (_row-1) * yRes
                _lon = lon_min + (_col+1) * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row-1][_col+1] = interpolation(_lon, _lat, _M1, k)
                    judge[_row-1][_col+1] = 1
            if judge[_row][_col-1] != 1:
                _lat = lat_max - _row * yRes
                _lon = lon_min + (_col-1) * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row][_col-1] = interpolation(_lon, _lat, _M1, k)
                    judge[_row][_col-1] = 1
            if judge[_row][_col] != 1:
                _lat = lat_max - _row * yRes
                _lon = lon_min + _col * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row][_col] = interpolation(_lon, _lat, _M1, k)
                    judge[_row][_col] = 1
            if judge[_row][_col+1] != 1:
                _lat = lat_max - _row * yRes
                _lon = lon_min + (_col+1) * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row][_col+1] = interpolation(_lon, _lat, _M1, k)
                    judge[_row][_col+1] = 1
            if judge[_row+1][_col-1] != 1:
                _lat = lat_max - (_row+1) * yRes
                _lon = lon_min + (_col-1) * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row+1][_col-1] = interpolation(_lon, _lat, _M1, k)
                    judge[_row+1][_col-1] = 1
            if judge[_row+1][_col] != 1:
                _lat = lat_max - (_row+1) * yRes
                _lon = lon_min + _col * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row+1][_col] = interpolation(_lon, _lat, _M1, k)
                    judge[_row+1][_col] = 1
            if judge[_row+1][_col+1] != 1:
                _lat = lat_max - (_row+1) * yRes
                _lon = lon_min + (_col+1) * xRes
                if (_lat > _latm) & (_lat < _latM) & (_lon > _lonm) & (_lon < _lonM):
                    for k in range(b - 2):
                        output[k][_row+1][_col+1] = interpolation(_lon, _lat, _M1, k)
                    judge[_row+1][_col+1] = 1
    end = time.time()
    print('Time: ', end - start)
    return output, outype


def interpolation(lon, lat, lst, _b):
    # 反距离权重插值
    band = lst.shape[0]
    sum0 = 0
    sum1 = 0
    mlat = lst[band-2].flatten()
    mlon = lst[band-1].flatten()
    val = lst[_b].flatten()
    for _i in range(16):
        Di = (lat - mlat[_i]) * (lat - mlat[_i]) + (lon - mlon[_i]) * (lon - mlon[_i])
        sum0 += val[_i] / Di
        sum1 += 1 / Di
    return sum0 / sum1


if __name__ == '__main__':
    pass
