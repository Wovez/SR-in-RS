"""
功能：
    直接读取Landsat8的TIF文件
    进行辐射定标
v1.1:输出bmp文件作为Ground Truth
v2.0：新加入和MERSI图像配准功能

v2.0 王剑 2020/02/23
"""

import numpy as np
import gdal
import cv2
import matplotlib.pyplot as plt
import skimage.exposure as exposure
from Read_Radiance import Read_Radiance as RR

# date = '20200101_0515'
date = '20191207_0450'
# date = '20191119_0530'
path = './Data/Landsat8/LC08_L1TP_123032_20191207_20191217_01_T1/LC08_L1TP_123032_20191207_20191217_01_T1~/' \
       'LC08_L1TP_123032_20191207_20191217_01_T1_'
"""path = './Data/Landsat8/Henan/LC08_L1TP_125036_20191119_20191202_01_T1/' \
       'LC08_L1TP_125036_20191119_20191202_01_T1_'"""
cor_path = './Data/MERSI/Final/' + date + '.npy'
save_path = './Data/MERSI/Final/' + date + '_Landsat8_data'
img_path = './Data/MERSI/Final/' + date + '_Landsat8.bmp'


def Read():
    # 读取各通道数据 + 辐射定标 + 保存坐标（WGS 84/UTM zone 50N）
    dataset_blue = gdal.Open(path + 'B2' + '.TIF')
    dataset_green = gdal.Open(path + 'B3' + '.TIF')
    dataset_red = gdal.Open(path + 'B4' + '.TIF')
    band_blue_data = dataset_blue.GetRasterBand(1)
    blue = np.mat(band_blue_data.ReadAsArray())
    band_green_data = dataset_green.GetRasterBand(1)
    green = np.mat(band_green_data.ReadAsArray())
    band_red_data = dataset_red.GetRasterBand(1)
    red = np.mat(band_red_data.ReadAsArray())

    blue_coordinate = dataset_blue.GetGeoTransform()
    print(blue_coordinate)

    _lon_0 = blue_coordinate[0]
    _lon_plus = blue_coordinate[1]
    _lat_0 = blue_coordinate[3]
    _lat_plus = blue_coordinate[5]

    [_lon_min, _lon_max, _lat_min, _lat_max] = np.load(cor_path)        # 读取MERSI裁剪范围
    print(_lon_min, _lon_max, _lat_min, _lat_max)

    _x_min = int(round((_lon_min - _lon_0) / _lon_plus))
    _x_max = int(round((_lon_max - _lon_0) / _lon_plus))
    _y_min = int(round((_lat_min - _lat_0) / _lat_plus))
    _y_max = int(round((_lat_max - _lat_0) / _lat_plus))
    print([_x_min, _x_max, _y_min, _y_max])

    blue = blue[_y_min: _y_max, _x_min: _x_max]
    green = green[_y_min:_y_max, _x_min: _x_max]
    red = red[_y_min:_y_max, _x_min: _x_max]

    # 对数据进行辐射定标
    """blue_radmul, blue_radadd = RR(2, path)
    green_radmul, green_radadd = RR(3, path)
    red_radmul, red_radadd = RR(4, path)
    blue = blue_radmul * blue - blue_radadd
    green = green_radmul * green - green_radadd
    red = red_radmul * red - red_radadd"""
    with open(path + 'MTL.txt', 'r') as fp:
        import functools
        import math
        _head_data = fp.read()
        _index = int(_head_data.find('SUN_ELEVATION'))
        _sun_data = [_head_data[_index + 16], _head_data[_index + 17]]
        for i in range(8):
            _ = int(_head_data[_index + 19 + i])
            _sun_data.append(_)
        _sun_elevation = int(functools.reduce(lambda x, y: str(x) + str(y), _sun_data)) / 1e8
        _miu = math.sin(_sun_elevation / 180 * math.pi)
    blue = (blue * 0.00002 - 0.1) / _miu
    green = (green * 0.00002 - 0.1) / _miu
    red = (red * 0.00002 - 0.1) / _miu
    """plt.hist(blue)
    plt.show()"""

    # 保存原始数据
    img = cv2.merge([blue, green, red])
    np.save(save_path, img)

    # 2%线性变换
    blue, green, red = linear_2(blue, green, red, 2)

    # 将三个通道的数据变为一张图像
    img = cv2.merge([blue, green, red])
    cv2.imwrite(img_path, img)


def Map_Pro(_data):
    # 将WGS84地理坐标转换为经纬度
    import pyproj
    _p1 = pyproj.Proj(init="epsg:32650")
    _p2 = pyproj.Proj(init="epsg:4326")
    _data[3], _data[4] = pyproj.transform(_p1, _p2, _data[3], _data[4])
    print(_data[3][0][0])
    return _data


def linear_2(red_0, green_0, blue_0, per=2):
    # linear per%线性拉伸
    # per: 线性拉伸百分比，默认为2
    p2, p98 = np.percentile(red_0, (per, 100-per))
    red = exposure.rescale_intensity(red_0, in_range=(p2, p98))
    p2, p98 = np.percentile(green_0, (per, 100-per))
    green = exposure.rescale_intensity(green_0, in_range=(p2, p98))
    p2, p98 = np.percentile(blue_0, (per, 100-per))
    blue = exposure.rescale_intensity(blue_0, in_range=(p2, p98))
    red *= 255
    green *= 255
    blue *= 255
    red = red.astype(np.int)
    green = green.astype(np.int)
    blue = blue.astype(np.int)
    return red, green, blue


if __name__ == '__main__':
    Read()
    # Data = Map_Pro(Data)
