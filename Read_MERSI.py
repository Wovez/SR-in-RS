"""
功能：
    读取MERSI三通道数据
    进行辐射定标
    linear2%线性拉伸
    几何校正
    保存四角坐标

王剑
"""
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Find_Close import find_close
# from mpl_toolkits.basemap import Basemap
from osgeo import gdal
from osgeo import osr
import skimage.exposure as exposure
from GEO_Projection import GEO_PRO

date = '20191207_0450'
# date = '20200101_0515'
# date = '20191205_0525'
# date = '20191119_0530'


def Read_MERSI():
    _imgpath = 'D:/Python/Data/FY3D-MERSI-250m/FY3D_MERSI_GBAL_L1_' + date + '_0250M_MS.HDF'
    _solarpath = 'D:/Python/Data/FY3D-GEOQK_GEO1K/GEO1K/FY3D_MERSI_GBAL_L1_' + date + '_GEO1K_MS.HDF'
    _crpath = 'D:/Python/Data/FY3D-GEOQK_GEO1K/GEOQK/FY3D_MERSI_GBAL_L1_' + date + '_GEOQK_MS.HDF'

    f = h5py.File(_imgpath, 'r')

    # 读取辐射定标系数
    # Calibration = f['Calibration'].keys()
    # print(Calibration)
    VIS_Cal_Coeff = f['Calibration']['VIS_Cal_Coeff'][:]
    # print(VIS_Cal_Coeff)

    # 日地距离率的平方
    des_2 = f.attrs['EarthSun Distance Ratio'][0] ** 2

    # 读取各通道数据
    b_Blue_0 = np.mat(f['Data']['EV_250_RefSB_b1'][:])
    b_Blue_1 = (VIS_Cal_Coeff[0][0] + VIS_Cal_Coeff[0][1] * b_Blue_0) / 100
    b_Green_0 = np.mat(f['Data']['EV_250_RefSB_b2'][:])
    b_Green_1 = (VIS_Cal_Coeff[1][0] + VIS_Cal_Coeff[1][1] * b_Green_0) / 100
    b_Red_0 = np.mat(f['Data']['EV_250_RefSB_b3'][:])
    b_Red_1 = (VIS_Cal_Coeff[2][0] + VIS_Cal_Coeff[2][1] * b_Red_0) / 100
    f.close()

    """# 消除错误数据
    b_Blue_1[b_Blue_1[:] > 1] = 1
    b_Blue_1[b_Blue_1[:] < 0] = 0
    b_Green_1[b_Green_1[:] > 1] = 1
    b_Green_1[b_Green_1[:] < 0] = 0
    b_Red_1[b_Red_1[:] > 1] = 1
    b_Red_1[b_Red_1[:] < 0] = 0"""

    shape = b_Blue_1.shape

    # 太阳高度角校正
    f_solar = h5py.File(_solarpath, 'r')
    solar_zenith = np.mat(f_solar['Geolocation']['SolarZenith'][:]) / 100
    solar_zenith = np.around(cv2.resize(solar_zenith, (shape[1], shape[0]), cv2.INTER_NEAREST), decimals=2)
    b_Blue_1 = b_Blue_1 * des_2 / np.cos(solar_zenith / 180 * np.pi)
    b_Green_1 = b_Green_1 * des_2 / np.cos(solar_zenith / 180 * np.pi)
    b_Red_1 = b_Red_1 * des_2 / np.cos(solar_zenith / 180 * np.pi)
    f_solar.close()

    # 读坐标系
    f_Geo = h5py.File(_crpath, 'r')
    Lat = np.mat(f_Geo['Latitude'][:])
    Lon = np.mat(f_Geo['Longitude'][:])
    f_Geo.close()

    # 图像和坐标合为一个
    data = np.zeros((5, shape[0], shape[1]), dtype=np.double)
    data[0, :] = np.flip(b_Blue_1)
    data[1, :] = np.flip(b_Green_1)
    data[2, :] = np.flip(b_Red_1)
    data[3, :] = np.flip(Lat)
    data[4, :] = np.flip(Lon)

    return data


def Geo_Area(data):
    # 三景
    leftup = find_close(data[4], data[3], 115, 41.5)
    leftdown = find_close(data[4], data[3], 115, 37.5)
    rightup = find_close(data[4], data[3], 120, 41.5)
    rightdown = find_close(data[4], data[3], 120, 37.5)

    # 单景
    """leftup = find_close(data[4], data[3], 115.3, 41.5)
    leftdown = find_close(data[4], data[3], 115.3, 39.5)
    rightup = find_close(data[4], data[3], 118, 41.5)
    rightdown = find_close(data[4], data[3], 118, 39.5)"""

    # 河南
    """leftup = find_close(data[4], data[3], 111, 35.5)
    leftdown = find_close(data[4], data[3], 111, 33.5)
    rightup = find_close(data[4], data[3], 113, 35.5)
    rightdown = find_close(data[4], data[3], 113, 33.5)"""

    i = np.array([min(leftup[0], rightup[0]), max(leftdown[0], rightdown[0]), min(leftdown[1], leftup[1]), max(rightdown[1], rightup[1])], dtype=int)
    # print(i)
    return i


def linear_2(red_0=None, green_0=None, blue_0=None, per=2):
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


def savedata(_data):
    # 将预处理完的图像保存为tif格式
    shape = _data[0].shape
    gtiff_driver = gdal.GetDriverByName('GTiff')
    _path = './Data/MERSI/Origin/' + date + '.tif'
    _latpath = './Data/MERSI/GEO/' + date + '_latitude.tif'
    _lonpath = './Data/MERSI/GEO/' + date + '_longtitude.tif'
    output = gtiff_driver.Create(_path, shape[1], shape[0], 3, gdal.GDT_Float32)
    out_band = output.GetRasterBand(3)
    out_band.WriteArray(_data[2])
    out_band = output.GetRasterBand(2)
    out_band.WriteArray(_data[1])
    out_band = output.GetRasterBand(1)
    out_band.WriteArray(_data[0])

    import pyproj
    p2 = pyproj.Proj(init="epsg:32650")
    p1 = pyproj.Proj(init="epsg:4326")
    _data[4], _data[3] = pyproj.transform(p1, p2, _data[4], _data[3])
    _lat = gtiff_driver.Create(_latpath, shape[1], shape[0], 1, gdal.GDT_Float32)
    _latband = _lat.GetRasterBand(1)
    _latband.WriteArray(_data[3])
    _lon = gtiff_driver.Create(_lonpath, shape[1], shape[0], 1, gdal.GDT_Float32)
    _lonband = _lon.GetRasterBand(1)
    _lonband.WriteArray(_data[4])

    # 保存VRT文件
    outfile = './Data/MERSI/GC/' + date + '.VRT'
    # 图像尺寸
    vrt = '<VRTDataset rasterXSize="' + str(shape[1]) + '" rasterYSize="' + str(shape[0]) + '">\n'
    vrt += '  <Metadata domain="GEOLOCATION">\n'
    # 经纬度文件位置
    vrt += '    <MDI key="X_DATASET">D:/Python/Data/MERSI/GEO/' + date + '_longtitude.tif</MDI>\n'
    vrt += '    <MDI key="X_BAND">1</MDI>\n'
    # 读取偏移
    vrt += '    <MDI key="PIXEL_OFFSET">0</MDI>\n'
    vrt += '    <MDI key="PIXEL_STEP">1</MDI>\n'
    vrt += '    <MDI key="Y_DATASET">D:/Python/Data/MERSI/GEO/' + date + '_latitude.tif</MDI>\n'
    vrt += '    <MDI key="Y_BAND">1</MDI>\n'
    vrt += '    <MDI key="LINE_OFFSET">0</MDI>\n'
    vrt += '    <MDI key="LINE_STEP">1</MDI>\n'
    vrt += '  </Metadata>\n'
    # 注册RGB三个波段
    vrt += '  <VRTRasterBand dataType="Float32" band="1">\n'
    vrt += '    <ColorInterp>Blue</ColorInterp>\n'
    vrt += '    <NoDataValue>0</NoDataValue>\n'
    vrt += '    <SimpleSource>\n'
    vrt += '      <SourceFilename relativeToVRT="1">D:/Python/Data/MERSI/Origin/20191207_0450.tif</SourceFilename>\n'
    vrt += '      <SourceBand>1</SourceBand>\n'
    vrt += '      <SrcRect xOff="0" yOff="0" xSize="' + str(shape[1]) + '" ySize="' + str(shape[0]) + '"/>\n'
    vrt += '      <DstRect xOff="0" yOff="0" xSize="' + str(shape[1]) + '" ySize="' + str(shape[0]) + '"/>\n'
    vrt += '    </SimpleSource>\n'
    vrt += '  </VRTRasterBand>\n'
    vrt += '  <VRTRasterBand dataType="Float32" band="2">\n'
    vrt += '    <ColorInterp>Green</ColorInterp>\n'
    vrt += '    <NoDataValue>0</NoDataValue>\n'
    vrt += '    <SimpleSource>\n'
    vrt += '      <SourceFilename relativeToVRT="1">D:/Python/Data/MERSI/Origin/20191207_0450.tif</SourceFilename>\n'
    vrt += '      <SourceBand>2</SourceBand>\n'
    vrt += '      <SrcRect xOff="0" yOff="0" xSize="' + str(shape[1]) + '" ySize="' + str(shape[0]) + '"/>\n'
    vrt += '      <DstRect xOff="0" yOff="0" xSize="' + str(shape[1]) + '" ySize="' + str(shape[0]) + '"/>\n'
    vrt += '    </SimpleSource>\n'
    vrt += '  </VRTRasterBand>\n'
    vrt += '  <VRTRasterBand dataType="Float32" band="3">\n'
    vrt += '    <ColorInterp>Red</ColorInterp>\n'
    vrt += '    <NoDataValue>0</NoDataValue>\n'
    vrt += '    <SimpleSource>\n'
    vrt += '      <SourceFilename relativeToVRT="1">D:/Python/Data/MERSI/Origin/20191207_0450.tif</SourceFilename>\n'
    vrt += '      <SourceBand>3</SourceBand>\n'
    vrt += '      <SrcRect xOff="0" yOff="0" xSize="' + str(shape[1]) + '" ySize="' + str(shape[0]) + '"/>\n'
    vrt += '      <DstRect xOff="0" yOff="0" xSize="' + str(shape[1]) + '" ySize="' + str(shape[0]) + '"/>\n'
    vrt += '    </SimpleSource>\n'
    vrt += '  </VRTRasterBand>\n'
    vrt += '</VRTDataset>\n'

    open(outfile, 'w').write(vrt)


def GC_GCPs(_data):
    # 根据地面控制点进行几何校正
    import pyproj
    _mult = 100
    _path = './Data/MERSI/Origin/' + date + '.tif'
    _otpath = './Data/MERSI/GC/' + date + '.tif'
    p2 = pyproj.Proj(init="epsg:32650")
    p1 = pyproj.Proj(init="epsg:4326")
    _dataset = gdal.Open(_path, gdal.GA_Update)
    _row, _col = _data[0].shape
    _row -= 1
    _col -= 1
    print(_row, _col)
    _rows = _row // _mult
    _cols = _col // _mult
    gcps_list = []
    for i in range(_rows):
        for j in range(_cols):
            _data[4][i*_mult][j*_mult], _data[3][i*_mult][j*_mult] = pyproj.transform(p1, p2, _data[4][i*_mult][j*_mult], _data[3][i*_mult][j*_mult])
            _temp = gdal.GCP(_data[4][i*_mult][j*_mult], _data[3][i*_mult][j*_mult], 0, j*_mult, i*_mult)
            gcps_list.append(_temp)
    sr = osr.SpatialReference()
    # sr.SetWellKnownGeogCS('WGS84')
    sr.ImportFromEPSG(32650)
    _dataset.SetGCPs(gcps_list, sr.ExportToWkt())
    print('start')
    dst_ds = gdal.Warp(_otpath, _dataset, format='GTiff', tps=True,
                       xRes=30, yRes=30, dstNodata=0, srcNodata=None, multithread=True,
                       resampleAlg=gdal.GRIORA_NearestNeighbour, outputType=gdal.GDT_Float32)
    print('end')


def GC_GEO():
    # 根据像素点经纬度信息进行几何校正
    _datapath = './Data/MERSI/GC/' + date + '.VRT'
    _otpath = './Data/MERSI/GC/' + date + '.tif'
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(32650)
    ds = gdal.Warp(_otpath, _datapath, dstNodata=0, srcNodata=None, multithread=True,
                   xRes=30, yRes=30, format='GTiff', geoloc=True, errorThreshold=0,
                   dstSRS=sr.ExportToWkt(), resampleAlg=gdal.GRIORA_NearestNeighbour)


def saveimg():
    # 将几何校正后的数据保存为bmp格式图片
    _row1 = 5000
    _row2 = 8000
    _col1 = 3000
    _col2 = 7500

    _datapath = './Data/MERSI/GC/' + date + '.tif'
    _imgpath = './Data/MERSI/Final/' + date + '_MERSI.bmp'
    _nppath = './Data/MERSI/Final/' + date + '_MERSI_data'
    _crpath = './Data/MERSI/Final/' + date
    _data = gdal.Open(_datapath)

    band_blue_data = _data.GetRasterBand(1)
    blue = np.mat(band_blue_data.ReadAsArray())
    print(blue.shape)
    band_green_data = _data.GetRasterBand(2)
    green = np.mat(band_green_data.ReadAsArray())
    band_red_data = _data.GetRasterBand(3)
    red = np.mat(band_red_data.ReadAsArray())

    blue = blue[_row1:_row2, _col1:_col2]
    green = green[_row1:_row2, _col1:_col2]
    red = red[_row1:_row2, _col1:_col2]

    _sr = _data.GetGeoTransform()

    _x1 = _sr[3] + _sr[5] * _row1
    _x2 = _sr[3] + _sr[5] * _row2
    _y1 = _sr[0] + _sr[1] * _col1
    _y2 = _sr[0] + _sr[1] * _col2
    _temp = [_x1, _x2, _y1, _y2]
    np.save(_crpath, _temp)

    """plt.hist(blue)
    plt.show()"""

    _rgb = cv2.merge([blue, green, red])
    np.save(_nppath, _rgb)
    red, green, blue = linear_2(red, green, blue, 2)
    _rgb = cv2.merge([blue, green, red])
    cv2.imwrite(_imgpath, _rgb)


def savePro(_data, _sr):
    # 三景
    _row1 = 600
    _row2 = 1600
    _col1 = 1000
    _col2 = 2360
    # 单幅
    """_row1 = 520
    _row2 = 1320
    _col1 = 530
    _col2 = 1230"""
    # 河南
    """_row1 = 250
    _row2 = 700
    _col1 = 400
    _col2 = 1200"""

    _imgpath = './Data/MERSI/Final/' + date + '_MERSI.bmp'
    _nppath = './Data/MERSI/Final/' + date + '_MERSI_data'
    _crpath = './Data/MERSI/Final/' + date

    blue = _data[0][_row1:_row2, _col1:_col2]
    green = _data[1][_row1:_row2, _col1:_col2]
    red = _data[2][_row1:_row2, _col1:_col2]

    _x1 = _sr[2] + _sr[3] * _col1
    _x2 = _sr[2] + _sr[3] * _col2
    _y1 = _sr[0] + _sr[1] * _row1
    _y2 = _sr[0] + _sr[1] * _row2
    _temp = [_x1, _x2, _y1, _y2]
    np.save(_crpath, _temp)

    _rgb = cv2.merge([blue, green, red])
    shape = _rgb.shape
    _rgb = cv2.resize(_rgb, (shape[1] * 4, shape[0] * 4), cv2.INTER_CUBIC)
    np.save(_nppath, _rgb)
    blue, green, red = cv2.split(_rgb)
    red, green, blue = linear_2(red, green, blue, 2)
    _rgb = cv2.merge([blue, green, red])
    cv2.imwrite(_imgpath, _rgb)


if __name__ == '__main__':
    """Data = Read_MERSI()        # 读取MERSI数据 + 辐射定标 + 太阳高度角订正 + 保存经纬度信息
    idx = Geo_Area(Data)
    Data = Data[:, idx[0]: idx[1]]
    Data = Data[:, :, idx[2]: idx[3]]
    # savedata(Data)
    # GC_GCPs(Data)
    # del Data
    # GC_GEO()
    # saveimg()
    import pyproj
    p2 = pyproj.Proj(init="epsg:32650")    # 北京
    # p2 = pyproj.Proj(init="epsg:32649")    # 河南
    p1 = pyproj.Proj(init="epsg:4326")
    Data[4], Data[3] = pyproj.transform(p1, p2, Data[4], Data[3])
    ot, otype = GEO_PRO(Data, 200., 200.)
    print(ot.shape)
    np.save('./Data/MERSI/GC/' + date + '_PRO', ot)
    np.save('./Data/MERSI/GC/' + date + '_sr', otype)"""

    ot = np.load('./Data/MERSI/GC/' + date + '_PRO.npy')
    otype = np.load('./Data/MERSI/GC/' + date + '_sr.npy')
    savePro(ot, otype)

    """blue = ot[0]
    green = ot[1]
    red = ot[2]
    blue, green, red = linear_2(blue, green, red)
    ot = cv2.merge([blue, green, red])
    cv2.imwrite('./test.bmp', ot)"""

    """# 投影到地图上
    fig = plt.figure(figsize=(8, 8))
    m = Basemap(llcrnrlon=110, llcrnrlat=35, urcrnrlon=125, urcrnrlat=45)
    m.drawcoastlines()
    m.drawcountries(linewidth=1)
    x, y = m(Data[4], Data[3])
    sc_1 = m.scatter(x, y, c=Data[0], alpha=0.5, s=10, cmap=plt.get_cmap('Reds_r'))
    # sc_2 = m.scatter(x, y, c=green, alpha=0.5, s=10, cmap=plt.get_cmap('Greens_r'))
    # sc_3 = m.scatter(x, y, c=blue, alpha=0.5, s=10, cmap=plt.get_cmap('Blues_r'))

    # plt.imshow(m)
    plt.show()"""
