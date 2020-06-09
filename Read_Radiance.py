"""
功能：读取Landsat8辐射定标系数

v1.0 王剑 2020/01/14
"""


def Read_Radiance(_band, _path):
    import functools

    # 读取头文件数据
    f = open(_path + 'MTL.txt', 'r')
    head_data = f.read()
    f.close()

    mult_data = []
    add_data = []

    path_name = 'RADIANCE_MULT_BAND_' + str(_band)
    mult_index = int(head_data.find(path_name))

    path_name = 'RADIANCE_ADD_BAND_' + str(_band)
    add_index = int(head_data.find(path_name))

    mult_data.append(head_data[mult_index + 23])
    add_data.append(head_data[add_index + 23])
    add_data.append(head_data[add_index + 24])

    for i in range(4):
        _ = int(head_data[mult_index + 25 + i])
        mult_data.append(_)

    for i in range(5):
        _ = int(head_data[add_index + 26 + i])
        add_data.append(_)

    radmult = int(functools.reduce(lambda x, y: str(x) + str(y), mult_data)) / 1000000
    radadd = int(functools.reduce(lambda x, y: str(x) + str(y), add_data)) / 100000

    return radmult, radadd


if __name__ == '__main__':
    print(Read_Radiance(2, './Data/Landsat8/LC08_L1TP_123032_20191207_20191217_01_T1'
                           '/LC08_L1TP_123032_20191207_20191217_01_T1~/LC08_L1TP_123032_20191207_20191217_01_T1_'))
    # print(Read_Radiance(3))
    # print(Read_Radiance(4))
