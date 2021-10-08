import numpy as np
import pandas as pd

from PIL import Image
from os import listdir
from os.path import isfile, join


def get_imgbw_data(img_file_name):
    img_file = Image.open(img_file_name)
    img_file_bw = img_file.convert('1')
    imgbw_data = img_file_bw.getdata()
    return imgbw_data


def count_coords(imgbw_data):
    imgWidth, imgHeight = 1920, 1080
    x_pos = 0
    y_pos = 1
    x = []
    y = []
    for item in imgbw_data:
        if (x_pos) == imgWidth:
            x_pos = 1
            y_pos += 1
        else:
            x_pos += 1
        if item != 0:
            x.append(x_pos)
            y.append(y_pos)
    xy = np.column_stack((x, y))
    x = np.array(x)
    y = np.array(y)
    return (x.mean().round(2), y.mean().round(2))


if __name__ == "__main__":
    # получение всех изображений из папки
    path = input("Введите папку для обработки , например 'imgs', 'dataset/photos', etc.\n" )
    imgs = [f for f in listdir(path) if isfile(join(path, f))]
    imgs_coomp_names = [f.split('.')[0].split('_') for f in imgs]

    # конвертация изображений в чёрно-белое
    coords = []
    for img in imgs:
        imgbw_data = get_imgbw_data(f'{path}/{img}')
        xy = count_coords(imgbw_data)
        coords.append(xy)


    particle_nums = [i[1] for i in imgs_coomp_names]
    iteration_num = [i[-1] for i in imgs_coomp_names]
    smt_frequency_value = [i[-2] for i in imgs_coomp_names]
    else_smt = [i[2] for i in imgs_coomp_names]

    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]

    whole_info = pd.DataFrame({
        'Particle_num': particle_nums,
        'Iteration_num': iteration_num,
        'Smt_freqval': smt_frequency_value,
        'else_smt': else_smt,
        'x': xs,
        'y': ys
    })
    info_mean = whole_info.groupby(['else_smt', 'Smt_freqval', 'Particle_num']).mean()[['x', 'y']]

    whole_info.to_csv('whole_info_2.csv')
    info_mean.to_csv('info_mean_2.csv')

