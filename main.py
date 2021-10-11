import argparse
import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# параметры установки
R = 4.5 / 1000
g = 9.8
pix_in_met = 9400
# x_from_left = 1490 # pixels
# y_from_top = 865


def get_imgbw_data(img_file_name):
    img_file = Image.open(img_file_name)
    img_file_bw = img_file.convert("1")
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
    # xy = np.column_stack((x, y))
    x = np.array(x) / pix_in_met
    y = np.array(y) / pix_in_met
    return (x.mean(), y.mean())


def process_file_names(imgs):
    imgs_coomp_names = [f.split(".")[0].split("_") for f in imgs]
    particle_nums = [i[1] for i in imgs_coomp_names]
    voltage_dc = [float(i[2]) / 10 for i in imgs_coomp_names]
    iteration_num = [i[-1] for i in imgs_coomp_names]
    return particle_nums, voltage_dc, iteration_num


def get_coords(imgs):
    coords = []
    for img in tqdm(imgs):
        imgbw_data = get_imgbw_data(f"{path}/{img}")
        xy = count_coords(imgbw_data)
        coords.append(xy)
    return coords


def build_output(imgs):
    coords = get_coords(imgs)
    xs = np.array([i[0] for i in coords])
    ys = np.array([i[1] for i in coords])
    particle_nums, voltage_dc, iteration_num = process_file_names(imgs)
    whole_info = pd.DataFrame(
        {
            "Particle_num": particle_nums,
            "Voltage_DC": voltage_dc,
            "Iteration_num": iteration_num,
            "x": xs,
            "y": ys,
        }
    )

    return whole_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Getting a folder to process")
    parser.add_argument("--p", default="imgs", help="Enter path to imgs to process")
    args = parser.parse_args()
    path = args.p

    imgs = [f for f in listdir(path) if isfile(join(path, f))]

    whole_info = build_output(imgs)
    i = 0
    for particle in whole_info["Particle_num"].unique():
        info_mean = (
            whole_info[whole_info["Particle_num"] == particle]
            .groupby("Voltage_DC")
            .mean()[["x", "y"]]
            .reset_index()
        )

        info_mean["Q/m"] = (g * (R ** 2 + info_mean.x ** 2) ** (3 / 2)) / (
            info_mean.Voltage_DC * R
        )

        info_mean.x = info_mean.x.round(4)
        info_mean.y = info_mean.y.round(4)
        info_mean["Q/m"] = info_mean.x.round(6)

        plt.figure(i + 1)  # to let the index start at 1
        sns.scatterplot(data=info_mean, x="Voltage_DC", y="Q/m")
        plt.title(f"Particle {particle}")

        if os.path.exists(f"info_mean_{particle}.csv"):
            existed_df = pd.read_csv(f"info_mean_{particle}.csv")
            info_mean = (
                pd.concat([existed_df, info_mean])
                .drop_duplicates()
                .reset_index(drop=True)
            )

        info_mean.to_csv(f"info_mean_{particle}.csv")

        i += 1
    plt.show()
