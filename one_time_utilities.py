from utils import *
import matplotlib.pyplot as plt
from os import listdir
import os, random

def plot_pair_spectra(x,y,x_smooth,y_smooth, title_left, title_right,
                      y_label_left="photon count", 
                      y_label_right="photon count",
                      x_label_left="energy,eV",
                      x_label_right="energy,eV", 
                      x_scale="log"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(x, y)
    axs[0].set_title(title_left)
    axs[0].set_xlabel(x_label_left)
    axs[0].set_ylabel(y_label_left)
    axs[0].set_xscale(x_scale)

    axs[1].plot(x_smooth, y_smooth)
    axs[1].set_title(title_right)
    axs[1].set_xlabel(x_label_right)
    axs[1].set_ylabel(y_label_right)
    axs[1].set_xscale(x_scale)


def generate_train_and_test_paths_files(root_directories,train_data_fraction:float=0.8):
    if os.path.exists("train_spectra_paths.txt") or os.path.exists("test_spectra_paths.txt"):
        print("Train or test spectra paths file exist. You have to manually delete them both if you want to generate new ones!")
        return
        
    all_spectra = []

    for root_dir in root_directories:
        for spectrum_path in listdir(root_dir):
            if "FULL_ALL.spectrum" in spectrum_path and "6075" not in spectrum_path and "7590" not in spectrum_path:
                all_spectra += [os.path.join(root_dir,spectrum_path)]

    random.shuffle(all_spectra)

    train_spectra_paths, test_spectra_paths = all_spectra[:int(train_data_fraction*len(all_spectra))], all_spectra[int(train_data_fraction*len(all_spectra)):]

    # print(len(all_spectra))
    # print(len(train_spectra_paths))
    # print(len(test_spectra_paths))
    # print(len(train_spectra_paths)+len(test_spectra_paths))

    with open("train_spectra_paths.txt","w") as train_paths_file:
        for path_ in train_spectra_paths:
            train_paths_file.write(f"{path_}\n")
            
    with open("test_spectra_paths.txt","w") as test_paths_file:
        for path_ in test_spectra_paths:
            test_paths_file.write(f"{path_}\n")