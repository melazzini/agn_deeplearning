from utils import *
import matplotlib.pyplot as plt

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