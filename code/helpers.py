import os, string
import numpy as np


def run_cmd(str, print_out=True):
    if print_out:
        print()
        print(str)
    os.system(str)
    if print_out:
        print()


def makeNice(axes, labelsize=8, lw=1.5, width=0):
    if type(axes) == list:
        for ax in axes:
            for i in ["left", "right", "top", "bottom"]:
                if i != "left" and i != "bottom":
                    ax.spines[i].set_visible(False)
                    ax.tick_params("both", width=0, labelsize=labelsize)
                else:
                    ax.spines[i].set_linewidth(lw)
                    ax.tick_params("both", width=width, labelsize=labelsize)
    else:
        for i in ["left", "right", "top", "bottom"]:
            if i != "left" and i != "bottom":
                axes.spines[i].set_visible(False)
                axes.tick_params("both", width=0, labelsize=labelsize)
            else:
                axes.spines[i].set_linewidth(lw)
                axes.tick_params("both", width=width, labelsize=labelsize)


def add_fig_labels(axes, fontsize=10):
    labels = string.ascii_uppercase
    for i in range(len(axes)):
        axes[i].text(
            -0.15,
            1.05,
            labels[i],
            fontsize=fontsize,
            transform=axes[i].transAxes,
            fontweight="bold",
            color="gray",
        )


def match_axis(axes, type="both"):
    if type == "x":
        min = np.min([ax.get_xlim()[0] for ax in axes])
        max = np.max([ax.get_xlim()[1] for ax in axes])
        for ax in axes:
            ax.set_xlim([min, max])
    elif type == "y":
        min = np.min([ax.get_ylim()[0] for ax in axes])
        max = np.max([ax.get_ylim()[1] for ax in axes])
        for ax in axes:
            ax.set_ylim([min, max])
    else:
        match_axis(axes, type="x")
        match_axis(axes, type="y")
