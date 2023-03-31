from argparse import ArgumentParser
import os
import pickle
import re
import math
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import datetime
from tqdm import tqdm
from utils import *
import time
from datetime import timedelta
from minizinc import Instance, Model, Solver
import numpy as np
from glob import glob
import tkinter as tk
import tkinter.filedialog as filedialog


def alphanumeric_sort(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def get_circuits(circuit_widths, circuit_heights, start_x, start_y):
    circuits = []
    for i in range(len(start_x)):
        circuits.append([circuit_widths[i], circuit_heights[i], start_x[i], start_y[i]])
    
    return circuits

def plot_solution(circuits, width, height, title, file='', rotation = False, r=[]):
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)

    for i,(w,h,x,y) in enumerate(circuits):
        if rotation and r[i]:
            w, h = h, w
        rect = patches.Rectangle((x, y), w, h, linewidth = 2, edgecolor= 'black', facecolor = colors.hsv_to_rgb((i / len(circuits), 1, 1)))
        ax.add_patch(rect)
        
    ax.set_yticks(np.arange(height+1))
    ax.set_xticks(np.arange(width+1))
    ax.grid(color='black', linewidth = 1)
    
    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()
        