# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:58:15 2022

@author: joana
"""

from pupil_functions import load_pupil, load_trials
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import zscore
import tkinter as tk
from os.path import join
from one.api import ONE
one = ONE(mode = 'local')



def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks",
            font="Arial",
            rc={"font.size": 7,
                 "axes.titlesize": 7,
                 "axes.labelsize": 7,
                 "axes.linewidth": 0.5,
                 "lines.linewidth": 1,
                 "lines.markersize": 3,
                 "xtick.labelsize": 7,
                 "ytick.labelsize": 7,
                 "savefig.transparent": True,
                 "xtick.major.size": 2.5,
                 "ytick.major.size": 2.5,
                 "xtick.major.width": 0.5,
                 "ytick.major.width": 0.5,
                 "xtick.minor.size": 2,
                 "ytick.minor.size": 2,
                 "xtick.minor.width": 0.5,
                 "ytick.minor.width": 0.5,
                 'legend.fontsize': 7,
                 'legend.title_fontsize': 7
                 })
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 12
    return  dpi







 #%% 
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    colors = {'general': 'orange',
              'grey': [0.75, 0.75, 0.75],
              'sert': sns.color_palette('Dark2')[0],
              'wt': [0.75, 0.75, 0.75],
              'left': sns.color_palette('colorblind')[1],
              'right': sns.color_palette('colorblind')[0],
              'enhanced': sns.color_palette('colorblind')[3],
              'suppressed': sns.color_palette('colorblind')[0],
              'stim': sns.color_palette('colorblind')[9],
              'no-stim': sns.color_palette('colorblind')[7],
              'glm_stim': '#CF453C',
              'glm_motion': '#6180E9',
              'probe': sns.color_palette('colorblind')[4],
              'block': sns.color_palette('colorblind')[6],
              'RS': sns.color_palette('Set2')[0],
              'FS': sns.color_palette('Set2')[1],
              'SC': sns.color_palette('tab20c')[4],
              'MRN': sns.color_palette('tab20c')[5],
              'PAG': sns.color_palette('tab20c')[6],
              'M2': sns.color_palette('Dark2')[2],
              'mPFC': sns.color_palette('Dark2')[1],
              'ORB': sns.color_palette('Dark2')[0],
              'M2-mPFC': sns.color_palette('Dark2')[1],
              'M2-ORB': sns.color_palette('Dark2')[0],
              'PPC': sns.color_palette('tab20c')[0],
              'BC': sns.color_palette('tab20c')[1],
              'Str': sns.color_palette('tab20c')[12],
              'SNr': sns.color_palette('tab20c')[13],
              'Thal': sns.color_palette('tab10')[3],
              'Amyg': sns.color_palette('tab20b')[12],
              'Pir': sns.color_palette('tab20b')[9],
              'Hipp': sns.color_palette('tab20')[12]}
    
    colors = {'block0.2': [],
              'block0.5': [],
              'block0.8': []}