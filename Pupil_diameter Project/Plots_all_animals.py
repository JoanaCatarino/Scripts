#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:10:43 2022

@author: joana
"""


from pupil_functions import load_pupil, load_trials
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import zscore
from plot_functions import (figure_style)
from os.path import join
from one.api import ONE
from pupil_size_plots import all_contrasts_by_blocks, all_contrasts_per_block_by_stim_side, all_contrasts_all_blocks_correct_error_by_stim_side_figure, n_trials_choice
one = ONE()


# Load data 
pupil_size_df = pd.read_csv('/home/joana/Desktop/IBL/Scripts/Pupil_diameter Project/df_files/All_animals_Mainen_mean.csv')
subject = pupil_size_df.subject.unique()

# ------ PLOTS ------


  pupil_size = pupil_size_df

  pupil_size = pupil_size.reset_index(drop=True)
  dpi = figure_style()
  colors = ['#F96E46']


# Transition from 0.8

  df_slice = pupil_size_df[((pupil_size_df['Stim_side'] == 1) & (pupil_size_df['transition_from'] == 0.8) & (pupil_size_df['contrast'] == 0.125))].reset_index()

  f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
  lineplt = sns.lineplot(x='time', y='baseline_subtracted', data=df_slice, legend='full',
                      ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
  ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1' '    ' 'All Animals Mainen', ylim=[-25, 25])
  plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
  ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
  sns.despine(trim=True)


# Transition from 0.2

  df_slice = pupil_size_df[((pupil_size_df['Stim_side'] == -1) & (pupil_size_df['transition_from'] == 0.2) & (pupil_size_df['contrast'] == -0.125))].reset_index()

  f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
  lineplt = sns.lineplot(x='time', y='baseline_subtracted', data=df_slice, legend='full',
                      ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
  ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1' '    ' 'All Animals Mainen', ylim=[-25, 25])
  plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
  ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
  sns.despine(trim=True)









all_contrasts_by_blocks(pupil_size_df, subject)

all_contrasts_per_block_by_stim_side(pupil_size_df, subject)

all_contrasts_all_blocks_correct_error_by_stim_side_figure(pupil_size_df, subject)

n_trials_choice(pupil_size_df, subject)


transition_from = 0.8
contrast = 1