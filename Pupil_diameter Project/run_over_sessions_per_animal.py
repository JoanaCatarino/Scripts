#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:24:35 2022

@author: joana
"""

'''
TEST
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import zscore
from plot_functions import (figure_style)
from os.path import join
from one.api import ONE
from pupil_functions import load_trials
from dlc_functions_new import (get_dlc_XYs, get_raw_and_smooth_pupil_dia)
#from pupil_size_plots import all_contrasts_by_blocks, all_contrasts_per_block_by_stim_side, all_contrasts_all_blocks_correct_error_by_stim_side_figure, n_trials_choice
one = ONE()


# Settings
TIME_BINS = np.arange(-1, 3.2, 0.2)
BIN_SIZE= 0.2 #seconds
BASELINE = [1, 0] #seconds
N_Trials = 10
pupil_size = pd.DataFrame()
results_df = pd.DataFrame()
results_df_baseline = pd.DataFrame()
df_Trials_RT2 = []
all_pupil_sizes = []


# Query sessions
eids = one.search(subject='ZFM-02368', dataset=['_ibl_leftCamera.dlc.pqt'], task_protocol='ephys')

# Loop over sessions
for i, eid in enumerate(eids):

    # Get pupil diameter
    times, _ = get_dlc_XYs(one, eid) 
    pupil_diameter, raw_pupil_diameter = get_raw_and_smooth_pupil_dia(eid, 'left', one)

    # Calculate percentage change
    diameter_perc = ((pupil_diameter - np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2))
                     / np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2)) * 100

    # Get session info
    info = one.get_details(eid)
    subject = info['subject']
    date = info['start_time'].split('T')[0]
    print(f'{subject} {date}')

    # Load in trials
    try:
        df_Trials = load_trials(eid)
    except Exception as err:
        print(err)
        continue
    
    
    # Select only the trial in which reaction time is equal or bigger than 2 seconnds
    df_Trials_RT2 = df_Trials[df_Trials['reaction_times'] >= 2]
    
    
    # Find Block transitions
    block_trans = np.append([0], np.array(np.where(np.diff(df_Trials['probabilityLeft']) != 0)) + 1)
    trans_to = df_Trials.loc[block_trans, 'probabilityLeft']


# Alignment (aligned to stimOn_times)

    np_stimOn = np.array(df_Trials['stimOn_times'])
    np_times = np.array(times)

    for t, trial_start in enumerate(np_stimOn):
        diameter_1 = np.array([np.nan] * TIME_BINS.shape[0])
        baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
        baseline = np.nanmedian(diameter_perc [(np_times > (trial_start - BASELINE[0])) & (np_times < (trial_start - BASELINE[1]))])
        diff_tr = t - block_trans
        last_trans = diff_tr[diff_tr >= 0].argmin()
        trials_since_switch = t - block_trans[last_trans]

        for b, time_bin in enumerate (TIME_BINS):
            diameter_1[b] = np.nanmedian(diameter_perc [(np_times > (trial_start + time_bin) - (BIN_SIZE / 2)) & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
            baseline_subtracted[b] = np.nanmedian(diameter_perc[(np_times > (trial_start + time_bin) - (BIN_SIZE / 2)) & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline

        pupil_size = pd.concat((pupil_size, pd.DataFrame(data={'diameter': diameter_1,
                                                              'baseline_subtracted': baseline_subtracted,
                                                              'eid': eid,
                                                              'subject': subject,
                                                              'trial': t,
                                                              'trial_after_switch': trials_since_switch,
                                                              'contrast': df_Trials.loc[t, 'signed_contrast'],
                                                              'time': TIME_BINS,
                                                              'Stim_side':df_Trials.loc[t, 'stim_side'],
                                                               'Feedback_type':df_Trials.loc[t, 'feedbackType'],
                                                              'probabilityLeft':df_Trials.loc[t, 'probabilityLeft']})))

    pupil_size['after_switch'] = pd.cut(pupil_size['trial_after_switch'], [-1, N_Trials, N_Trials*2, N_Trials*3, np.inf], labels=['0-10 trials', '10-20 trials', '20-30 trials', '+30 trials'])
    
    all_pupil_sizes.append(pupil_size) 
    
pupil_size_df = pd.concat(all_pupil_sizes, axis=0)  

pupil_size_df.to_csv('/home/joana/Desktop/IBL/Scripts/Pupil_diameter Project/df_files/All_sessions_' + subject + '.csv')

pupil_size_mean = pupil_size_df.groupby('time').mean()

pupil_size_mean.to_csv('/home/joana/Desktop/IBL/Scripts/Pupil_diameter Project/df_files/Mean_' + subject + '.csv')


#%%

# ------ PLOTS ------

all_contrasts_by_blocks(pupil_size_df, subject)

all_contrasts_per_block_by_stim_side(pupil_size_df, subject)

all_contrasts_all_blocks_correct_error_by_stim_side_figure(pupil_size_df, subject)

n_trials_choice(pupil_size_df, subject)

