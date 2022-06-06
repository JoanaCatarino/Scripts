#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:26:53 2022

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
one = ONE() # if we want to use data already available in the computer use -> one = ONE (mode='local')


# Settings
TIME_BINS = np.arange(-1, 3.2, 0.2)
BIN_SIZE= 0.2 #seconds
BASELINE = [1, 0] #seconds
N_Trials = 10
pupil_size = pd.DataFrame()
results_df = pd.DataFrame()
results_df_baseline = pd.DataFrame()
all_pupil = pd.DataFrame()
all_sessions_one_animal = []


# Query sessions
eids, ses_details = one.search(lab='mainenlab', dataset=['_ibl_leftCamera.dlc.pqt', '_ibl_leftCamera.times.npy'], task_protocol='ephys',
                               details=True, project='ibl_neuropixel_brainwide_01')
subjects = np.unique([s['subject'] for s in ses_details])

                               

for s, subject in enumerate(subjects):
    print(f'Starting {subject}')
    
    
    # Query sesssions of this subject
    eids = one.search(subject=subject, lab='mainenlab', dataset=['_ibl_leftCamera.dlc.pqt', '_ibl_leftCamera.times.npy'],
                      task_protocol='ephys', project='ibl_neuropixel_brainwide_01')
    
    this_sub = pd.DataFrame() 
    for e, eid in enumerate(eids):
        print(f'Session {e+1} of {len(eids)}')        
        
        # Get pupil diameter
        try:
            times, pupil_diameter, raw_pupil_diameter = load_pupil(eid, one=one)
        except Exception as err:
            print(err)
            continue

        # Calculate percentage change
        diameter_perc = ((pupil_diameter - np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2))
                         / np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2)) * 100
        
        # Get session info
        info = one.get_details(eid)
        subject = info['subject']
        date = info['date']

        # Load in trials
        df_Trials = load_trials(eid)
        
        # Find Block transitions
        block_trans = np.append([0], np.array(np.where(np.diff(df_Trials['probabilityLeft']) != 0)) + 1)
        trans_to = df_Trials.loc[block_trans[1:], 'probabilityLeft']
        trans_from = np.concatenate(([np.nan], df_Trials.loc[block_trans[1:] - 1, 'probabilityLeft'].values))
        
        
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
            transition_from = trans_from[last_trans]
            
            for b, time_bin in enumerate (TIME_BINS):
                diameter_1[b] = np.nanmedian(diameter_perc [(np_times > (trial_start + time_bin) - (BIN_SIZE / 2)) & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
                baseline_subtracted[b] = np.nanmedian(diameter_perc[(np_times > (trial_start + time_bin) - (BIN_SIZE / 2)) & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
                
            this_sub = pd.concat((this_sub, pd.DataFrame(data={
                                                        'diameter': diameter_1,  
                                                        'baseline_subtracted': baseline_subtracted, 
                                                        'eid': eid,  
                                                        'subject': subject,  
                                                        'trial': t, 
                                                        'trial_after_switch': trials_since_switch,
                                                        'transition_from': transition_from,
                                                        'contrast': df_Trials.loc[t, 'signed_contrast'], 
                                                        'time': TIME_BINS, 
                                                        'Stim_side':df_Trials.loc[t, 'stim_side'], 
                                                        'Feedback_type':df_Trials.loc[t, 'feedbackType'], 
                                                        'probabilityLeft':df_Trials.loc[t, 'probabilityLeft']})))
        
        
        this_sub['after_switch'] = pd.cut(this_sub['trial_after_switch'], [-1, N_Trials, N_Trials*2, N_Trials*3, np.inf], labels=['0-10 trials', '10-20 trials', '20-30 trials', '+30 trials'])
        subject_1 = this_sub.subject.unique()
        this_sub.to_csv('/home/joana/Desktop/IBL/Scripts/Pupil_diameter Project/df_files/All_sessions_' + subject_1[0] + '.csv')
        all_sessions_one_animal_mean = this_sub.groupby(['time', 'probabilityLeft', 'trial_after_switch', 'transition_from', 'contrast', 'Feedback_type', 'Stim_side']).mean()  
        all_sessions_one_animal_mean.to_csv('/home/joana/Desktop/IBL/Scripts/Pupil_diameter Project/df_files/Mean_' + subject_1[0] + '.csv')
    
    all_sessions_one_animal_mean['subject'] = subject
    all_pupil = pd.concat((all_pupil, all_sessions_one_animal_mean))
    all_pupil.to_csv('/home/joana/Desktop/IBL/Scripts/Pupil_diameter Project/df_files/All_animals_Mainen_mean.csv')
    
print ('Done')















