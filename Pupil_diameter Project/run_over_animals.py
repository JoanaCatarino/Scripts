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



# Query sessions
eids, ses_details = one.search(lab='mainenlab', dataset=['_ibl_leftCamera.dlc.pqt', '_ibl_leftCamera.times.npy'], task_protocol='ephys',
                               details=True, project='ibl_neuropixel_brainwide_01')
subjects = np.unique([s['subject'] for s in ses_details])

 #%%                              

# Aligned to StimOn Times 

all_pupil_so = pd.DataFrame()
all_sessions_one_animal_so = []

for s, subject in enumerate(subjects):
    print(f'Starting {subject}')
    
    
    # Query sesssions of this subject
    eids = one.search(subject=subject, lab='mainenlab', dataset=['_ibl_leftCamera.dlc.pqt', '_ibl_leftCamera.times.npy'],
                      task_protocol='ephys', project='ibl_neuropixel_brainwide_01')
    
    this_sub_so = pd.DataFrame() 
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
        trans_from = np.concatenate(([0], df_Trials.loc[block_trans[1:] - 1, 'probabilityLeft'].values))
        
        
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
                
            this_sub_so = pd.concat((this_sub_so, pd.DataFrame(data={
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
        
        
        this_sub_so['after_switch'] = pd.cut(this_sub_so['trial_after_switch'], [-1, N_Trials, N_Trials*2, N_Trials*3, np.inf], labels=['0-10 trials', '10-20 trials', '20-30 trials', '+30 trials'])
        subject_1 = this_sub_so.subject.unique()
        this_sub_so.to_csv('/home/joana/Desktop/data_Pupil_Project/All_sessions_so_' + subject_1[0] + '.csv')
        
        """
        # Make tables
        p_left_df = this_sub.groupby(['time', 'probabilityLeft']).mean()['baseline_subtracted'].reset_index('probabilityLeft')
        stim_side_df = this_sub.groupby(['time', 'Stim_side']).mean()['baseline_subtracted'].reset_index('Stim_side')
        
        # Pivot long form dfs
        p_left_df = pd.pivot(p_left_df, columns=['probabilityLeft'])
        stim_side_df = pd.pivot(stim_side_df, columns=['Stim_side'])
        
        # Concatenate all dfs
        all_sessions_one_animal_mean = pd.concat((p_left_df, stim_side_df), axis=1)
        """
        
    this_sub_so = this_sub_so[~this_sub_so['baseline_subtracted'].isnull()]
    all_sessions_one_animal_mean_so = this_sub_so.groupby(['time', 'probabilityLeft', 'after_switch', 'transition_from', 'contrast', 'Feedback_type', 'Stim_side']).mean()['baseline_subtracted'].reset_index()
    all_sessions_one_animal_mean_so = all_sessions_one_animal_mean_so[~all_sessions_one_animal_mean_so['baseline_subtracted'].isnull()]
    all_sessions_one_animal_mean_so.to_csv('/home/joana/Desktop/data_Pupil_Project/Mean_so_' + subject_1[0] + '.csv')
    
    all_sessions_one_animal_mean_so['subject'] = subject
    all_pupil_so = pd.concat((all_pupil_so, all_sessions_one_animal_mean_so), ignore_index=True)
    all_pupil_so.to_csv('/home/joana/Desktop/data_Pupil_Project/All_animals_Mainen_mean_so.csv')
    
print ('Done')


#%%

# Aligned to Feedback Times 

all_pupil_ft = pd.DataFrame()
all_sessions_one_animal_ft = []

for s, subject in enumerate(subjects):
    print(f'Starting {subject}')
    
    
    # Query sesssions of this subject
    eids = one.search(subject=subject, lab='mainenlab', dataset=['_ibl_leftCamera.dlc.pqt', '_ibl_leftCamera.times.npy'],
                      task_protocol='ephys', project='ibl_neuropixel_brainwide_01')
    
    this_sub_ft = pd.DataFrame() 
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
        trans_from = np.concatenate(([0], df_Trials.loc[block_trans[1:] - 1, 'probabilityLeft'].values))
        
        
        # Alignment (aligned to stimOn_times)
        np_feedbacktimes = np.array(df_Trials['feedback_times'])
        np_times = np.array(times)
      
        for t, trial_start in enumerate(np_feedbacktimes):
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
                
            this_sub_ft = pd.concat((this_sub_ft, pd.DataFrame(data={
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
        
        
        this_sub_ft['after_switch'] = pd.cut(this_sub_ft['trial_after_switch'], [-1, N_Trials, N_Trials*2, N_Trials*3, np.inf], labels=['0-10 trials', '10-20 trials', '20-30 trials', '+30 trials'])
        subject_1 = this_sub_ft.subject.unique()
        this_sub_ft.to_csv('/home/joana/Desktop/data_Pupil_Project/Aligned to feedback times/All_sessions_ft_' + subject_1[0] + '.csv')
        
        """
        # Make tables
        p_left_df = this_sub.groupby(['time', 'probabilityLeft']).mean()['baseline_subtracted'].reset_index('probabilityLeft')
        stim_side_df = this_sub.groupby(['time', 'Stim_side']).mean()['baseline_subtracted'].reset_index('Stim_side')
        
        # Pivot long form dfs
        p_left_df = pd.pivot(p_left_df, columns=['probabilityLeft'])
        stim_side_df = pd.pivot(stim_side_df, columns=['Stim_side'])
        
        # Concatenate all dfs
        all_sessions_one_animal_mean = pd.concat((p_left_df, stim_side_df), axis=1)
        """
        
    this_sub_ft = this_sub_ft[~this_sub_ft['baseline_subtracted'].isnull()]
    all_sessions_one_animal_mean_ft = this_sub_ft.groupby(['time', 'probabilityLeft', 'after_switch', 'transition_from', 'contrast', 'Feedback_type', 'Stim_side']).mean()['baseline_subtracted'].reset_index()
    all_sessions_one_animal_mean_ft = all_sessions_one_animal_mean_ft[~all_sessions_one_animal_mean_ft['baseline_subtracted'].isnull()]
    all_sessions_one_animal_mean_ft.to_csv('/home/joana/Desktop/data_Pupil_Project/Aligned to feedback times/Mean_ft_' + subject_1[0] + '.csv')
    
    all_sessions_one_animal_mean_ft['subject'] = subject
    all_pupil_ft = pd.concat((all_pupil_ft, all_sessions_one_animal_mean_ft), ignore_index=True)
    all_pupil_ft.to_csv('/home/joana/Desktop/data_Pupil_Project/Aligned to feedback times/All_animals_Mainen_mean_ft.csv')
    
print ('Done')







