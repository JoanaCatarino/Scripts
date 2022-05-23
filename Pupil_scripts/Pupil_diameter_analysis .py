"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

Created on Thu Dec 16 15:59:18 2021

@author: joana
"""

from pupil_functions import load_pupil, load_trials
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from scipy.stats import zscore
from functions import (figure_style)
from os.path import join
from one.api import ONE
one = ONE()


# Settings
TIME_BINS = np.arange(-1, 3.2, 0.2)
BIN_SIZE= 0.2 #seconds
BASELINE = [1, 0] #seconds
N_Trials = 10
pupil_size = pd.DataFrame()
results_df = pd.DataFrame()
results_df_baseline = pd.DataFrame()
#Fig_path =


# Query sessions
eids = one.search(subject='ZFM-02368', dataset=['_ibl_leftCamera.dlc.pqt'], task_protocol='ephys')
    #eids = [eids[0]] #When we only want to run 1 specific animal

# Loop over sessions
for i, eid in enumerate(eids):

    # Get pupil diameter
    times, pupil_diameter, raw_pupil_diameter = load_pupil(eid, one=one)

    # Calculate percentage change
    diameter_perc = ((pupil_diameter - np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2))
                     / np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2)) * 100

    # Get session info
    info = one.get_details(eid)
    subject = info['subject']
    date = info['start_time'].split('T')[0]

    # Load in trials
    df_Trials = load_trials(eid)

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

    pupil_size['after_switch'] = pd.cut(pupil_size['trial_after_switch'], [-1, N_Trials, N_Trials*2, np.inf], labels=[1, 2, 3])

#%%

    # Plot pupil size per contrast for all block types

    pupil_size = pupil_size.reset_index(drop=True)

    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF', '#FF9561']


    # Contrast = -1
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -1)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -1' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg1.png'))

    # Contrast = -0.25
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.25)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.25' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg025.png'))

    # Contrast = -0.125
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.125)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.125' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg0125.png'))

    # Contrast = -0.0625
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == -0.0625)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = -0.0625' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_neg00625.png'))

    # Contrast = +0
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0) & (pupil_size['contrast'] == -0)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 right side' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_0.png'))

    # Contrast = +1
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 1)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos1.png'))

    # Contrast = +0.25
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.25)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos025.png'))

    # Contrast = +0.125
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.125)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos0125.png'))

    # Contrast = +0.0625
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='probabilityLeft', data=pupil_size[(pupil_size['contrast'] == 0.0625)], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrast_pos00625.png'))

 #%%

    # Plot pupil size per block trial with stim appearing on left and right (per constrast)

    pupil_size = pupil_size.reset_index(drop=True)

    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF']

    # Contrast 1 probability 0.2
    full_contrast_l = pupil_size[((np.abs(pupil_size['contrast']) == 1) & (pupil_size['probabilityLeft'] == 0.2))]
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=full_contrast_l, legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 1 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 1 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

 # -----------------


    # Contrast 0.25 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.25 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5)) ], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.25 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8)) ], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.25 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

# -----------------

    # Contrast 0.125 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.125 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.125 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.125 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis


# -----------------

    # Contrast 0.0625 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.0625 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.0625 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0.0625 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

 # -----------------

    # Contrast 0 probability 0.2
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.2' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0 probability 0.5
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.5' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0 probability 0.8
    f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8))], legend='full', ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 0 and probability 0.8' '    ' f'{subject}, {date}', ylim=[-25, 25])
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

#%%
    # CORRECT vs INCORRECT TRIALS

    pupil_size = pupil_size.reset_index(drop=True)

    dpi = figure_style()
    colors = ['#47BFD1', '#C89AFF']


    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10,4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 1 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 1 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 1 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')


 # ----------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0.25 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.25 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.25 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.25) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')


# ------------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0.125 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.125 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.125 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.125) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')

# ------------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0.0625 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0.0625 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0.0625 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0.0625) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')

# ------------

    f, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(10, 4), sharey=False, sharex=False, dpi=800)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.6)


    # Contrast 0 probability 0.2 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - CORRECT', ylim=[-25, 25])
    ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.2 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax2, estimator=np.median, palette = sns.color_palette(colors))
    ax2.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Probability 0.2 - INCORRECT', ylim=[-25, 25])
    ax2.plot([0, 0], ax2.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.5 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == 1))], legend=None, ci=68, ax=ax3, estimator=np.median, palette = sns.color_palette(colors))
    ax3.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - CORRECT', ylim=[-25, 25])
    ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.5 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.5) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax4, estimator=np.median, palette = sns.color_palette(colors))
    ax4.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.5 - INCORRECT', ylim=[-25, 25])
    ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='black', label='Stim Onset')

    # Contrast 0 probability 0.8 - CORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == 1))], legend='full', ci=68, ax=ax5, estimator=np.median, palette = sns.color_palette(colors))
    ax5.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - CORRECT', ylim=[-25, 25])
    ax5.plot([0, 0], ax5.get_ylim(), ls='--', color='black', label='Stim Onset')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False) # Put a legend to the right of the current axis

    # Contrast 0 probability 0.8 - INCORRECT TRIALS
    sns.lineplot(x='time', y='baseline_subtracted', hue='Stim_side', data=pupil_size[((pupil_size['contrast'] == -0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1)) | ((pupil_size['contrast'] == 0) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['Feedback_type'] == -1))], legend=None, ci=68, ax=ax6, estimator=np.median, palette = sns.color_palette(colors))
    ax6.set(xlabel='Time relative to StimON (s)', ylabel='', title=f' Probability 0.8 - INCORRECT', ylim=[-25, 25])
    ax6.plot([0, 0], ax6.get_ylim(), ls='--', color='black', label='Stim Onset')


#%%

    #  BLOCK SWITCH


# plot only the contrast in the side it appears more often

pupil_size = pupil_size.reset_index(drop=True)

dpi = figure_style()
colors = ['#F96E46', '#8E4162', '#1DC7C6']

# Contrast = 1
df_slice = pupil_size[((pupil_size['Stim_side'] == -1) & (pupil_size['probabilityLeft'] == 0.8) & (pupil_size['contrast'] == 1))
                      | ((pupil_size['Stim_side'] == 1) & (pupil_size['probabilityLeft'] == 0.2) & (pupil_size['contrast'] == 1))].reset_index()


f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=dpi)
lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='after_switch', data=df_slice, legend='full',
                        ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1' '    ' f'{subject}, {date}', xticks=[-1, 0, 1, 2, 3])
plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=['0-20 trials', "", '20-40 trials', '40+ trials', 'Stim Onset'], frameon=False) # Put a legend to the right of the current axis
sns.despine(trim=True)


# Contrast = 0.25
df_slice = pupil_size[((pupil_size['Stim_side'] == -0.25) & (pupil_size['probabilityLeft'] == 0.8))
                      | ((pupil_size['Stim_side'] == 0.25) & (pupil_size['probabilityLeft'] == 0.2))].reset_index()


f, (ax1) = plt.subplots(1, 1, sharey=True, sharex=False, dpi=800)
lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='after_switch', data=df_slice, legend='full',
                        ci=68, ax=ax1, estimator=np.median, palette = sns.color_palette(colors))
ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f' Contrast = 1' '    ' f'{subject}, {date}', xticks=[-1, 0, 1, 2, 3])
plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),  labels=['0-20 trials', '20-40 trials', '40+ trials'], frameon=False) # Put a legend to the right of the current axis
sns.despine(trim=True)






dpi = figure_style()
df_slice = pupil_size[((pupil_size['Stim_side'] == -1) & (pupil_size['probabilityLeft'] == 0.8))
                      | ((pupil_size['Stim_side'] == 1) & (pupil_size['probabilityLeft'] == 0.2))].reset_index()
#df_slice.loc[df_slice['after_switch'] == 1, 'after_switch'] = '0-20'
#df_slice.loc[df_slice['after_switch'] == 2, 'after_switch'] = '20-40'
#df_slice.loc[df_slice['after_switch'] == 3, 'after_switch'] = '40+'
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
sns.lineplot(x='time', y='baseline_subtracted', data=df_slice, hue='after_switch', ax=ax1,
             ci=68)
ax1.set(xlabel='Time (s)', ylabel='Pupil size (%)', xticks=[-1, 0, 1, 2, 3])
ax1.legend(title='', frameon=False)
plt.tight_layout()
sns.despine(trim=True)




























#%%

 #------------------- PLOTS FOR BLOCK CHANGE (0.5 + 0.2 + 0.8) -------------------------


#  Divide session into blocks depending on the probability left

    results_df = results_df_baseline.append(pd.DataFrame(data={
        'left_20': pupil_size[pupil_size['probabilityLeft'] == 0.2].groupby('time').median()['diameter'],
        'left_80': pupil_size[pupil_size['probabilityLeft'] == 0.8].groupby('time').median()['diameter'],
        'left_50': pupil_size[pupil_size['probabilityLeft'] == 0.5].groupby('time').median()['diameter']}))





#%%

# -------------- SIMPLE PLOTS -----------

    pupil_size = pupil_size.reset_index(drop=True)

# Plot all Contrasts
    f, ax1 = plt.subplots(1, 1, dpi=500)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='contrast', data=pupil_size, legend='full', ci=68, estimator=np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'All Contrasts' '    ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Put a legend to the right of the current axis
    plt.tight_layout()
    #plt.savefig(join(Fig_path, f'{subject}_{date}_All_contrasts.png'))

# Plot only Contrasts on the RIGHT
    f, ax1 = plt.subplots(1, 1, dpi=500)
    pupil_size['contrast_abs'] = pupil_size['contrast'].abs()
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='contrast_abs', data=pupil_size, legend='full', ci=68, estimator=np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Contrasts on the Right' '    ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrasts_Right.png'))

# Plot only Contrasts on the LEFT
    f, ax1 = plt.subplots (1, 1, dpi=500)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue= 'contrast', data=pupil_size[pupil_size['Stim_side']==-1], ci=68, estimator= np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Contrasts on the Left' '    ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Contrasts_Left.png'))

# Plot correct trials with contrasts on the right
    f, ax1 = plt.subplots (1, 1, dpi=500)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='contrast', data=pupil_size[(pupil_size['Feedback_type']==1) & (pupil_size['Stim_side']== 1)], ci=68, estimator=np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Correct on the Right' '   ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Correct_Right.png'))

# Plot correct trials with contrasts on the left
    f, ax1 = plt.subplots (1, 1, dpi=500)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='contrast', data=pupil_size[(pupil_size['Feedback_type']==1) & (pupil_size['Stim_side']==-1)], ci=68, estimator=np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Correct on the Left''   ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Correct_Left.png'))

# Plot correct trials for all contrasts
    f, ax1 = plt.subplots (1, 1, dpi=500)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='contrast', data=pupil_size[pupil_size['Feedback_type']==1], ci=68, estimator=np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Correct all contrasts''   ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Correct_AllContrats.png'))

# Plot correct trials per contrast individually (probab.Left = 0.5 + 0.2 + 0.8)
    f, ax1 = plt.subplots (1, 1, dpi=700)
    lineplt = sns.relplot(x='time', y='baseline_subtracted', hue='Feedback_type', col = 'contrast', kind='line', row = 'probabilityLeft',  #col_wrap = 3,
                data=pupil_size, ci=68, estimator=np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Correct all contrasts''   ' f'{subject}, {date}')
    for ax in lineplt.axes.flat:
        ax.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    plt.tight_layout()
    #plt.savefig(join(Fig_path, f'{subject}_{date}_Correct_perContrast.png'))



#%%
# Alignment (aligned to feedback_times)

    np_stimOn = np.array(df_Trials['stimOn_times'])
    np_times = np.array(times)

    for t, trial_start in enumerate(np_stimOn):
        diameter_1 = np.array([np.nan] * TIME_BINS.shape[0])
        baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
        baseline = np.nanmedian(diameter_perc [(np_times > (trial_start - BASELINE[0])) & (np_times < (trial_start - BASELINE[1]))])

        for b, time_bin in enumerate (TIME_BINS):
            diameter_1[b] = np.nanmedian(diameter_perc [(np_times > (trial_start + time_bin) - (BIN_SIZE / 2)) & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
            baseline_subtracted[b] = np.nanmedian(diameter_perc[(np_times > (trial_start + time_bin) - (BIN_SIZE / 2)) & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
            pupil_size = pupil_size.append(pd.DataFrame(data={'diameter': diameter_1,
                                                              'baseline_subtracted': baseline_subtracted,
                                                              'eid': eid,
                                                              'subject': subject,
                                                              'trial': t,
                                                              'contrast': df_Trials.loc[t, 'signed_contrast'],
                                                              'time': TIME_BINS,
                                                              'Stim_side':df_Trials.loc[t, 'stim_side'],
                                                              'Feedback_type':df_Trials.loc[t, 'feedbackType'],
                                                              'probabilityLeft':df_Trials.loc[t, 'probabilityLeft']}))

# Plot correct trials per contrast individually (probab.left = 0.5 + 0.2 + 0.8)
    pupil_size = pupil_size.reset_index(drop=True)
    f, ax1 = plt.subplots (1, 1, dpi=700)
    lineplt = sns.relplot(x='time', y='baseline_subtracted', hue='Feedback_type', col = 'contrast', kind='line', row = 'probabilityLeft',  #col_wrap = 3,
                data=pupil_size, ci=68, estimator=np.median, palette='colorblind')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Correct all contrasts''   ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    for ax in lineplt.axes.flat:
        ax.axvline(x=0, color = 'red', label = 'feedback', linestyle = 'dashed')



#%%
#-------------------------- EXTRA stuff -------------------------------------------------

# PLot figure for Contrast = 1
    pupil_size = pupil_size.reset_index(drop = True)
    f, ax1 = plt.subplots(1, 1, dpi=500)
    lineplt = sns.lineplot(x='time', y='baseline_subtracted',data=pupil_size[pupil_size['contrast']==1], color='darkturquoise')
    ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'Contrast = 1' '    ' f'{subject}, {date}')
    plt.axvline(x = 0, color = 'black', label = 'Stim Onset', linestyle='dashed')
    plt.tight_layout()


# PLot Contrasts using log10
pupil_size = pupil_size.reset_index(drop=True)
pupil_size['contrast_abs'] = pupil_size['contrast'].abs()
pupil_size['contrast_abs_log'] = pupil_size['contrast_abs'].copy()
pupil_size.loc[pupil_size['contrast_abs_log'] == 0, 'contrast_abs_log'] = 0.01
pupil_size['contrast_abs_log'] = np.log10(pupil_size['contrast_abs_log'])
f, ax1 = plt.subplots(1, 1, dpi=500)
lineplt = sns.lineplot(x='time', y='baseline_subtracted', hue='contrast_abs_log', data=pupil_size, legend='full', ci=68, estimator=np.median, palette='flare')
ax1.set(xlabel='Time relative to StimON (s)', ylabel='Pupil size (%)', title=f'All Contrasts' '    ' f'{subject}, {date}')
ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', label='Stim Onset')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Put a legend to the right of the current axis
plt.tight_layout()

 '''
 # Plot example
 f, ax1 = plt.subplots(1, 1, dpi=500)
 ax1.plot(times[1:1000], diameter_perc[1:1000], color='#7EB09B')
 ax1.set(xlabel='Time (s)', ylabel='Pupil diameter (%)', title=f'{subject}, {date}')
 plt.tight_layout()
     # to plot only a part of the session we need to use time[:1000] , the 'diameter perc' needs to match this value so: diameter_perc[:1000]
 '''


#Nice sns palette = 'flare', 'husl'
#Nice sns palette = 'colorblind'
#To place the legend on top of the plot with a certain fontsize = ax1.legend(loc='upper right', fontsize=9)
#To save the plots:   plt.savefig(join(fig_path, f'{nickname}_pupil_opto.png'))
#                     plt.savefig(join(fig_path, f'{nickname}_pupil_opto.pdf'))

#Plot only higher and lower contrast after block change aligned to 'feedback times' and 'stim ON'!!!


# %%

dpi = figure_style()
df_slice = pupil_size[((pupil_size['Stim_side'] == -1) & (pupil_size['probabilityLeft'] == 0.8))
                      | ((pupil_size['Stim_side'] == 1) & (pupil_size['probabilityLeft'] == 0.2))].reset_index()
#df_slice.loc[df_slice['after_switch'] == 1, 'after_switch'] = '0-20'
#df_slice.loc[df_slice['after_switch'] == 2, 'after_switch'] = '20-40'
#df_slice.loc[df_slice['after_switch'] == 3, 'after_switch'] = '40+'
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=dpi)
sns.lineplot(x='time', y='baseline_subtracted', data=df_slice, hue='after_switch', ax=ax1,
             ci=68)
ax1.set(xlabel='Time (s)', ylabel='Pupil size (%)', xticks=[-1, 0, 1, 2, 3])
ax1.legend(title='', frameon=False)
plt.tight_layout()
sns.despine(trim=True)




