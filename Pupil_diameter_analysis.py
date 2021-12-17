 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:59:18 2021

@author: joana
"""

from pupil_functions import load_pupil, load_trials
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy as sns
from one.api import ONE
one = ONE()

# Query sessions
eids = one.search(subject='ZFM-02368', dataset=['_ibl_leftCamera.dlc.pqt'],
                  task_protocol='ephys')

# Loop over sessions
for i, eid in enumerate(eids):

    # Get pupil diameter
    times, pupil_diameter, raw_pupil_diameter = load_pupil(eid, one=one)

    # Percentage change
    diameter_perc = ((pupil_diameter - np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2))
                     / np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2)) * 100
    
    # Table for %changes in pupil diameter across time
    df_Pupil = pd.DataFrame(times, columns=['Time'])
    df_Pupil['% change']=diameter_perc
    
    # Get session info
    info = one.get_details(eid)
    subject = info['subject']
    date = info['start_time'][:10]

    # Load in trials
    df_Trials = load_trials(eid)

    # Plot example
    f, ax1 = plt.subplots(1, 1, dpi=500)
    ax1.plot(times, diameter_perc, color='#BE97C6')
    ax1.set(xlabel='Time (s)', ylabel='Pupil diameter (%)', title=f'{subject}, {date}')
    plt.tight_layout()
        # to plot only a part of the session we need to use time[:1000] , the diameter perc needs to match this value so: diameter_perc[:1000]

    # Choose event to plot 
    
    f, ax2 = plt.subplots(1, 1, dpi=500)
    

    

#%% Random stuff 

    # Table for incorrect trials   
  IncorrectTrials = df_Trials['feedbackType']==-1
    
  # Get trial triggered baseline subtracted pupil diameter (Guido's script)
        for t, trial_start in enumerate(trials['stimOn_times']):
            this_diameter = np.array([np.nan] * TIME_BINS.shape[0])
            baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
            baseline = np.nanmedian(diameter_perc[(video_times > (trial_start - BASELINE[0]))
                                                  & (video_times < (trial_start - BASELINE[1]))])
            for b, time_bin in enumerate(TIME_BINS):
                this_diameter[b] = np.nanmedian(diameter_perc[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))])
                baseline_subtracted[b] = np.nanmedian(diameter_perc[
                    (video_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                    & (video_times < (trial_start + time_bin) + (BIN_SIZE / 2))]) - baseline
            pupil_size = pupil_size.append(pd.DataFrame(data={
                'diameter': this_diameter, 'baseline_subtracted': baseline_subtracted, 'eid': eid,
                'subject': nickname, 'trial': t, 'contrast': trials.loc[t, 'signed_contrast'],
                'sert': subjects.loc[i, 'sert-cre'], 'laser': trials.loc[t, 'laser_stimulation'],
                'laser_prob': trials.loc[t, 'laser_probability'], 'laser_block_progress': trials.loc[t, 'block_progress'],
                'time': TIME_BINS}))

    


#%% plot ---> use matplotlib instead of seaborn I guess

sns.set_theme(style=“whitegrid”) #I can change this for ticks
#palette = sns.color_palette(“tab10",2)
# Plot the responses of the 5-HT DRN neurons according to x event
sns.lineplot(x=“times_subtracted”, y=“zdFF”,
             data=b, color= ‘darkgreen’, linewidth = 0.25, alpha = 0.8, units=“trial”, estimator=None)#,hue=“feedbackType”,palette=palette)
#sns.lineplot(x=“times_subtracted”, y=(“zdFF”),
#             data=b, color= ‘darkgreen’, linewidth = 0.25, alpha = 0.8)
plt.axvline(x=0, color = “palevioletred”, alpha=0.75, linewidth = 3, label = “stimOn_times”)
plt.axhline(y=0, color = “gray”, alpha=0.75, linewidth = 1.5)
plt.title(“TrainingCW_S12 - 5-HT DRN signal M1 13Sep2021", fontsize=15)
plt.legend(loc=“upper right”, fontsize=12)
plt.xlabel(“seconds”, fontsize=12)
plt.ylabel(“zdFF”, fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.xlim(-3,3)
#plt.ylim(-2,3)
sns.set(rc={‘figure.figsize’:(15,10)})
plt.savefig(‘…’)
plt.show()

#748288
# Table for incorrect trials   
df_IncorrectTrials = trials[trials['feedbackType']==-1]  
    