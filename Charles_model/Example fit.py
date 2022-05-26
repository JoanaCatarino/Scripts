# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:02:41 2022

@author: Joana
"""
'''
Hackathon 2022May22

(base) Kcenias-MacBook-Pro-2:~ kcenia$ cd /Users/kcenia/Desktop/IBL/coding
(base) Kcenias-MacBook-Pro-2:coding kcenia$ cd ../
(base) Kcenias-MacBook-Pro-2:IBL kcenia$ cd Coding
(base) Kcenias-MacBook-Pro-2:Coding kcenia$ conda develop ./behavior_models
added /Users/kcenia/Desktop/IBL/Coding/behavior_models
completed operation for: /Users/kcenia/Desktop/IBL/Coding/behavior_models
(base) Kcenias-MacBook-Pro-2:Coding kcenia$ 
conda activate iblenv #do this 1st
conda develop ./behavior_models #I guess
#conda install sobol_seq
pip install sobol-seq==0.1.1
'''


#compute sinl function 

#%%

# from pupil_functions import load_pupil, load_trials
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from scipy.stats import zscore
from os.path import join
from one.api import ONE
one = ONE()


#%%
def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    if one is None:
        one = ONE()
    trials = pd.DataFrame()
    if laser_stimulation:
        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice'],
         trials['feedback_times'], trials['firstMovement_times'], trials['laser_stimulation'],
         trials['laser_probability']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 'trials.feedback_times', 'trials.firstMovement_times',
                                                 '_ibl_trials.laser_stimulation',
                                                 '_ibl_trials.laser_probability'])
        if trials.shape[0] == 0:
            return
        if trials.loc[0, 'laser_stimulation'] is None:
            trials = trials.drop(columns=['laser_stimulation'])
        if trials.loc[0, 'laser_probability'] is None:
            trials = trials.drop(columns=['laser_probability'])
    else:
#        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
#          trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
#          trials['feedbackType'], trials['choice'], trials['firstMovement_times'],
#          trials['feedback_times']) = one.load(
#                              eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
#                                                  'trials.goCue_times', 'trials.probabilityLeft',
#                                                  'trials.contrastLeft', 'trials.contrastRight',
#                                                  'trials.feedbackType', 'trials.choice',
#                                                  'trials.firstMovement_times',
#                                                  'trials.feedback_times'])
        try:
            trials = one.load_object(eid, 'trials') #210810 Updated by brandon due to ONE update
        except:
            return {}
            
            
            
    if len(trials['probabilityLeft']) == 0: # 210810 Updated by brandon due to ONE update
        return
#     if trials.shape[0] == 0:
#         return
#     trials['signed_contrast'] = trials['contrastRight']
#     trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
#     trials['correct'] = trials['feedbackType']
#     trials.loc[trials['correct'] == -1, 'correct'] = 0
#     trials['right_choice'] = -trials['choice']
#     trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
#     trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
#     trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
#                'stim_side'] = 1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
#                'stim_side'] = -1
    assert np.all(np.logical_xor(np.isnan(trials['contrastRight']),np.isnan(trials['contrastLeft'])))
    
    trials['signed_contrast'] = np.copy(trials['contrastRight'])
    use_trials = np.isnan(trials['signed_contrast'])
    trials['signed_contrast'][use_trials] = -np.copy(trials['contrastLeft'])[use_trials]
    trials['correct'] = trials['feedbackType']
    use_trials = (trials['correct'] == -1)
    trials['correct'][use_trials] = 0
    trials['right_choice'] = -np.copy(trials['choice'])
    use_trials = (trials['right_choice'] == -1)
    trials['right_choice'][use_trials] = 0
    trials['stim_side'] = (np.isnan(trials['contrastLeft'])).astype(int)
    use_trials = (trials['stim_side'] == 0)
    trials['stim_side'][use_trials] = -1
#     if 'firstMovement_times' in trials.columns.values:
    trials['reaction_times'] = np.copy(trials['firstMovement_times'] - trials['goCue_times'])
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials
#%%
# Query sessions
eids = one.search(subject='ZFM-02368', dataset=['_ibl_leftCamera.dlc.pqt'], task_protocol='ephys')
    #eids = [eids[0]] #When we only want to run 1 specific animal
eid = eids[1]
# Get session info
info = one.get_details(eid)
subject = info['subject']
date = info['start_time'][:10]
# Load in trials
df_Trials = load_trials(eid)
    
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAction
from models.optimalBayesian import optimal_Bayesian
savepath= 'C:/Users/Asus/Desktop/IBL/Scripts/Charles_model/Example_fit_model' # it works with either / or \\
actions = df_Trials.choice
stimuli = df_Trials.signed_contrast
stim_side = df_Trials.stim_side
prior_model = exp_prevAction(savepath, [eid], subject, actions, stimuli, stim_side)
prior_model.load_or_train(remove_old=False)
prior_model_output =prior_model.compute_signal(signal=['prior','prediction_error','score'],
                                            parameter_type ='posterior_mean',
                                            act=actions,
                                            stim=stimuli,
                                            side=stim_side)
prior_model_signal =  prior_model_output['prior']
prior_model_accuracy =  prior_model_output['accuracy']
prior_model_pred_error =  prior_model_output['prediction_error']

#%%
"""
take out trials where prior_model_pred_error is 0
"""
prior_model_pred_error =  prior_model_output['prediction_error']
prior_model_pred_error.shape
pred_error = prior_model_pred_error.squeeze().mean(0)
plt.figure()
plt.plot(pred_error)
plt.show()
#%%
plt.figure()
plt.plot(prior_model_signal)
plt.show()







