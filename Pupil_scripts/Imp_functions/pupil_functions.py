#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:47:40 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from brainbox.behavior.dlc import (get_pupil_diameter, get_smooth_pupil_diameter,
                                   likelihood_threshold)
from one.api import ONE


def load_pupil(eid, view='left', likelihood_thresh=0.9, outlier_thresh=5, one=None):
    if one is None:
        one = ONE()
    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        dlc = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except KeyError:
        print('not all dlc data available')
        return None, None

    # Get smooth pupil
    dlc = likelihood_threshold(dlc, threshold=likelihood_thresh)
    raw_pupil_diameter = get_pupil_diameter(dlc)
    pupil_diameter = get_smooth_pupil_diameter(raw_pupil_diameter, 'left',
                                               std_thresh=outlier_thresh)

    return times, pupil_diameter, raw_pupil_diameter


def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False,
                patch_old_opto=True, one=None):
    one = one or ONE()
    data, _ = one.load_datasets(eid, datasets=[
        '_ibl_trials.stimOn_times.npy', '_ibl_trials.feedback_times.npy',
        '_ibl_trials.goCue_times.npy', '_ibl_trials.probabilityLeft.npy',
        '_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy',
        '_ibl_trials.feedbackType.npy', '_ibl_trials.choice.npy',
        '_ibl_trials.firstMovement_times.npy'])
    trials = pd.DataFrame(data=np.vstack(data).T, columns=[
        'stimOn_times', 'feedback_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice', 'firstMovement_times'])
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    if laser_stimulation:
        trials['laser_stimulation'] = one.load_dataset(eid, dataset='_ibl_trials.laserStimulation.npy')
        try:
            trials['laser_probability'] = one.load_dataset(eid, dataset='_ibl_trials.laserProbability.npy')
            trials['probe_trial'] = ((trials['laser_stimulation'] == 0) & (trials['laser_probability'] == 0.75)
                                     | (trials['laser_stimulation'] == 1) & (trials['laser_probability'] == 0.25)).astype(int)
        except:
            trials['laser_probability'] = trials['laser_stimulation'].copy()
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 0), 'laser_probability'] = 0.25
            trials.loc[(trials['signed_contrast'] == 0)
                       & (trials['laser_stimulation'] == 1), 'laser_probability'] = 0.75

    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1
    if 'firstMovement_times' in trials.columns.values:
        trials['reaction_times'] = trials['firstMovement_times'] - trials['goCue_times']
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']

    return trials


