#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:33:41 2022

@author: Mayo, Julia & Olivier 
"""

"""
Set of functions to deal with dlc data
"""
import logging
import pandas as pd
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.stats import zscore

#from neurodsp.smooth import smooth_interpolate_savgol
#from brainbox.processing import bincount2D
#import brainbox.behavior.wheel as bbox_wheel

logger = logging.getLogger('ibllib')

SAMPLING = {'left': 60,
            'right': 150,
            'body': 30}
RESOLUTION = {'left': 2,
              'right': 1,
              'body': 1}

T_BIN = 0.02  # sec
WINDOW_LEN = 2  # sec
WINDOW_LAG = -0.5  # sec


# For plotting we use a window around the event the data is aligned to WINDOW_LAG before and WINDOW_LEN after the event
def plt_window(x):
    return x + WINDOW_LAG, x + WINDOW_LEN


def insert_idx(array, values):
    idx = np.searchsorted(array, values, side="left")
    # Choose lower index if insertion would be after last index or if lower index is closer
    idx[idx == len(array)] -= 1
    idx[np.where(abs(values - array[idx - 1]) < abs(values - array[idx]))] -= 1
    # If 0 index was reduced, revert
    idx[idx == -1] = 0
    if np.all(idx == 0):
        raise ValueError('Something is wrong, all values to insert are outside of the array.')
    return idx


# Function used fot pupil analysis 
def likelihood_threshold(dlc, threshold=0.9):
    """
    Set dlc points with likelihood less than threshold to nan
    :param dlc: dlc pqt object
    :param threshold: likelihood threshold
    :return:
    """
    features = np.unique(['_'.join(x.split('_')[:-1]) for x in dlc.keys()])
    for feat in features:
        nan_fill = dlc[f'{feat}_likelihood'] < threshold
        dlc[f'{feat}_x'][nan_fill] = np.nan
        dlc[f'{feat}_y'][nan_fill] = np.nan

    return dlc


def get_speed(dlc, dlc_t, camera, feature='paw_r'):
    """
    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param feature: dlc feature to compute speed over
    :return:
    """
    x = dlc[f'{feature}_x'] / RESOLUTION[camera]
    y = dlc[f'{feature}_y'] / RESOLUTION[camera]

    # get speed in px/sec [half res]
    s = ((np.diff(x) ** 2 + np.diff(y) ** 2) ** .5) * SAMPLING[camera]

    dt = np.diff(dlc_t)
    tv = dlc_t[:-1] + dt / 2

    # interpolate over original time scale
    if tv.size > 1:
        ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
        return ifcn(dlc_t)


def get_speed_for_features(dlc, dlc_t, camera, features=['paw_r', 'paw_l', 'nose_tip']):
    """
    Wrapper to compute speed for a number of dlc features and add them to dlc table
    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param features: dlc features to compute speed for
    :return:
    """
    for feat in features:
        dlc[f'{feat}_speed'] = get_speed(dlc, dlc_t, camera, feat)

    return dlc


def get_feature_event_times(dlc, dlc_t, features):
    """
    Detect events from the dlc traces. Based on the standard deviation between frames
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :param features: features to consider
    :return:
    """

    for i, feat in enumerate(features):
        f = dlc[feat]
        threshold = np.nanstd(np.diff(f)) / 4
        if i == 0:
            events = np.where(np.abs(np.diff(f)) > threshold)[0]
        else:
            events = np.r_[events, np.where(np.abs(np.diff(f)) > threshold)[0]]

    return dlc_t[np.unique(events)]


def get_licks(dlc, dlc_t):
    """
    Compute lick times from the tongue dlc points
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :return:
    """
    lick_times = get_feature_event_times(dlc, dlc_t, ['tongue_end_l_x', 'tongue_end_l_y',
                                                      'tongue_end_r_x', 'tongue_end_r_y'])
    return lick_times


def get_sniffs(dlc, dlc_t):
    """
    Compute sniff times from the nose tip
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :return:
    """

    sniff_times = get_feature_event_times(dlc, dlc_t, ['nose_tip_y'])
    return sniff_times


def get_dlc_everything(dlc_cam, camera):
    """
    Get out features of interest for dlc
    :param dlc_cam: dlc object
    :param camera: camera type e.g 'left', 'right'
    :return:
    """

    aligned = True
    if dlc_cam.times.shape[0] != dlc_cam.dlc.shape[0]:
        # logger warning and print out status of the qc, specific serializer django!
        logger.warning('Dimension mismatch between dlc points and timestamps')
        min_samps = min(dlc_cam.times.shape[0], dlc_cam.dlc.shape[0])
        dlc_cam.times = dlc_cam.times[:min_samps]
        dlc_cam.dlc = dlc_cam.dlc[:min_samps]
        aligned = False

    dlc_cam.dlc = likelihood_threshold(dlc_cam.dlc)
    dlc_cam.dlc = get_speed_for_features(dlc_cam.dlc, dlc_cam.times, camera)
    dlc_cam['licks'] = get_licks(dlc_cam.dlc, dlc_cam.times)
    dlc_cam['sniffs'] = get_sniffs(dlc_cam.dlc, dlc_cam.times)
    dlc_cam['aligned'] = aligned

    return dlc_cam


# Function used fot pupil analysis 
def get_pupil_diameter(dlc):
    """
    Estimates pupil diameter by taking median of different computations.
    The two most straightforward estimates: d1 = top - bottom, d2 = left - right
    In addition, assume the pupil is a circle and estimate diameter from other pairs of points
    :param dlc: dlc pqt table with pupil estimates, should be likelihood thresholded (e.g. at 0.9)
    :return: np.array, pupil diameter estimate for each time point, shape (n_frames,)
    """
    diameters = []
    # Get the x,y coordinates of the four pupil points
    top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                                for point in ['top', 'bottom', 'left', 'right']]
    # First compute direct diameters
    diameters.append(np.linalg.norm(top - bottom, axis=0))
    diameters.append(np.linalg.norm(left - right, axis=0))

    # For non-crossing edges, estimate diameter via circle assumption
    for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
        diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5)

    # Ignore all nan runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(diameters, axis=0)


# Function used fot pupil analysis 
def get_smooth_pupil_diameter(diameter_raw, camera, std_thresh=5, nan_thresh=1):
    """
    :param diameter_raw: np.array, raw pupil diameters, calculated from (thresholded) dlc traces
    :param camera: str ('left', 'right'), which camera to run the smoothing for
    :param std_thresh: threshold (in standard deviations) beyond which a point is labeled as an outlier
    :param nan_thresh: threshold (in seconds) above which we will not interpolate nans, but keep them
                       (for long stretches interpolation may not be appropriate)
    :return:
    """
    # set framerate of camera
    if camera == 'left':
        fr = SAMPLING['left']  # set by hardware
        window = 31  # works well empirically
    elif camera == 'right':
        fr = SAMPLING['right']  # set by hardware
        window = 75  # works well empirically
    else:
        raise NotImplementedError("camera has to be 'left' or 'right")

    # Raise error if too many NaN time points, in this case it doesn't make sense to interpolate
    if np.mean(np.isnan(diameter_raw)) > 0.9:
        raise ValueError(f"Raw pupil diameter for {camera} is too often NaN, cannot smooth.")
    # run savitzy-golay filter on non-nan time points to denoise
    diameter_smoothed = smooth_interpolate_savgol(diameter_raw, window=window, order=3, interp_kind='linear')

    # find outliers and set them to nan
    difference = diameter_raw - diameter_smoothed
    outlier_thresh = std_thresh * np.nanstd(difference)
    without_outliers = np.copy(diameter_raw)
    without_outliers[(difference < -outlier_thresh) | (difference > outlier_thresh)] = np.nan
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
    diameter_smoothed = smooth_interpolate_savgol(without_outliers, window=window, order=3, interp_kind='linear')

    # don't interpolate long strings of nans
    t = np.diff(np.isnan(without_outliers).astype(int))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            diameter_smoothed[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff

    return diameter_smoothed


'''
def plot_pupil_diameter_hist(pupil_diameter, cam_times, trials_df, cam='left'):
    """
    Plots histogram of pupil diameter aligned to simulus onset and feedback time.
    :param pupil_diameter: np.array, (smoothed) pupil diameter estimate
    :param cam_times: np.array, camera timestamps
    :param trials_df: pd.DataFrame, with column 'stimOn_times' (time of stimulus onset for each trial) and
                      feedback_times (time of feedback for each trial)
    :param cam: str, camera to use ('body', 'left', 'right') default is 'left'
    :returns: matplotlib.axis
    """
    for align_to, color in zip(['stimOn_times', 'feedback_times'], ['red', 'purple']):
        start_window, end_window = plt_window(trials_df[align_to])
        start_idx = insert_idx(cam_times, start_window)
        end_idx = np.array(start_idx + int(WINDOW_LEN * SAMPLING[cam]), dtype='int64')
        # Per trial norm
        pupil_all = [zscore(list(pupil_diameter[start_idx[i]:end_idx[i]])) for i in range(len(start_idx))]
        pupil_all_norm = [trial - trial[0] for trial in pupil_all]

        pupil_mean = np.nanmean(pupil_all_norm, axis=0)
        pupil_std = np.nanstd(pupil_all_norm, axis=0) / np.sqrt(len(pupil_all_norm))
        times = np.arange(len(pupil_all_norm[0])) / SAMPLING[cam] + WINDOW_LAG

        plt.plot(times, pupil_mean, label=align_to.split("_")[0], color=color)
        plt.fill_between(times, pupil_mean + pupil_std, pupil_mean - pupil_std, color=color, alpha=0.5)
    plt.axvline(x=0, linestyle='--', c='k')
    plt.title(f'Pupil diameter trial avg\n({cam.upper()} cam)')
    plt.xlabel('time [sec]')
    plt.xticks([-0.5, 0, 0.5, 1, 1.5])
    plt.ylabel('z-scored smoothed pupil diameter [px]')
    plt.legend(loc='lower right', title='aligned to')
Footer
'''