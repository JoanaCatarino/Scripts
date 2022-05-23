#!/usr/bin/env python
# @File: Pupil_scripts/Untitled-1.py
# @Author: Niccolo' Bonacchi (@nbonacchi)
# @Date: Monday, May 23rd 2022, 11:46:37 am
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

Created on Thu Dec 16 15:59:18 2021

@author: joana
"""

from pupil_functions import load_pupil, load_trials
import numpy as np
import pandas as pd
from one.api import ONE
from pathlib import Path


def query_sessions(subject, dataset="_ibl_leftCamera.dlc.pqt", protocol: str = "ephys", one=None):
    """Query sessions from ONE with the dlc.pqt file for ephys sessions."""
    one = one or ONE()
    dataset = dataset if isinstance(dataset, list) else [dataset]
    eids = one.search(subject=subject, dataset=dataset, task_protocol=protocol)
    return eids


def save_pupil_size_df(data_frame, subject, folderpath="./"):
    filename = subject + "_pupil_size_df.pkl"
    filepath = Path(folderpath).joinpath(filename)
    data_frame.to_csv(filepath)
    print(f"Saved pupil size dataframe for subject {subject} to {filepath}")


def load_pupil_size_df(subject, folderpath="./"):
    filename = subject + "_pupil_size_df.pkl"
    filepath = Path(folderpath).joinpath(filename)
    if not filepath.exists():
        print(f"Pupil size dataframe for subject {subject} not found. Exiting.")
        return
    return pd.read_pickle(filepath)


def compute_pupil_size_df(subject, eids=None, one=None, save=True, folderpath="./"):
    one = one or ONE()
    eids = eids or query_sessions(
        subject, protocol="ephys", dataset="_ibl_leftCamera.dlc.pqt", one=one
    )
    # Settings
    TIME_BINS = np.arange(-1, 3.2, 0.2)
    BIN_SIZE = 0.2  # seconds
    BASELINE = [1, 0]  # seconds
    N_Trials = 10
    pupil_size = pd.DataFrame()
    results_df = pd.DataFrame()
    results_df_baseline = pd.DataFrame()
    all_pupil_sizes = []
    # Loop over sessions
    for i, eid in enumerate(eids):

        # Get pupil diameter
        times, pupil_diameter, raw_pupil_diameter = load_pupil(eid, one=one)

        # Calculate percentage change
        diameter_perc = (
            (pupil_diameter - np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2))
            / np.percentile(pupil_diameter[~np.isnan(pupil_diameter)], 2)
        ) * 100

        # Get session info
        info = one.get_details(eid)
        subject = info["subject"]
        date = info["start_time"].split("T")[0]

        # Load in trials
        df_Trials = load_trials(eid)

        # Find Block transitions
        block_trans = np.append(
            [0], np.array(np.where(np.diff(df_Trials["probabilityLeft"]) != 0)) + 1
        )
        trans_to = df_Trials.loc[block_trans, "probabilityLeft"]

        # Alignment (aligned to stimOn_times)

        np_stimOn = np.array(df_Trials["stimOn_times"])
        np_times = np.array(times)

        for t, trial_start in enumerate(np_stimOn):
            diameter_1 = np.array([np.nan] * TIME_BINS.shape[0])
            baseline_subtracted = np.array([np.nan] * TIME_BINS.shape[0])
            baseline = np.nanmedian(
                diameter_perc[
                    (np_times > (trial_start - BASELINE[0]))
                    & (np_times < (trial_start - BASELINE[1]))
                ]
            )
            diff_tr = t - block_trans
            last_trans = diff_tr[diff_tr >= 0].argmin()
            trials_since_switch = t - block_trans[last_trans]

            for b, time_bin in enumerate(TIME_BINS):
                diameter_1[b] = np.nanmedian(
                    diameter_perc[
                        (np_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                        & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))
                    ]
                )
                baseline_subtracted[b] = (
                    np.nanmedian(
                        diameter_perc[
                            (np_times > (trial_start + time_bin) - (BIN_SIZE / 2))
                            & (np_times < (trial_start + time_bin) + (BIN_SIZE / 2))
                        ]
                    )
                    - baseline
                )

            pupil_size = pd.concat(
                (
                    pupil_size,
                    pd.DataFrame(
                        data={
                            "diameter": diameter_1,
                            "baseline_subtracted": baseline_subtracted,
                            "eid": eid,
                            "subject": subject,
                            "trial": t,
                            "trial_after_switch": trials_since_switch,
                            "contrast": df_Trials.loc[t, "signed_contrast"],
                            "time": TIME_BINS,
                            "Stim_side": df_Trials.loc[t, "stim_side"],
                            "Feedback_type": df_Trials.loc[t, "feedbackType"],
                            "probabilityLeft": df_Trials.loc[t, "probabilityLeft"],
                        }
                    ),
                )
            )

        pupil_size["after_switch"] = pd.cut(
            pupil_size["trial_after_switch"],
            [-1, N_Trials, N_Trials * 2, np.inf],
            labels=[1, 2, 3],
        )
        all_pupil_sizes.append(pupil_size)
    pupil_size_df = pd.concat(all_pupil_sizes, axis=0)
    if save:
        save_pupil_size_df(pupil_size_df, subject, folderpath)
    return pupil_size_df


def compute_pupil_size_df_for_subjects(
    subjects,
    eids=None,
    protocol="ephys",
    dataset="_ibl_leftCamera.dlc.pqt",
    one=None,
    save=True,
    folderpath="./",
):
    one = one or ONE()
    subjects = subjects if isinstance(subjects, list) else [subjects]
    for subject in subjects:
        eids = query_sessions(
            subject, protocol=protocol, dataset=dataset, one=one
        )
        pupil_size_df = compute_pupil_size_df(subject, eids, one=one, save=False)
        if save:
            save_pupil_size_df(pupil_size_df, subject=subject, folderpath=folderpath)


if __name__ == "__main__":
    one = ONE()
    subject = "ZFM-02368"
    eids = query_sessions(subject, protocol="ephys", dataset="_ibl_leftCamera.dlc.pqt", one=one)
    pupil_size_df = compute_pupil_size_df("ZFM-02368", eids, one=one, save=False)
    save_pupil_size_df(pupil_size_df, subject=subject, folderpath="./")
    # pupil_size_df = load_pupil_size_df(subject)
