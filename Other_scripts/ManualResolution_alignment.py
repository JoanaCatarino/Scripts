#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:14:50 2022

@author: joana
"""

from one.api import ONE
one = ONE()

# Get the probe id for the relevant alignment
probe_id = one.alyx.rest('insertions', 'list', subject='CSHL049', date='2020-01-09', name='probe00')[0]['id'] 

# Find the key of the alignment that has been agreed upon 
traj = one.alyx.rest('trajectories', 'list', probe_insertion=probe_id, provenance='Ephys aligned histology track')[0]
alignment_keys = traj['json'].keys()

# Manually resolve the alignment 
from ibllib.qc.alignment_qc import AlignmentQC

align_key = "2022-06-27T15:37:05_joana.catarino"  # change this to your chosen alignment key
align_qc = AlignmentQC(probe_id, one=one)
align_qc.resolve_manual(align_key)


#If you get a warning saying Alignment for insertion {probe_id} already resolved, channels won't be  updated. 
#To overwrite stored channels with alignment {align_key} set 'force=True’ 
#This means that the alignment for this probe_id has already been set to resolved in the database. 
#If you want to overwrite the alignment stored with the alignment that you have agreed upon, the following code should be used. 
#N.B. This should only be conducted in exceptional cases, for example if the wrong alignment key was entered in the previous steps
# Use:
    
align_qc.resolve_manual(align_key, force=True)