# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:52:42 2022

@author: Joana
"""

from one.api import ONE


# from reproducible_ephys_functions import query
import brainbox.io.one as bbone
from ibllib.atlas import AllenAtlas


ba = AllenAtlas()
one = ONE(base_url='https://alyx.internationalbrainlab.org')



T_BIN = 0.02  # time bin size in seconds
freq=1/T_BIN; # frequency in Hz
max_cluster=10;
eid='6a601cc5-7b79-4c75-b0e8-552246532f82' #
currentprobe='probe00'

spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, probe=currentprobe, brain_atlas=ba)