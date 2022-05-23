# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:25:02 2021

@author: joana
"""

"""
Post adlib CA water for all mice of specified user on Friday, Saturday and Sunday

Guido Meijer and Joana Catarino
"""

from one.api import ONE
import datetime
import pandas as pd
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# Days to input Citric Acid water, including today (for the weekend on Friday input 3 days)
DAYS = 11
USER = 'joana.catarino'

# Get vector of dates
dates = []
for i in range(DAYS):
    dates.append((datetime.datetime.today()
                  + datetime.timedelta(days=i)).strftime('%Y-%m-%dT%H:%M'))

# Query subjects
subjects = pd.DataFrame(one.alyx.get(
    f'/subjects?&alive=True&water_restricted=True&responsible_user={USER}'))
sub = subjects['nickname'].unique()

for s in sub:
    userresponse = input(f'Post adlib Water to mouse {s} for {DAYS} days? (y/n) ')
    if userresponse.lower() == 'y':
        for dat in dates:

            # settings
            wa_ = {
                'subject': s,
                'date_time': dat,
                'water_type': 'Water',
                'water_administered': 1,
                'user': 'joana.catarino',
                'adlib': True}

            # post on Alyx
            rep = one.alyx.rest('water-administrations', 'create', data=wa_)
            print('POSTED adlib Water to mouse %s on date %s.' % (s, dat))