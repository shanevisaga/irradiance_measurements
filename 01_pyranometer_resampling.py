#!/usr/bin/env python
# coding: utf-8
'''
01_pyranometer_resampling
This code calls daily files containing pyranometer VDC values
and perform:
a) calibrate to get irradiance data
b) quality check to remove negative irradiance values
c) resample at 1 minute and 10 minute
d) save the processed files (each file contains 1 month worth of data)

By: Shane Visaga
updated: February 20 2023
'''
import glob
import os

import numpy as np
import pandas as pd
import pytz

tz = pytz.timezone("Asia/Manila")

path = 'pyranometer/' # use your path
for m in np.arange(1, 13, 1):
#for m in np.arange(3, 4, 1):
    all_files = glob.glob(os.path.join(path, f'{m:02}*.csv'))
    obs = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    #calibration
    #obs['CMP22_Total_Solar'] = obs['CMP22_Total_Solar_VDC'] * 109.62
    obs['SPN1_Total_Solar'] = obs['SPN1_Total_Solar_VDC'] * 1000
    obs['SPN1_Diff_Solar'] = obs['SPN1_Diff_Solar_VDC'] * 1000
    obs['CGR4_Temp'] = -245.69 + (2.3554* obs['CGR4_Temp_OHM']) + (0.0010138* ((obs['CGR4_Temp_OHM']**2)))
    obs['CGR4_IR'] = ((92.837*obs['CGR4_IR_VDC'])  - 22.815) + ((5.67*(1E-8)) * ((obs['CGR4_Temp'] + 273.16)**4))

    obs = obs[['Time', 'SPN1_Total_Solar', 'SPN1_Diff_Solar','CGR4_IR']]

    #localizing time to UTC
    obs['Time'] = pd.to_datetime(obs['Time']).dt.tz_localize("utc")

    #valid values only; no negatives
    obs = obs[obs['SPN1_Total_Solar'] >= 0]

    obs = obs.set_index('Time')


    #resample to 1 minute
    obs1 = obs.resample('1min').mean()
    obs1.to_csv(f'processed/{m:02}_1min_resample.csv')

    #resample to 10 minute
    obs10 = obs.resample('10min').mean()
    obs10.to_csv(f'processed/{m:02}_10min_resample.csv')
    print(f'processed/{m:02}_10min_resample.csv')



