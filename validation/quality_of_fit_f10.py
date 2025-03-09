"""
This is to quantify the quality of the fit for F10.7. The only data I currently have of relevance is that for Cycle 24 (Cycle 25 is computed in update_cycle_prediction.py).  So, there is no need for a loop.
"""

import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.io import netcdf_file
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoMinorLocator

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# Read observations

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

print(obsfile)

obsdata = json.loads(open(obsfile).read())
Nobs = len(obsdata)

obstime_full = []

fobs10_full = []
fobs10_full_sm = []

for d in obsdata:
    t = np.array(d['time-tag'].split('-'), dtype='int')
    obstime_full.append(datetime.date(t[0], t[1], 15))
    fobs10_full.append(d['f10.7'])
    fobs10_full_sm.append(d['smoothed_f10.7'])

obstime_full = np.array(obstime_full)
fobs10_full = np.array(fobs10_full)
fobs10_full_sm = np.array(fobs10_full_sm)

#------------------------------------------------------------------------------
# extract Cycle 24

tmin = datetime.date(2008,12,15)
tmax = datetime.date(2019,12,15)

idx = np.where((obstime_full >= tmin) & (obstime_full < tmax))
obstime = obstime_full[idx]
fobs10 = fobs10_full[idx]
fobs10_sm = fobs10_full_sm[idx]

for i in np.arange(len(fobs10)):
   print(f"{obstime[i]} {fobs10[i]} {fobs10_sm[i]}")

#------------------------------------------------------------------------------

