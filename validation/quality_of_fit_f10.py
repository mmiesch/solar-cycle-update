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

deltak = 9

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

#for i in np.arange(len(fobs10)):
#   print(f"{obstime[i]} {fobs10[i]} {fobs10_sm[i]}")

#------------------------------------------------------------------------------

# time of available observations in decimal year
nobs = len(obstime)
odect = np.zeros(nobs)
for i in np.arange(nobs):
    dt = datetime.datetime(obstime[i].year,obstime[i].month,obstime[i].day)
    odect[i] = Time(dt).to_value('decimalyear')

# time of predictions in decimal year
ptime = obstime
nn = len(ptime)
pdect = np.zeros(nn)
for i in np.arange(nn):
    dt = datetime.datetime(ptime[i].year,ptime[i].month,ptime[i].day)
    pdect[i] = Time(dt).to_value('decimalyear')

tstart = odect[0]
tobs = (odect - tstart)*12
tpred = (pdect - tstart)*12

pmonth = np.rint(tobs[-1]).astype(np.int32)
print(f"Prediction month = {pmonth}")

#------------------------------------------------------------------------------
# do the curve fit

ffit = curve_fit(u.fclette10,tobs,fobs10,p0=(170.0,0.0))
f10 = u.fclette10(tpred,ffit[0][0],ffit[0][1])

if (deltak > 0) and (pmonth > (deltak + 23)):
  k2 = pmonth - deltak
  ffit2 = curve_fit(u.fclette10,tobs[0:k2],fobs10[0:k2],p0=(170.0,0.0))
  f102 = u.fclette10(tpred,ffit2[0][0],ffit2[0][1])
  f10 = 0.5*(f10+f102)

#--------------------

print('\n\nMAE F10.7')

resid = fobs10 - f10

MAE = np.mean(np.abs(resid))
bias = np.mean(resid)

eidx = np.where(fobs10_sm > 0.0)
err = fobs10[eidx] - fobs10_sm[eidx]
sigma = np.mean(np.abs(err))

resid_sm = fobs10_sm[eidx] - f10[eidx]
MAE_sm = np.mean(np.abs(resid_sm))

MAE_metric = MAE/sigma
bias_metric = bias/sigma
MAE_metric_sm = MAE_sm/sigma

print(f'MAE = {MAE:.1f} {MAE_metric:.1f}, bias = {bias:.1f} {bias_metric:.1f}, sigma = {sigma:.1f}, MAE_sm = {MAE_sm:.1f} {MAE_metric_sm:.1f}') 

#------------------------------------------------------------------------------
# plot as a sanity check

plt.plot(obstime,fobs10,'black',label='Observed',linewidth=2)
plt.plot(ptime,f10,'blue',label='Predicted',linewidth=2)
plt.plot(obstime,fobs10_sm,'red',label='Smoothed',linewidth=2)

plt.show()
