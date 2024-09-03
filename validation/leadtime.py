"""
The intention of this function is to overplot three curves:
* the smoothed SSN at time t
* the prediction made at t - 1 year
* the prediction made at t - 2 years

I could add other curves to it, like the unsmoothed obs, but let's start with these as the main goal.
"""
import datetime
import json
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.io import netcdf_file
from scipy.signal import savgol_filter

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# Preliminaries

# the two lead times you want to plot, in months

lead_times = [12, 24]

# month for averaging
deltak = 9

#------------------------------------------------------------------------------
# Define files

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

indir, outdir, valdir = u.get_data_dirs()

# Figure product
outfig_ssn = outdir + '/leadtime_ssn.png'
outfig_f10 = outdir + '/leadtime_f10.png'

# optionally archive previous predictions
archive = True

# official start time of cycle 25 from SIDC, in decimal year
tstart = 2019.96

#------------------------------------------------------------------------------
# read observations

# ignore observations before this date in the fit
# official start of Cycle 25
tmin = datetime.date(2019,12,15)

obsdata = json.loads(open(obsfile).read())
Nobs = len(obsdata)

obstime = []
ssn = []
ssn_sm = []
fobs10 = []
fobs10_sm = []

for d in obsdata:
    t = np.array(d['time-tag'].split('-'), dtype='int')
    thist = datetime.date(t[0], t[1], 15)
    if thist >= tmin:
        obstime.append(datetime.date(t[0], t[1], 15))
        ssn.append(d['ssn'])
        ssn_sm.append(d['smoothed_ssn'])
        fobs10.append(d['f10.7'])
        fobs10_sm.append(d['smoothed_f10.7'])

obstime = np.array(obstime)
ssn = np.array(ssn)
ssn_sm = np.array(ssn_sm)
fobs10 = np.array(fobs10)
fobs10_sm = np.array(fobs10_sm)

#------------------------------------------------------------------------------

# time of available observations in decimal year
nobs = len(obstime)
odect = np.zeros(nobs)
for i in np.arange(nobs):
    dt = datetime.datetime(obstime[i].year,obstime[i].month,obstime[i].day)
    odect[i] = Time(dt).to_value('decimalyear')

#------------------------------------------------------------------------------
# read average residuals for this fit type

lab = 'panel2_d9'

if resfile is None:
    resfile = valdir + '/residuals/quartiles_'+lab+'.nc'

r = netcdf_file(resfile,'r')
print(r.history)

a = r.variables['time']
rtime = a[:].copy()

a = r.variables['prediction month']
kmon = a[:].copy()

a = r.variables['quartile']
q = a[:].copy()

a = r.variables['positive quartiles']
presid = a[:,:,:].copy()

a = r.variables['negative quartiles']
nresid = a[:,:,:].copy()

del a
r.close()

#------------------------------------------------------------------------------
# do the fitting

# the fitting functions want time in months since cycle beginning
tobs = (odect - tstart)*12

# predictions
nlt = len(lead_times)
Nq = 3
pp = np.zeros((nlt,nobs)) - 1
pmin = np.zeros((nlt,nobs,Nq)) - 1
pmax = np.zeros((nlt,nobs,Nq)) - 1

pp10 = np.zeros((nlt,nobs)) - 1
pmin10 = np.zeros((nlt,nobs,Nq)) - 1
pmax10 = np.zeros((nlt,nobs,Nq)) - 1

cmonth = np.rint(tobs[-1]).astype(np.int32)
print(f"Current month = {cmonth}")

# loop through all obs times past two years
for idx in np.arange(nobs):

  omonth = np.rint(tobs[idx]).astype(np.int32)
  if omonth < 24:
    continue

  if ssn_sm[idx] < 0:
    continue

  print(80*'*'+f"\nobs month = {omonth}")

  for ilt in np.arange(len(lead_times)):

    pidx = idx - lead_times[ilt]
    pmonth = np.rint(tobs[pidx]).astype(np.int32)
    if pmonth < 24:
       continue

    print(f"Prediction month = {pmonth}")

    # do the fit
    afit = curve_fit(u.fpanel,tobs[:pmonth+1],ssn[:pmonth+1],p0=(170.0,0.0))
    f = u.fpanel(tobs[idx],afit[0][0],afit[0][1])

    if (deltak > 0) and (pmonth > (deltak + 23)):
      k2 = pmonth - deltak
      afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
      f2 = u.fpanel(tobs[idx],afit2[0][0],afit2[0][1])
      f = 0.5*(f+f2)

    # now the same for F10.7
    ffit = curve_fit(u.fclette10,tobs[:pmonth+1],fobs10[:pmonth+1],p0=(170.0,0.0))
    f10 = u.fclette10(tobs[idx],ffit[0][0],ffit[0][1])

    if (deltak > 0) and (pmonth > (deltak + 23)):
      k2 = pmonth - deltak
      ffit2 = curve_fit(u.fclette10,tobs[0:k2],fobs10[0:k2],p0=(170.0,0.0))
      f102 = u.fclette10(tobs[idx],ffit2[0][0],ffit2[0][1])
      f10 = 0.5*(f10+f102)

    #--------------------------------------------------------------------------
    # convert residuals to f10.7

    # converted f10 as opposed to fitted
    f10c = u.f10_from_ssn_2021(f)

    pp[ilt,idx] = f
    pp10[ilt,idx] = f10

    for q in np.arange(Nq):

      kidx = pmonth - kmon[0]
      smin = f - nresid[omonth,kidx,q]
      smax = f + presid[omonth,kidx,q]

      smax10 = u.f10_from_ssn_2021(smax) - f10c + f10
      smin10 = u.f10_from_ssn_2021(smin) - f10c + f10

      pmin[ilt,idx,q] = smin
      pmax[ilt,idx,q] = smax

      pmin10[ilt,idx,q] = smin10
      pmax10[ilt,idx,q] = smax10

#------------------------------------------------------------------------------
# plot out results.  Show SSN and F10.7 in separate files for 
# inclusion in presentations

plt.rc("font", weight = 'bold')

sns.set_theme(style={'axes.facecolor': '#F5F5F5'}, palette='colorblind')

fig1 = plt.figure(figsize = [6,3], dpi = 300)

ax = sns.lineplot(x = obstime[:-6], y = ssn_sm[:-6], color='blue', label='Smoothed SSN', linewidth=4)

#sns.lineplot(x = obstime, y = ssn, color='gray', label='SSN', alpha=0.2)

idx = np.where(pp[0,:] > 0)
sns.lineplot(x = obstime[idx], y = pp[0,idx[0]], color='darkmagenta', label='1 year lead time')
plt.fill_between(obstime[idx], y1 = pmin[0,idx[0],2], y2 = pmax[0,idx[0],2], color='darkmagenta', alpha=0.1)
plt.fill_between(obstime[idx], y1 = pmin[0,idx[0],1], y2 = pmax[0,idx[0],1], color='darkmagenta', alpha=0.2)
plt.fill_between(obstime[idx], y1 = pmin[0,idx[0],0], y2 = pmax[0,idx[0],0], color='darkmagenta', alpha=0.3)

idx = np.where(pp[1,:] > 0)
sns.lineplot(x = obstime[idx], y = pp[1,idx[0]], color='red', label='2 year lead time')
plt.fill_between(obstime[idx], y1 = pmin[1,idx[0],2], y2 = pmax[1,idx[0],2], color='red', alpha=0.1)
plt.fill_between(obstime[idx], y1 = pmin[1,idx[0],1], y2 = pmax[1,idx[0],1], color='red', alpha=0.2)
plt.fill_between(obstime[idx], y1 = pmin[1,idx[0],0], y2 = pmax[1,idx[0],0], color='red', alpha=0.3)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax.tick_params(axis='both', which='major', labelsize=8)

ax.set_ylabel('SSN', fontweight='bold', fontsize=10)
ax.set_xlabel('Date', fontweight='bold', fontsize=10)

ax.set_xlim([datetime.date(2022,12,1),datetime.date(obstime[-6].year,obstime[-6].month,1)])

ax.legend().set_visible(False)

fig1.tight_layout()
plt.savefig(outfig_ssn)

#------------------------------------------------------------------------------
#  Now F10.7

plt.rc("font", weight = 'bold')

sns.set_theme(style={'axes.facecolor': '#F5F5F5'}, palette='colorblind')

fig2 = plt.figure(figsize = [6,3], dpi = 300)

ax2 = sns.lineplot(x = obstime[:-6], y = fobs10_sm[:-6], color='blue', label='Smoothed F10.7',linewidth=4)

idx = np.where(pp10[0,:] > 0)
sns.lineplot(x = obstime[idx], y = pp10[0,idx[0]], color='blue', label='F10.7 1 year lead time')
plt.fill_between(obstime[idx], y1 = pmin10[0,idx[0],2], y2 = pmax10[0,idx[0],2], color='darkmagenta', alpha=0.1)
plt.fill_between(obstime[idx], y1 = pmin10[0,idx[0],1], y2 = pmax10[0,idx[0],1], color='darkmagenta', alpha=0.2)
plt.fill_between(obstime[idx], y1 = pmin10[0,idx[0],0], y2 = pmax10[0,idx[0],0], color='darkmagenta', alpha=0.3)

idx = np.where(pp10[1,:] > 0)
sns.lineplot(x = obstime[idx], y = pp10[1,idx[0]], color='red', label='F10.7 2 year lead time')
plt.fill_between(obstime[idx], y1 = pmin10[1,idx[0],2], y2 = pmax10[1,idx[0],2], color='red', alpha=0.1)
plt.fill_between(obstime[idx], y1 = pmin10[1,idx[0],1], y2 = pmax10[1,idx[0],1], color='red', alpha=0.2)
plt.fill_between(obstime[idx], y1 = pmin10[1,idx[0],0], y2 = pmax10[1,idx[0],0], color='red', alpha=0.3)

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax2.tick_params(axis='both', which='major', labelsize=8)

ax2.set_ylabel('F10.7 (sfu)', fontweight='bold', fontsize=10)
ax2.set_xlabel('Date', fontweight='bold', fontsize=10)

ax2.set_xlim([datetime.date(2022,12,1),datetime.date(obstime[-6].year,obstime[-6].month,1)])

ax2.legend().set_visible(False)

fig2.tight_layout()
plt.savefig(outfig_f10)

#------------------------------------------------------------------------------

print(obstime[-6])

plt.show()