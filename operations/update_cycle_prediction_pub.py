"""
This is a modified version of update_cycle_prediction.py that is used to generate a figure for the paper
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
# set the issue date

#issue_date = datetime.date.today()
issue_date = datetime.date(2024,1,2)
print(f"Issue date: {issue_date}")

# set this to False for normal operations
# set to true to rerun a past date
reanalysis = True

#failsafe
if issue_date != datetime.date.today():
   reanalysis = True

#------------------------------------------------------------------------------
# optionally average an earlier fit for stability
# units are months.  Set to -1 to disable

deltak = 9

#------------------------------------------------------------------------------
# Define files

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

indir, outdir, valdir = u.get_data_dirs()

if reanalysis:
   outdir = valdir + '/reanalysis'

# Figure product
outfig = outdir + '/cycle_update-pub.png'

# optionally archive previous predictions
archive = False

# json product
outfile = outdir + "/predicted-solar-cycle-pub.json"

# official start time of cycle 25 from SIDC, in decimal year
tstart = 2019.96

#------------------------------------------------------------------------------
# define colors for output
red = '\033[91m'
yellow = '\033[93m'
cend = '\033[0m'

#------------------------------------------------------------------------------
def get_label(tm, tstart):
    # given input in decimal year, write as a date
    tdec = tstart + tm/12.
    t = Time(tdec,format='decimalyear').datetime
    return f"{t.year}-{t.month}-{t.day}"

#------------------------------------------------------------------------------
# estimate date range of max
def get_date(t, g, gmin, gmax, tnow = None, label = None):

  # First see where the mean prediction peaks
  i = np.argmax(g)

  tmean = t[i]

  for m in np.arange(len(gmin)):
     print(f"{label}: {t[m]} {g[m]:.1f} {gmin[m]:.1f} {gmax[m]:.1f}")

  # now see where the min and max curves peak on either side 
  # of the mean

  iin = np.where(t <= tmean)
  iip = np.where(t >= tmean)

  tn = t[iin]
  tp = t[iip]

  tmin1 = tn[np.argmax(gmin[iin])]
  tmin2 = tn[np.argmax(gmax[iin])]

  tmax1 = tp[np.argmax(gmin[iip])]
  tmax2 = tp[np.argmax(gmax[iip])]

  tt = np.array([tmin1, tmax1, tmean, tmin2, tmax2])

  if tnow is None:
     tnow = datetime.date.today()

  ttmin = np.min(tt)
  print(f"ttmin: {ttmin}")

  #if ttmin < tnow:
  #   ttmin = tnow

  # now find amplitude range for the future
  idx = np.where(t > tnow)
  amin = int(np.max(gmin[idx]))
  amax = int(np.max(gmax[idx]))

  msg = f"{g[i]} {month[ptime[i].month]} {ptime[i].year}"

  if label is not None:
     msg = label + ': ' + msg

  print(80*'*')
  print(yellow+"Mean prediction:"+cend)
  print(msg)
  print(80*'*')

  return [ttmin, np.max(tt), amin, amax]

#------------------------------------------------------------------------------
# determine whether or not you are in the declining phase

def declining_phase(tp, p, pmin, pmax, tdata, data, tnow = None, label = 'SSN'):
  # input parameters
  # tp = time axis for p, pmin, and pmax
  # p = mean prediction
  # pmin = lower median quartile for p
  # pmax = upper median quartile for p
  # tdata = time axis for data
  # data = smoothed observations

  if tnow is None:
     tnow = datetime.date.today()

  # these indices correspond to future time, beginning now
  pif = np.where(tp > tnow)

  # First check to see if the max observed SSN is greater than
  # the max of the positive median quartile.  If yes, then the
  # prediction is that this is the declining phase.
  if np.max(data) > np.max(pmax[pif]):
     dec = True
     tpeak = obstime[np.argmax(data)]
     print(red+f"{label} declining phase {np.max(data)} {np.max(pmax[pif])}"+cend)

  else:
     # This may or may not be the declining phase
     # So, we should still quote a range for possible max
     dec = False

     # if the observed smoothed SSN is larger than the lower median quartile
     # then there is a possibility that the max has already passed. Shift the
     # estimated range to a value earlier than today
     if np.max(data) > np.max(pmin[pif]):
        tpeak = obstime[np.argmax(data)]
        print(red+f"{label} max may have passed: {np.max(data)} {np.max(pmin[pif])}"+cend)
     else:
        # set this to a large value so it doesn't change the time range determined
        # by get_date()
        tpeak = np.max(ptime)

  print(80*'*')
  print(yellow+f"Max observed {label} : {np.max(data)}"+cend)
  print(f"low  prediction {label} : {np.max(pmin[pif])}")
  print(f"mean prediction {label} : {np.max(p[pif])}")
  print(f"high prediction {label} : {np.max(pmax[pif])}")
  print(80*'*')

  return dec, tpeak

#------------------------------------------------------------------------------
# read SSN panel prediction range
rfile = open(ssnfile)

data = json.loads(rfile.read())
N = len(data)

ptime = []
pmin = []
pmax = []

for d in data:
    t = np.array(d['time-tag'].split('-'), dtype='int')
    ptime.append(datetime.date(t[0], t[1], 15))
    pmin.append(d['smoothed_ssn_min'])
    pmax.append(d['smoothed_ssn_max'])

ptime = np.array(ptime)
pmin = np.array(pmin)
pmax = np.array(pmax)

#------------------------------------------------------------------------------
# read F10.7 panel prediction range
rfile10 = open(f10file)

data = json.loads(rfile10.read())
N = len(data)

ptime10 = []
pmin10 = []
pmax10 = []

for d in data:
    t = np.array(d['time-tag'].split('-'), dtype='int')
    ptime10.append(datetime.date(t[0], t[1], 15))
    pmin10.append(d['smoothed_f10.7_min'])
    pmax10.append(d['smoothed_f10.7_max'])

ptime10 = np.array(ptime10)
pmin10 = np.array(pmin10)
pmax10 = np.array(pmax10)

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

for i in np.arange(len(fobs10)):
   print(f"{obstime[i]} {fobs10[i]} {fobs10_sm[i]}")

#------------------------------------------------------------------------------
# If a reanalysis, truncate observations

if reanalysis:

  iobs = np.where(obstime < issue_date)

  obstime = obstime[iobs]
  ssn = ssn[iobs]
  ssn_sm = ssn_sm[iobs]
  fobs10 = fobs10[iobs]
  fobs10_sm = fobs10_sm[iobs]

  ssn_sm[-6:] = -1
  fobs10_sm[-6:] = -1

#------------------------------------------------------------------------------

# time of available observations in decimal year
nobs = len(obstime)
odect = np.zeros(nobs)
for i in np.arange(nobs):
    dt = datetime.datetime(obstime[i].year,obstime[i].month,obstime[i].day)
    odect[i] = Time(dt).to_value('decimalyear')

# time of predictions in decimal year
nn = len(ptime)
pdect = np.zeros(nn)
for i in np.arange(nn):
    dt = datetime.datetime(ptime[i].year,ptime[i].month,ptime[i].day)
    pdect[i] = Time(dt).to_value('decimalyear')

#------------------------------------------------------------------------------
# choose fit type
# 1: 2-parameter panel fit (A, t0)
# 2: 2-parameter upton-hathaway 2023 (A,t0)

ftype = 1

# the fitting functions want time in months since cycle beginning
tobs = (odect - tstart)*12
tpred = (pdect - tstart)*12

# prediction month
pmonth = np.rint(tobs[-1]).astype(np.int32)
print(f"Prediction month = {pmonth}")

if ftype == 2:
  lab = "uh"
  afit = curve_fit(u.fuh,tobs,ssn,p0=(170.0,0.0))
  f = u.fuh(tpred,afit[0][0],afit[0][1])
else:
  lab = "panel2"
  afit = curve_fit(u.fpanel,tobs,ssn,p0=(170.0,0.0))
  f = u.fpanel(tpred,afit[0][0],afit[0][1])

if (deltak > 0) and (pmonth > (deltak + 23)):
  lab = lab+f"_d{deltak}"
  k2 = pmonth - deltak
  if ftype == 2:
    afit2 = curve_fit(u.fuh,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
    f2 = u.fuh(tpred,afit2[0][0],afit2[0][1])
    print(f"fit 1: {afit[0][0]} {afit[0][1]}")
  else:
    afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
    f2 = u.fpanel(tpred,afit2[0][0],afit2[0][1])
    print(f"fit 2: {afit2[0][0]} {afit2[0][1]}")
  f = 0.5*(f+f2)

#------------------------------------------------------------------------------
# read average residuals for this fit type

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

kidx = pmonth - kmon[0]

Nerr = len(rtime)

Np = len(ptime)
Nmax = np.min([Np,Nerr])

smin = np.zeros((Np,4))
smax = np.zeros((Np,4))

for q in np.arange(4):
  srn = savgol_filter(nresid[:,kidx,q], 21, 3)
  srp = savgol_filter(presid[:,kidx,q], 21, 3)
  np.clip(srn, 0.0, None, out = srn)
  np.clip(srp, 0.0, None, out = srp)

  for i in np.arange(Nmax):
    smin[i,q] = f[i] - srn[i]
    smax[i,q] = f[i] + srp[i]

  # if prediction extends beyond residual data, then duplicate residual data
  for i in np.arange(Nmax,Np):
    smin[i,q] = f[i] - srn[-1]
    smax[i,q] = f[i] + srp[-1]

# make sure SSN does not go negative
np.clip(smin, 0.0, None, out = smin)
np.clip(smax, 0.0, None, out = smax)

#------------------------------------------------------------------------------
# fit f10.7 directly
# for now assume that the time array is the same as ssn

ffit = curve_fit(u.fclette10,tobs,fobs10,p0=(170.0,0.0))
f10 = u.fclette10(tpred,ffit[0][0],ffit[0][1])

if (deltak > 0) and (pmonth > (deltak + 23)):
  k2 = pmonth - deltak
  ffit2 = curve_fit(u.fclette10,tobs[0:k2],fobs10[0:k2],p0=(170.0,0.0))
  f102 = u.fclette10(tpred,ffit2[0][0],ffit2[0][1])
  f10 = 0.5*(f10+f102)


#------------------------------------------------------------------------------
# convert residuals to f10.7

# converted f10 as opposed to fitted
f10c = u.f10_from_ssn_2021(f)

smax10 = u.f10_from_ssn_2021(smax)
smin10 = u.f10_from_ssn_2021(smin)

# minimum value
sclip10 = np.min(smin10)

# recenter on direct curve fit
for i in np.arange(smax10.shape[1]):
   smax10[:,i] = smax10[:,i] - f10c + f10
   smin10[:,i] = smin10[:,i] - f10c + f10

# make sure you haven't changed the min
np.clip(smin10, sclip10, None, out = smin10)
np.clip(smax10, sclip10, None, out = smax10)

#------------------------------------------------------------------------------
# find min index to plot prediction: fidx = forecast index

tnow = np.max(obstime)
print(f"Current time: {tnow}")

fidx = np.where(ptime > tnow)

# time to start the prediction in the json file
pstart = ptime[fidx[0][0] - 6]
fidx_json = np.where(ptime >= pstart)

#------------------------------------------------------------------------------
# write prediction to a json file

# but replace first months with a 13-month smoothing between
# observed monthly values and prediction
# this gives a smooth transition from the smoothed observations
# to the prediction

#number of months to replace with smoothed values
Ntransition = 6

ptimej = ptime[fidx_json[0]]

fj = f[fidx_json[0]]
sminj = smin[fidx_json[0],:3]
smaxj = smax[fidx_json[0],:3]

f10j = f10[fidx_json[0]]
smin10j = smin10[fidx_json[0],:3]
smax10j = smax10[fidx_json[0],:3]

Nj = len(fj)

x = np.zeros(13, dtype='float')
y = np.zeros(13, dtype='float')

fs = np.zeros(Ntransition, dtype='float')
f10s = np.zeros(Ntransition, dtype='float')

for i in np.arange(Ntransition):

  nn = 13-i-1

  # construct averaging array for ssn
  x[:nn] = ssn[-nn:]
  x[nn:] = fj[5:i+6]

  # construct averaging array for f10.7
  y[:nn] = fobs10[-nn:]
  y[nn:] = f10j[5:i+6]

  fs[i] = np.sum(x)/13.0
  f10s[i] = np.sum(y)/13.0

for q in np.arange(sminj.shape[1]):
   sminj[:Ntransition,q] = sminj[:Ntransition,q] - fj[:Ntransition] + fs
   smaxj[:Ntransition,q] = smaxj[:Ntransition,q] - fj[:Ntransition] + fs
   smin10j[:Ntransition,q] = smin10j[:Ntransition,q] - f10j[:Ntransition] + f10s
   smax10j[:Ntransition,q] = smax10j[:Ntransition,q] - f10j[:Ntransition] + f10s

fj[:Ntransition] = fs
f10j[:Ntransition] = f10s

outdata = []
for i in np.arange(Nj):
   if ptimej[i].year < 2031:

     # sanity check
     for q in np.arange(sminj.shape[1]):
       if sminj[i,q] > fj[i]:
          print(f"ERROR in SSN min {i} {ptimej[i]} {sminj[i,q]} {fj[i]}")
          sminj[i,q] = fj[i]
       if smaxj[i,q] < fj[i]:
          print(f"ERROR in SSN max {i} {ptimej[i]} {fj[i]} {smaxj[i,q]}")
          smaxj[i,q] = fj[i]
       if smin10j[i,q] > f10j[i]:
          print(f"ERROR in F10.7 min {i} {ptimej[i]} {smin10j[i,q]} {f10j[i]}")
          smin10j[i,q] = f10j[i]
       if smax10j[i,q] < f10j[i]:
          print(f"ERROR in F10.7 max {i} {ptimej[i]} {f10j[i]} {smax10j[i,q]}")
          smax10j[j,q] = f10j[i]

     out = {
        "time-tag": f"{ptimej[i].year}-{ptimej[i].month:02d}",
        "predicted_ssn": round(fj[i],1),
        "high25_ssn": round(smaxj[i,0],1),
        "high_ssn": round(smaxj[i,1],1),
        "high75_ssn": round(smaxj[i,2],1),
        "low25_ssn": round(sminj[i,0],1),
        "low_ssn": round(sminj[i,1],1),
        "low75_ssn": round(sminj[i,2],1),
        "predicted_f10.7": round(f10j[i],1),
        "high25_f10.7": round(smax10j[i,0],1),
        "high_f10.7": round(smax10j[i,1],1),
        "high75_f10.7": round(smax10j[i,2],1),
        "low25_f10.7": round(smin10j[i,0],1),
        "low_f10.7": round(smin10j[i,1],1),
        "low75_f10.7": round(smin10j[i,2],1)
     }
     outdata.append(out)

jout = json.dumps(outdata)

with open(outfile, "w") as file:
   file.write(jout)

# save another copy for the archive
if archive == True:

    dir = outdir + '/archive'
    os.makedirs(dir, exist_ok = True)

    basename = os.path.basename(outfile).split('.json')[0]

    mymonth = f"{obstime[-1].month:02d}"

    fname = f"{dir}/{basename}_{obstime[-1].year}_{mymonth}.json"
    with open(fname, "w") as file:
       file.write(jout)


#------------------------------------------------------------------------------
month = {
   1:"Jan",
   2:"Feb",
   3:"Mar",
   4:"Apr",
   5:"May",
   6:"Jun",
   7:"Jul",
   8:"Aug",
   9:"Sep",
   10:"Oct",
   11:"Nov",
   12:"Dec"
}

#------------------------------------------------------------------------------
# plot SSN

fig, ax = plt.subplots(2, 1, figsize = [12.8,6.5])

fig.tight_layout(rect=(0.12,0.18,0.9,.96))
ax[0].xaxis.set_tick_params(labelsize=14)
ax[1].xaxis.set_tick_params(labelsize=14)
ax[0].yaxis.set_tick_params(labelsize=12)
ax[1].yaxis.set_tick_params(labelsize=12)

minor_locator = AutoMinorLocator(2)
ax[0].xaxis.set_minor_locator(minor_locator)
ax[1].xaxis.set_minor_locator(minor_locator)

yminor = AutoMinorLocator(5)
ax[0].yaxis.set_minor_locator(yminor)
ax[1].yaxis.set_minor_locator(yminor)

ax[0].yaxis.set_ticks_position('both')
ax[1].yaxis.set_ticks_position('both')

ymax = np.max([np.max(smax[fidx[0],2]),np.max(ssn)]) * 1.05

ssn_sm_nz = np.ma.masked_less(ssn_sm, 0.0)

tmin = np.min(ptime)
tmax = datetime.date(2032,1,1)

sns.set_theme(style={'axes.facecolor': '#FFFFFF'}, palette='colorblind')

sns.lineplot(x=obstime,y=ssn, color='black', ax = ax[0])
ax[0].set_xlim([tmin,tmax])
ax[0].set_ylim([0,ymax])

sns.lineplot(x=obstime, y=ssn_sm_nz, color='blue', linewidth = 4, ax = ax[0])

ax[0].fill_between(x=ptime[fidx], y1=smin[fidx[0],0], y2=smax[fidx[0],0], color='darkmagenta', alpha=0.3)
ax[0].fill_between(x=ptime[fidx], y1=smin[fidx[0],1], y2=smax[fidx[0],1], color='darkmagenta', alpha=0.2)
ax[0].fill_between(x=ptime[fidx], y1=smin[fidx[0],2], y2=smax[fidx[0],2], color='darkmagenta', alpha=0.1)

ax[0].fill_between(x=ptime, y1=pmin, y2=pmax, color='red', alpha=0.2)

px = np.insert(ptimej,0,obstime[:-6])
py = np.insert(fj,0,ssn_sm[:-6])
sns.lineplot(x=px,y=py, color='darkmagenta', ax = ax[0])

ax[0].set_ylabel('Sunspot Number',fontsize=16)

#------------------------------------------------------------------------------
# plot F10.7

fobs10_sm_nz = np.ma.masked_less(fobs10_sm, 0.0)

ymax = np.max([np.max(smax10[fidx[0],2]),np.max(fobs10)]) * 1.05

sns.lineplot(x=obstime,y=fobs10, color='black', ax = ax[1], label = 'Monthly observations')
ax[1].set_xlim([tmin,tmax])
ax[1].set_ylim([50,ymax])

sns.lineplot(x=obstime, y=fobs10_sm_nz, color='blue', linewidth = 4, ax = ax[1], label = "Smoothed monthly observations")

idx = np.where(pmin10 > 0.0)
ax[1].fill_between(x=ptime10[idx], y1=pmin10[idx], y2=pmax10[idx], color='red', alpha=0.2, label = "2019 NOAA/NASA/ISES Panel Prediction (range)")

py10 = np.insert(f10j,0,fobs10_sm[:-6])
sns.lineplot(x=px,y=py10, color='darkmagenta', ax = ax[1], label = "Base Prediction")

ax[1].fill_between(x=ptime[fidx[0]], y1=smin10[fidx[0],0], y2=smax10[fidx[0],0], color='darkmagenta', alpha=0.3, label = "25% quartile")
ax[1].fill_between(x=ptime[fidx[0]], y1=smin10[fidx[0],1], y2=smax10[fidx[0],1], color='darkmagenta', alpha=0.2, label = "50% quartile")
ax[1].fill_between(x=ptime[fidx[0]], y1=smin10[fidx[0],2], y2=smax10[fidx[0],2], color='darkmagenta', alpha=0.1, label = "75% quartile")

#------------------------------------------------------------------------------
# show grids

ax[0].grid(visible=True)
ax[1].grid(visible=True)

#------------------------------------------------------------------------------
# compute amplitude and date ranges based on median values

t1, t2, a1, a2 = get_date(ptime, f, smin[:,1], smax[:,1], tnow = issue_date, label = "SSN")
trange = [t1, t2]
arange = [a1, a2]

t1, t2, a1, a2 = get_date(ptime, f10, smin10[:,1], smax10[:,1], tnow = issue_date, label = "F10.7")
trange10 = [t1, t2]
arange10 = [a1, a2]

#------------------------------------------------------------------------------
# determine whether or not you have already passed the peak
# if so, override the trange and arange as previously defined

declining_ssn, td_ssn = declining_phase(ptime, f, smin[:,1], smax[:,1], obstime, ssn_sm)
declining_f10, td_f10 = declining_phase(ptime, f10, smin10[:,1], smax10[:,1], obstime, fobs10_sm, label='F10.7')

# shift the quoted range earlier if there is a chance that max has already passed
trange[0] = np.min([trange[0], td_ssn])
trange10[0] = np.min([trange10[0], td_f10])

# quoted amplitude range should be no lower than observed
arange[0] = round(np.max([arange[0], np.max(ssn_sm)]))
arange10[0] = round(np.max([arange10[0], np.max(fobs10_sm)]))

#------------------------------------------------------------------------------
# labels

if declining_ssn:
   idx = np.argmax(ssn_sm)
   lab1 = f"Max {ssn[idx]}"
   lab2 = f"{month[obstime[idx].month]} {obstime[idx].year}"

else:
  lab1 = f"Predicted Max {arange[0]} - {arange[1]}"
  if trange[0].year == trange[1].year:
    lab2 = f"{month[trange[0].month]} - {month[trange[1].month]} {trange[1].year}"
  else:
    lab2 = f"{month[trange[0].month]} {trange[0].year} - {month[trange[1].month]} {trange[1].year}"

top0 = ax[0].get_position().get_points()[1][1]
top1 = ax[1].get_position().get_points()[1][1]

xx = .74
yy = top0 - .05
dy = .033

ax[0].annotate("International Sunspot Number", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='black',
               ha='center', weight = 'bold', bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))
yy -= dy
ax[0].annotate(lab1, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center',
                              bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))
yy -= dy
ax[0].annotate(lab2, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center',
                              bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))

yy = top1 - .06

if declining_f10:
  idx = np.argmax(fobs10_sm)
  lab1 = f"Max {fobs10[idx]}"
  lab2 = f"{month[obstime[idx].month]} {obstime[idx].year}"
else:
  lab1 = f"Predicted Max {arange10[0]} - {arange10[1]}"
  if trange10[0].year == trange10[1].year:
    lab2 = f"{month[trange10[0].month]} - {month[trange10[1].month]} {trange10[1].year}"
  else:
    lab2 = f"{month[trange10[0].month]} {trange10[0].year} - {month[trange10[1].month]} {trange10[1].year}"

ax[1].annotate("F10.7cm Radio Flux", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='black', ha='center', weight = 'bold',
               bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))
yy -= dy
ax[1].annotate(lab1, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center',
               bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))
yy -= dy
ax[1].annotate(lab2, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center',
               bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))

#--------------------

plt.ylabel('Solar Flux Units',fontsize=16)
plt.xlabel('Years',fontsize=16)

#fig.suptitle("Experimental Solar Cycle 25 Prediction", weight="bold")

#--------------------

hh, ss = ax[1].get_legend_handles_labels()

leg1 = ax[1].legend(hh[0:3],ss[0:3],loc="lower center", bbox_to_anchor=(0.47,-0.76), frameon = False)

plt.legend(hh[3:],ss[3:],loc="lower center", bbox_to_anchor=(0.89,-0.76), frameon = False)
ax[1].add_artist(leg1)

from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as mpimg

logo = mpimg.imread("noaa-logo-rgb-2022.png")
imagebox = OffsetImage(logo, zoom = 0.024)
ab = AnnotationBbox(imagebox, (.185, .13), frameon = False, xycoords='figure fraction', annotation_clip = False)
ax[1].add_artist(ab)

nwslogo = mpimg.imread("NWS_logo.png")
imagebox = OffsetImage(nwslogo, zoom = 0.042)
ab = AnnotationBbox(imagebox, (.235, .13), frameon = False, xycoords='figure fraction', annotation_clip = False)
ax[1].add_artist(ab)

# creation date
cd = issue_date
clab = f"issued {cd.day} {month[cd.month]} {cd.year}"
ax[1].annotate("Space Weather Prediction Testbed", (.21,.055),xycoords='figure fraction', ha='center', annotation_clip = False, fontsize = 10)
ax[1].annotate(clab, (.21,.03),xycoords='figure fraction', ha='center', annotation_clip = False, fontsize = 10)

#--------------------

plt.savefig(outfig, dpi = 100)

# save another copy for the archive
if archive == True:

    dir = outdir + '/archive'
    os.makedirs(dir, exist_ok = True)

    basename = os.path.basename(outfig).split('.png')[0]

    mymonth = f"{obstime[-1].month:02d}"

    fname = f"{dir}/{basename}_{obstime[-1].year}_{mymonth}.png"
    plt.savefig(fname, dpi = 300)

#--------------------
# Write first 13 rows of the prediction json file to a csv file for SWFO monitoring 

import csv

fields = ["date","predicted ssn"]

csvfile = outdir + '/cycle_short_term_prediction.csv'

rows = []
for i in np.arange(13):
   date = f"{ptimej[i].month:02d}/{ptimej[i].year}"
   rows.append([date, fj[i]])

with open(csvfile, 'w') as csvfile:
   csvwriter = csv.writer(csvfile)
   csvwriter.writerow(fields)
   csvwriter.writerows(rows)

#------------------------------------------------------------------------------
