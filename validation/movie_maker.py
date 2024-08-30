"""
Function to make a movie of how the prediction evolves with time
"""

import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import matplotlib.image as mpimg

from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.io import netcdf_file
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoMinorLocator
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# set the issue date

issue_date = datetime.date.today()
#issue_date = datetime.date(2023,11,1)

# start month for movie
mstart = 24

#------------------------------------------------------------------------------
# optionally average an earlier fit for stability
# units are months.  Set to -1 to disable

deltak = 9

#------------------------------------------------------------------------------
# Define files

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

indir, outdir, valdir = u.get_data_dirs()

# movie file
mfile = outdir + '/prediction_evolution.mp4'

# official start time of cycle 25 from SIDC, in decimal year
tstart = 2019.96

#------------------------------------------------------------------------------
# estimate date range of max
def get_date(t, g, gmin, gmax, tnow = None, label = None):

  # First see where the mean prediction peaks
  i = np.argmax(g)

  tmean = t[i]

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
  if ttmin < tnow:
     ttmin = tnow

  # now find amplitude range for the future
  idx = np.where(t > tnow)
  amin = int(np.max(gmin[idx]))
  amax = int(np.max(gmax[idx]))

  msg = f"{f[i]} {ptime[i].month}/{ptime[i].year}"

  if label is not None:
     msg = label + ': ' + msg

  print(80*'*')
  print("Mean prediction:")
  print(msg)
  print(80*'*')

  return [ttmin, np.max(tt), amin, amax]

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

for i in np.arange(len(obstime)):
   print(f"{i} {obstime[i]} {ssn[i]} {ssn_sm[i]}")

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
# read average residuals for this fit type

if resfile is None:
    resfile = valdir + '/residuals/quartiles_panel2_d9.nc'

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

Nerr = len(rtime)
Np = len(ptime)
Nmax = np.min([Np,Nerr])

#------------------------------------------------------------------------------

# the fitting functions want time in months since cycle beginning
tobs = (odect - tstart)*12
tpred = (pdect - tstart)*12

if mstart < kmon[0]:
    mstart = kmon[0]

# end at current prediction month 
mend = len(obstime)
print(f"Movie range = {mstart} {mend} {tobs[-1]}")

tmin = datetime.date(2020,1,15)
tmax = datetime.date(2032,1,1)

#------------------------------------------------------------------------------
fig = plt.figure(figsize = (11,5))
ax = fig.add_subplot(111)

plt.rcParams.update({'font.size': 12., 'font.weight': 'bold'})

plt.xlim(tmin, tmax)
plt.ylim(0, 200)

plt.xlabel('year', weight = 'bold')
plt.ylabel('SSN', weight = 'bold')

fig.tight_layout(rect=(0.02,0.18,0.99,.98))

ax.yaxis.set_tick_params(labelsize=12)

minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)

#------------------------------------------------------------------------------

frames = []
amps = []
pdates = []
maxdates = []
arange1 = []
arange2 = []
drange1 = []
drange2 = []

for pmonth in np.arange(mstart, mend):
  print(f"pmonth {pmonth}")

  afit = curve_fit(u.fpanel,tobs[:pmonth+1],ssn[:pmonth+1],p0=(170.0,0.0))
  f = u.fpanel(tpred,afit[0][0],afit[0][1])
  print(f"fit 1: {afit[0][0]} {afit[0][1]}")

  if (deltak > 0) and (pmonth > (deltak + 23)):
    k2 = pmonth - deltak
    afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
    f2 = u.fpanel(tpred,afit2[0][0],afit2[0][1])
    f = 0.5*(f+f2)
    print(f"fit 2: {afit2[0][0]} {afit2[0][1]}")

  # compute quartiles
  kidx = pmonth - kmon[0]
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
  # find min index to plot prediction: fidx = forecast index

  tnow = np.max(obstime[:pmonth+1])

  fidx = np.where(ptime > tnow)

  # time to start the mean prediction
  pstart = ptime[fidx[0][0] - 6]
  fidx_json = np.where(ptime >= pstart)

  #------------------------------------------------------------------------------
  # transition period

  #number of months to replace with smoothed values
  Ntransition = 6

  ptimej = ptime[fidx_json[0]]
  fj = f[fidx_json[0]]

  Nj = len(fj)
  x = np.zeros(13, dtype='float')
  y = np.zeros(13, dtype='float')
  fs = np.zeros(Ntransition, dtype='float')

  for i in np.arange(Ntransition):
    nn = 13-i-1
    # construct averaging array for ssn
    x[:nn] = ssn[pmonth-nn+1:pmonth+1]
    x[nn:] = fj[5:i+6]
    fs[i] = np.sum(x)/13.0

  fj[:Ntransition] = fs

  #------------------------------------------------------------------------------
  # plot SSN

  p0, = ax.plot(obstime[:pmonth+1],ssn[:pmonth+1], color='black', label = 'Monthly observations')

  sidx = pmonth + 1 - 6
  #p1, = ax.plot(obstime[:sidx], ssn_sm[:sidx], color='blue', linewidth = 4, label = 'Smoothed monthly observations')
  p1, = ax.plot(obstime[:-6], ssn_sm[:-6], color='blue', linewidth = 4, label = 'Smoothed monthly observations')

  px = np.insert(ptimej,0,obstime[sidx-1])
  py = np.insert(fj,0,ssn_sm[sidx-1])
  p2, = ax.plot(px,py, color='darkmagenta', label = "Experimental Prediction")

  p3 = ax.fill_between(x=ptime[fidx], y1=smin[fidx[0],0], y2=smax[fidx[0],0], color='darkmagenta', alpha=0.3, label = "25% quartile")
  p4 = ax.fill_between(x=ptime[fidx], y1=smin[fidx[0],1], y2=smax[fidx[0],1], color='darkmagenta', alpha=0.2, label = "50% quartile")
  p5 = ax.fill_between(x=ptime[fidx], y1=smin[fidx[0],2], y2=smax[fidx[0],2], color='darkmagenta', alpha=0.1, label = "75% quartile")

  #------------------------------------------------------------------------------
  # first accumulate information for csv output

  idx = np.argmax(f)
  amp = np.rint(f[idx]).astype(np.int32)
  amps.append(amp)
  maxdate = ptime[idx]
  maxdates.append(maxdate)
  pdates.append(tnow)

  t1, t2, a1, a2 = get_date(ptime, f, smin[:,1], smax[:,1], label = "SSN")
  arange1.append(a1)
  arange2.append(a2)
  drange1.append(t1)
  drange2.append(t2)

  #------------------------------------------------------------------------------
  # annotations

  lab = f"Max of Mean Prediction: {amp}"
  a1 = ax.annotate(lab,(.5,.5),xytext = (.8,.86), xycoords='figure fraction',color='darkmagenta', ha='center')

  lab2 = f"{ptime[idx].month}/{ptime[idx].year}"
  a2 = ax.annotate(lab2,(.5,.5),xytext = (.8,.8), xycoords='figure fraction',color='darkmagenta', ha='center')

  lab3 = f"Prediction Date: {tnow.month}/{tnow.year}"
  a7 = ax.annotate(lab3,(.5,.5),xytext = (.1,.86), xycoords='figure fraction',color='black', ha='left')

  logo = mpimg.imread("../operations/noaa-logo-rgb-2022.png")
  imagebox = OffsetImage(logo, zoom = 0.024)
  ab = AnnotationBbox(imagebox, (.14, .18), frameon = False, xycoords='figure fraction', annotation_clip = False)
  a3 = ax.add_artist(ab)

  nwslogo = mpimg.imread("../operations/NWS_logo.png")
  imagebox = OffsetImage(nwslogo, zoom = 0.042)
  ab = AnnotationBbox(imagebox, (.2, .18), frameon = False, xycoords='figure fraction', annotation_clip = False)
  a4 = ax.add_artist(ab)

  # creation date
  cd = issue_date
  clab = f"issued {cd.month}/{cd.year}"
  a5 = ax.annotate("Space Weather Prediction Testbed", (.18,.07),xycoords='figure fraction', ha='center', annotation_clip = False, fontsize = 10)
  a6 = ax.annotate(clab, (.18,.03),xycoords='figure fraction', ha='center', annotation_clip = False, fontsize = 10)

  if len(frames) == 0:
    hh, ss = ax.get_legend_handles_labels()
    leg1 = ax.legend(hh[0:3],ss[0:3],loc="lower center", bbox_to_anchor=(0.56,-0.48), frameon = False)
    l1 = ax.add_artist(leg1)
    leg2 = ax.legend(hh[3:],ss[3:],loc="lower center", bbox_to_anchor=(0.9,-0.48), frameon = False)
    l2 = ax.add_artist(leg2)
  frames.append([p0,p1,p2,p3,p4,p5,a1,a2,a3,a4,a5,a6,a7,l1,l2])

mov = animation.ArtistAnimation(fig, frames, interval = 200, blit = True,
              repeat = True, repeat_delay = 1000)
mov.save(mfile)

#------------------------------------------------------------------------------
# output results to a csv file

import csv

fields = ["prediction month","prediction year", "amp", "max month",
          "max year", "min amp", "max amp", "min date month", \
          "min date year", "max date month", "max date year"]

csvfile = outdir + '/prediction_evolution.csv'

rows = []
for i in np.arange(len(amps)):
   rows.append([pdates[i].month, pdates[i].year, amps[i], maxdates[i].month, \
                maxdates[i].year, arange1[i], arange2[i], drange1[i].month, \
                drange1[i].year, drange2[i].month, drange2[i].year])

with open(csvfile, 'w') as csvfile:
   csvwriter = csv.writer(csvfile)
   csvwriter.writerow(fields)
   csvwriter.writerows(rows)
