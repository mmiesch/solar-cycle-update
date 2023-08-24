"""
The purpose of this function is to compute an updated prediction for the current solar cycle.  It is intended to be applied at least 3 years into the cycle, when there is enough SSN data to make a reasonable projection based on the average cycle progression (formula due to Hathaway et al).  The prediction is done by fitting the current data to the nonlinear function that approximates an average cycle.
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

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# optionally average an earlier fit for stability
# units are months.  Set to -1 to disable

#deltak = -1
deltak = 9

#------------------------------------------------------------------------------
# Define files

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

indir, outdir, valdir = u.get_data_dirs()

# Figure product
outfig = outdir + '/cycle_update.png'

# optionally archive previous predictions
archive = True

# json product
outfile = outdir + "/predicted-solar-cycle.json"

# official start time of cycle 25 from SIDC, in decimal year
tstart = 2019.96

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

ssn = np.array(ssn)
ssn_sm = np.array(ssn_sm)
fobs10 = np.array(fobs10)
fobs10_sm = np.array(fobs10_sm)

for i in np.arange(len(fobs10)):
   print(f"{obstime[i]} {fobs10[i]} {fobs10_sm[i]}")

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
  else:
    afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
    f2 = u.fpanel(tpred,afit2[0][0],afit2[0][1])
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

rmin = np.zeros(Np)
rmax = np.zeros(Np)

smin = np.zeros((Np,4))
smax = np.zeros((Np,4))

for q in np.arange(4):
  for i in np.arange(Nmax):
    rmin[i] = f[i] - nresid[i,kidx,q]
    rmax[i] = f[i] + presid[i,kidx,q]

  smin[:,q] = savgol_filter(rmin, 21, 3)
  smax[:,q] = savgol_filter(rmax, 21, 3)

#------------------------------------------------------------------------------
# fit f10.7 directly
# for now assume that the time array is the same as ssn

ffit = curve_fit(u.fpanel10,tobs,fobs10,p0=(170.0,0.0))
f10 = u.fpanel10(tpred,ffit[0][0],ffit[0][1])

if (deltak > 0) and (pmonth > (deltak + 23)):
  k2 = pmonth - deltak
  ffit2 = curve_fit(u.fpanel10,tobs[0:k2],fobs10[0:k2],p0=(170.0,0.0))
  f102 = u.fpanel10(tpred,ffit2[0][0],ffit2[0][1])
  f10 = 0.5*(f10+f102)

#------------------------------------------------------------------------------
# convert residuals to f10.7

# converted f10 as opposed to fitted
f10c = u.f10_from_ssn_2019(f)

smax10 = u.f10_from_ssn_2019(smax)
smin10 = u.f10_from_ssn_2019(smin)

# recenter on direct curve fit
for i in np.arange(smax10.shape[1]):
   smax10[:,i] = smax10[:,i] - f10c + f10
   smin10[:,i] = smin10[:,i] - f10c + f10

#------------------------------------------------------------------------------
# find min index to plot prediction: fidx = forecast index

tnow = np.max(obstime)
print(f"Current time: {tnow}")

fidx = np.where(ptime > tnow)

#------------------------------------------------------------------------------
# write prediction to a json file

outdata = []
for i in fidx[0]:
   if ptime[i].year < 2033:
     out = {
        "time-tag": f"{ptime[i].year}-{ptime[i].month:02d}",
        "predicted_ssn": f[i],
        "high_ssn": smax[i,1],
        "low_ssn": smin[i,1],
        "predicted_f10.7": f10[i],
        "high_f10.7": smax10[i,1],
        "low_f10.7": smin10[i,1]
     }
     outdata.append(out)

jout = json.dumps(outdata)

with open(outfile, "w") as file:
   file.write(jout)

#------------------------------------------------------------------------------
def get_label(tm, tstart):
    # given input in decimal year, write as a date
    tdec = tstart + tm/12.
    t = Time(tdec,format='decimalyear').datetime
    return f"{t.year}-{t.month}-{t.day}"

def qlabels(ax, top = False):

  if top:
     yy = .78
  else:
     yy = .3

  xx = .74
  ax.annotate("quartiles: ", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='right', weight='bold')
  ax.annotate("25% ", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='left', weight='bold', alpha=0.7)
  dx = .04
  xx += dx
  ax.annotate("50% ", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='left', weight='bold', alpha = 0.5)
  xx += dx
  ax.annotate("75% ", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='left', weight='bold', alpha = 0.36)

#------------------------------------------------------------------------------
month = {
   1:"January",
   2:"February",
   3:"March",
   4:"April",
   5:"May",
   6:"June",
   7:"July",
   8:"August",
   9:"September",
   10:"October",
   11:"November",
   12:"December"
}
#------------------------------------------------------------------------------
# plot SSN

fig, ax = plt.subplots(2, 1, figsize = [12.8,6.5])
fig.tight_layout(rect=(0.14,0.04,0.9,1.))
ax[0].xaxis.set_tick_params(labelsize=14)
ax[1].xaxis.set_tick_params(labelsize=14)
ax[0].yaxis.set_tick_params(labelsize=12)
ax[1].yaxis.set_tick_params(labelsize=12)

ymax = np.max(smax[fidx[0],2]) * 1.05

ssn_sm_nz = np.ma.masked_less(ssn_sm, 0.0)

tmin = np.min(ptime)
tmax = datetime.date(2032,1,1)

sns.set_theme(style={'axes.facecolor': '#FFFFFF'}, palette='colorblind')

sns.lineplot(x=obstime,y=ssn, color='black', ax = ax[0])
ax[0].set_xlim([tmin,tmax])
ax[0].set_ylim([0,ymax])

#plt.fill_between(x=time[fidx], y1=smin[fidx[0],3], y2=smax[fidx[0],3], color='darkmagenta', alpha=0.05)
ax[0].fill_between(x=ptime[fidx], y1=smin[fidx[0],2], y2=smax[fidx[0],2], color='darkmagenta', alpha=0.1)
ax[0].fill_between(x=ptime[fidx], y1=smin[fidx[0],1], y2=smax[fidx[0],1], color='darkmagenta', alpha=0.2)
ax[0].fill_between(x=ptime[fidx], y1=smin[fidx[0],0], y2=smax[fidx[0],0], color='darkmagenta', alpha=0.3)

sns.lineplot(x=obstime, y=ssn_sm_nz, color='blue', linewidth = 4, ax = ax[0])

ax[0].fill_between(x=ptime, y1=pmin, y2=pmax, color='red', alpha=0.3)

sns.lineplot(x=ptime,y=f, color='darkmagenta', ax = ax[0])

ax[0].set_ylabel('Sunspot Number',fontsize=16)

#------------------------------------------------------------------------------
# plot F10.7

fobs10_sm_nz = np.ma.masked_less(fobs10_sm, 0.0)

ymax = np.max(smax10[fidx[0],2]) * 1.05

sns.lineplot(x=obstime,y=fobs10, color='black', ax = ax[1])
ax[1].set_xlim([tmin,tmax])
ax[1].set_ylim([50,ymax])

ax[1].fill_between(x=ptime[fidx[0]], y1=smin10[fidx[0],2], y2=smax10[fidx[0],2], color='darkmagenta', alpha=0.1)
ax[1].fill_between(x=ptime[fidx[0]], y1=smin10[fidx[0],1], y2=smax10[fidx[0],1], color='darkmagenta', alpha=0.2)
ax[1].fill_between(x=ptime[fidx[0]], y1=smin10[fidx[0],0], y2=smax10[fidx[0],0], color='darkmagenta', alpha=0.3)

sns.lineplot(x=obstime, y=fobs10_sm_nz, color='blue', linewidth = 4, ax = ax[1])

idx = np.where(pmin10 > 0.0)
ax[1].fill_between(x=ptime10[idx], y1=pmin10[idx], y2=pmax10[idx], color='red', alpha=0.3)

sns.lineplot(x=ptime,y=f10, color='darkmagenta', ax = ax[1])

#------------------------------------------------------------------------------
def checktime(t1, t2, t3):

  rng = []
  rng.append(np.min((t1, t2, t3)))
  rng.append(np.max((t1, t2, t3)))

  return rng

#------------------------------------------------------------------------------
# compute amplitude and date ranges based on median values

i = np.argmax(f)
i1 = np.argmax(smin[:,1])
i2 = np.argmax(smax[:,1])
arange = [int(smin[i1,1]), int(smax[i2,1])]
trange = checktime(ptime[i1], ptime[i2], ptime[i])

i = np.argmax(f10)
i1 = np.argmax(smin10[:,1])
i2 = np.argmax(smax10[:,1])
arange10 = [int(smin10[i1,1]), int(smax10[i2,1])]
trange10 = checktime(ptime[i1], ptime[i2], ptime[i])

#------------------------------------------------------------------------------
# labels

lab1 = f"Cycle 25 Predicted Max: {arange[0]} - {arange[1]}"
if trange[0].year == trange[1].year:
  lab2 = f"In: {month[trange[0].month]} - {month[trange[1].month]}, {trange[1].year}"
else:
  lab2 = f"In: {month[trange[0].month]}, {trange[0].year} - {month[trange[1].month]}, {trange[1].year}"

xx = .69
yy = .92
dy = .034

ax[0].annotate("International Sunspot Number", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='black', ha='center', weight = 'bold')
yy -= dy
ax[0].annotate(lab1, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center')
yy -= dy
ax[0].annotate(lab2, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center')

yy = .46

lab1 = f"Cycle 25 Predicted Max: {arange10[0]} - {arange10[1]}"
if trange10[0].year == trange10[1].year:
  lab2 = f"In: {month[trange10[0].month]} - {month[trange10[1].month]}, {trange10[1].year}"
else:
  lab2 = f"In: {month[trange10[0].month]}, {trange10[0].year} - {month[trange10[1].month]}, {trange10[1].year}"

ax[1].annotate("F10.7cm Radio Flux", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='black', ha='center', weight = 'bold')
yy -= dy
ax[1].annotate(lab1, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center')
yy -= dy
ax[1].annotate(lab2, (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='darkmagenta', ha='center')

#--------------------

#ax[1].annotate("2019 NOAA/NASA/ISES Mean Panel Prediction", (.5,.5), xytext=(.47,.13),xycoords='figure fraction',color='red', ha='center')
#ax[0].annotate("2019 NOAA/NASA/ISES Mean Panel Prediction", (.5,.5), xytext=(.48,.6),xycoords='figure fraction',color='red', ha='center')
ax[1].annotate("2019 NOAA/NASA/ISES Panel Prediction", (.5,.5), xytext=(.49,.13),xycoords='figure fraction',color='red', ha='center')
ax[0].annotate("2019 NOAA/NASA/ISES Panel Prediction", (.5,.5), xytext=(.5,.6),xycoords='figure fraction',color='red', ha='center')

xx = .24
yy = .72
dy = .03

ax[0].annotate("smoothed", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='blue', ha='center',weight='bold')
yy -= dy
ax[0].annotate("observations", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='blue', ha='center',weight='bold')

yy = .3
ax[1].annotate("smoothed", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='blue', ha='center',weight='bold')
yy -= dy
ax[1].annotate("observations", (.5,.5), xytext=(xx,yy),xycoords='figure fraction',color='blue', ha='center',weight='bold')

qlabels(ax[0], top = True)
qlabels(ax[1])

plt.ylabel('F10.7 Flux (solar flux units)',fontsize=16)
plt.xlabel('Universal Time',fontsize=16)

plt.savefig(outfig, dpi = 100)

# save another copy for the archive
if archive == True:

    dir = outdir + '/archive'
    os.makedirs(dir, exist_ok = True)

    basename = os.path.basename(outfig).split('.png')[0]

    fname = f"{dir}/{basename}_{obstime[-1].year}_{obstime[-1].month}.png"
    plt.savefig(fname, dpi = 100)

#------------------------------------------------------------------------------
plt.show()