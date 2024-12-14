"""
This is a validation for the method of fitting cycles and computing error bars.  It applies the method to previous cycles, one cycle at a time, to see how well it does.  Usage:

python validate_cycle.py 22

replace `22` here with other cycle numbers as desired.

"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.io import netcdf_file
from scipy.signal import savgol_filter

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# which cycle do you want to fit?

nargs = len(sys.argv)

if nargs < 2:
  cycle = 24
else:
  cycle = int(sys.argv[1])

print(f"Cycle {cycle}")

# exclude the first 5 cycles
cycle_idx = cycle - 5

# set this to generate figures for the paper
pub = True

#------------------------------------------------------------------------------
# optionally average an earlier fit for stability
# units are months.  Set to -1 to disable

deltak = 9

#------------------------------------------------------------------------------
# read observations

tmon, d, dsm, t1 = u.get_cycles(tstart = True)
t = t1[cycle_idx-1]
tstart = Time(datetime.datetime(t.year,t.month,15)).to_value('decimalyear')

tobs = tmon[cycle_idx-1]
ssn = d[cycle_idx-1]
ssn_sm = dsm[cycle_idx-1]

# time of available observations in decimal year
nobs = len(tobs)
tdec = np.zeros(nobs)
for i in np.arange(nobs):
   tdec[i] = tstart + tobs[i]/12.

# time as a datetime object
time = []
for i in np.arange(nobs):
   time.append(Time(tdec[i],format='decimalyear').datetime)

#------------------------------------------------------------------------------
# choose fit type
# 1: 2-parameter panel fit (A, t0)
# 2: 2-parameter upton-hathaway (A, t0)

ftype = 1

if ftype == 2:
  lab = "uh"
  plot_lab = "Upton-Hathaway 2023 fit"
else:
  lab = "panel2"
  plot_lab = "Panel fit"

if deltak > 0:
  lab = lab+f"_d{deltak}"
  plot_lab = plot_lab + f": -{deltak} month avg"

#------------------------------------------------------------------------------
# read average residuals for this fit type

indir, outdir, valdir = u.get_data_dirs()

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
# choose four years for illustration
if pub:
  #years = np.array([2, 3, 5, 8])
  years = np.array([2, 3, 4, 6])
else:
  years = np.array([3, 5, 7, 9])

klist = 12*years
Nsam = len(klist)

Nerr = len(rtime)
Nmax = np.min([nobs,Nerr])

f = np.zeros((Nsam, nobs))
perr = np.zeros((Nsam, nobs, 4))
nerr = np.zeros((Nsam, nobs, 4))

for kidx in np.arange(Nsam):

  k = klist[kidx]

  if ftype == 2:
    afit = curve_fit(u.fuh,tobs[0:k],ssn[0:k],p0=(170.0,0.0))
    fk = u.fuh(tobs,afit[0][0],afit[0][1])
  else:
    afit = curve_fit(u.fpanel,tobs[0:k],ssn[0:k],p0=(170.0,0.0))
    fk = u.fpanel(tobs,afit[0][0],afit[0][1])

  if (deltak > 0) and (k > (deltak + 23)):
    k2 = k - deltak
    if ftype == 2:
      afit2 = curve_fit(u.fuh,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
      fk2 = u.fuh(tobs,afit2[0][0],afit2[0][1])
    else:
      afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
      fk2 = u.fpanel(tobs,afit2[0][0],afit2[0][1])
    fk = 0.5*(fk+fk2)

  f[kidx,:] = fk

  kk = k - kmon[0]

  for i in np.arange(Nmax):
    for q in np.arange(4):
      perr[kidx,i,q] = presid[i,kk,q]
      nerr[kidx,i,q] = nresid[i,kk,q]

#------------------------------------------------------------------------------
# plot positions

p = []
p.append((0,0))
p.append((0,1))
p.append((1,0))
p.append((1,1))

#------------------------------------------------------------------------------
# plot

ssn_sm_nz = np.ma.masked_less(ssn_sm, 0.0)

tmin = np.min(time)
tmax = np.max(time)

sns.set_theme(style={'axes.facecolor': '#FFFFFF'}, palette='colorblind')

fig, ax = plt.subplots(2,2,figsize=[12,6])

if pub:
  ymax = 290

else:
  ymax = np.max(ssn) + 30
  title = f"Cycle {cycle}"
  fig.suptitle(title,fontsize=20,x=0.05,y=.97,fontweight='bold', horizontalalignment = 'left')

obsamp = []
amp = []
amp25 = []
amp50 = []
amp75 = []

obsdate = []
pdate = []
pdate25 = []
pdate50 = []
pdate75 = []

for iframe in np.arange(Nsam):

  a = ax[p[iframe][0],p[iframe][1]]

  a.plot(time,ssn, color='black', linestyle=':')
  a.set_xlim([tmin,tmax])
  a.set_ylim([0,ymax])

  rmin = f[iframe,:] - nerr[iframe,:,3]
  rmax = f[iframe,:] + perr[iframe,:,3]
  a.plot(time, rmin, color='red')
  a.plot(time, rmax, color='red')

  rmin = savgol_filter(f[iframe,:] - nerr[iframe,:,2],21,3)
  rmax = savgol_filter(f[iframe,:] + perr[iframe,:,2],21,3)
  a.fill_between(x=time, y1=rmin, y2=rmax, color='darkmagenta', alpha=0.1)
  amp75.append([np.max(rmin), np.max(rmax)])
  pdate75.append([time[np.argmax(rmin)], time[np.argmax(rmax)]])

  rmin = savgol_filter(f[iframe,:] - nerr[iframe,:,1],21,3)
  rmax = savgol_filter(f[iframe,:] + perr[iframe,:,1],21,3)
  a.fill_between(x=time, y1=rmin, y2=rmax, color='darkmagenta', alpha=0.2)
  amp50.append([np.max(rmin), np.max(rmax)])
  pdate50.append([time[np.argmax(rmin)], time[np.argmax(rmax)]])

  rmin = savgol_filter(f[iframe,:] - nerr[iframe,:,0],21,3)
  rmax = savgol_filter(f[iframe,:] + perr[iframe,:,0],21,3)
  a.fill_between(x=time, y1=rmin, y2=rmax, color='darkmagenta', alpha=0.3)
  amp25.append([np.max(rmin), np.max(rmax)])
  pdate25.append([time[np.argmax(rmin)], time[np.argmax(rmax)]])

  k = klist[iframe]
  a.plot(time[0:k], ssn_sm_nz[0:k], color='blue', linewidth = 4)
  obsamp.append(np.max(ssn_sm_nz))
  obsdate.append(time[np.argmax(ssn_sm_nz)])

  a.plot(time, f[iframe,:], color='darkmagenta')
  amp.append(np.max(f[iframe,:]))
  pdate.append(time[np.argmax(f[iframe,:])])

fig.tight_layout()

#------------------------------------------------------------------------------
# label

x1 = .41
y1 = .37

x2 = .91
y2 = .82

ax[0,0].annotate(f"{years[0]} years", (x1,y2), xycoords='figure fraction', weight = "bold")

ax[0,1].annotate(f"{years[1]} years", (x2,y2), xycoords='figure fraction', weight = "bold")

ax[1,0].annotate(f"{years[2]} years", (x1,y1), xycoords='figure fraction', weight = "bold")

ax[1,1].annotate(f"{years[3]} years", (x2,y1), xycoords='figure fraction', weight = "bold")

if pub:
  x1 = .09
  x2 = .58
  y1 = .41
  y2 = .89
  ax[0,0].annotate("(a)", (x1,y2), xycoords='figure fraction', weight = "bold")
  ax[0,1].annotate("(b)", (x2,y2), xycoords='figure fraction', weight = "bold")
  ax[1,0].annotate("(c)", (x1,y1), xycoords='figure fraction', weight = "bold")
  ax[1,1].annotate("(d)", (x2,y1), xycoords='figure fraction', weight = "bold")

  ax[0,0].set_ylabel("SSNV2")
  ax[1,0].set_ylabel("SSNV2")

  ax[1,0].set_xlabel("Date")
  ax[1,1].set_xlabel("Date")

  plt.subplots_adjust(bottom=0.1, left = 0.07)

else:
  ax[0,0].annotate(plot_lab, (.94,.92), xycoords='figure fraction', weight = "bold", fontsize = 12, horizontalalignment='right')

#------------------------------------------------------------------------------
# save to a file
dir = valdir + '/output/'

if pub:
  lab = lab + "_pub"

fname = f"validation_cycle{cycle}_{lab}.png"

plt.savefig(dir+fname)

#------------------------------------------------------------------------------
# print results for analysis

print(80*'='+f"\nCycle {cycle}")
camp = obsamp[0]
print(f'observed amplitude: {camp:.1f}')

print(f'\npredicted amplitude:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {amp[i]:.1f} {np.abs(amp[i]-camp)/camp:.3f}')

print(f'\n75 percentile:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {amp75[i][0]:.1f}--{amp75[i][1]:.1f}')

print(f'\n50 percentile:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {amp50[i][0]:.1f}--{amp50[i][1]:.1f}')

print(f'\n25 percentile:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {amp25[i][0]:.1f}--{amp25[i][1]:.1f}')

print('\n'+80*'-')
print(f'\nobserved date: {obsdate[0].date()}')

print(f'\npredicted date:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {pdate[i].date()}')

print(f'\n75 percentile:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {pdate75[i][1].date()}--{pdate75[i][0].date()}')

print(f'\n50 percentile:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {pdate50[i][1].date()}--{pdate50[i][0].date()}')

print(f'\n25 percentile:')
for i in np.arange(Nsam):
  print(f'{years[i]} years: {pdate25[i][1].date()}--{pdate25[i][0].date()}')

#------------------------------------------------------------------------------
plt.show()