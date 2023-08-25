"""
This is a validation for the method of fitting cycles and computing error bars.  It applies the method to previous cycles, one cycle at a time, to see how well it does.  Usage:

python validate_cycle.py 22

replace `22` here with other cycle numbers as desired.

"""

import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.io import netcdf_file

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

#------------------------------------------------------------------------------
# optionally average an earlier fit for stability
# units are months.  Set to -1 to disable

deltak = -1
#deltak = 9

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
  plot_lab = plot_lab + f"-{deltak} month avg"

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
years = np.array([3, 5, 7, 9])
klist = 12*years - 1
Nsam = len(klist)

Nerr = len(rtime)
Nmax = np.min([nobs,Nerr])

f = np.zeros((Nsam, nobs))

for kidx in np.arange(Nsam):

  k = klist[kidx]

  if ftype == 2:
    afit = curve_fit(u.fhath,tobs[0:k],ssn[0:k],p0=(170.0,0.0))
    fk = u.fhath(tobs,afit[0][0],afit[0][1])
  if ftype == 3:
    afit = curve_fit(u.fuh,tobs[0:k],ssn[0:k],p0=(170.0,0.0))
    fk = u.fuh(tobs,afit[0][0],afit[0][1])
  else:
    afit = curve_fit(u.fpanel,tobs[0:k],ssn[0:k],p0=(170.0,0.0))
    fk = u.fpanel(tobs,afit[0][0],afit[0][1])

  if (deltak > 0) and (k > (deltak + 23)):
    k2 = k - deltak
    if ftype == 2:
      afit2 = curve_fit(u.fhath,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
      fk2 = u.fhath(tobs,afit2[0][0],afit2[0][1])
    elif ftype == 3:
      afit2 = curve_fit(u.fuh,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
      fk2 = u.fuh(tobs,afit2[0][0],afit2[0][1])
    else:
      afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
      fk2 = u.fpanel(tobs,afit2[0][0],afit2[0][1])
    fk = 0.5*(fk+fk2)

  f[kidx,:] = fk

  kk = k - kmon[0]

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

title = f"Cycle {cycle}"
fig.suptitle(title,fontsize=20,x=0.05,y=.97,fontweight='bold', horizontalalignment = 'left')

ymax = np.max(ssn) + 30

for iframe in np.arange(Nsam):

  a = ax[p[iframe][0],p[iframe][1]]

  a.plot(time,ssn, color='black', linestyle=':')
  a.set_xlim([tmin,tmax])
  a.set_ylim([0,ymax])

  #rmin = f[iframe,:] - nerr[iframe,:]
  #rmax = f[iframe,:] + perr[iframe,:]
  #a.fill_between(x=time, y1=rmin, y2=rmax, color='blue', alpha=0.3)

  k = klist[iframe]
  a.plot(time[0:k], ssn_sm_nz[0:k], color='black', linewidth = 4)
  a.plot(time, f[iframe,:], color='blue')

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

ax[0,0].annotate(plot_lab, (.94,.92), xycoords='figure fraction', weight = "bold", fontsize = 12, horizontalalignment='right')

#------------------------------------------------------------------------------
# save to a file
dir = valdir + '/output/'

fname = f"validation_cycle{cycle}_{lab}.png"

plt.savefig(dir+fname)

#------------------------------------------------------------------------------
plt.show()