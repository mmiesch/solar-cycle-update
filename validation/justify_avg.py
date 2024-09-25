"""
The purpose of this function is to justify why I'm averaging the current prediction with the one mad 9 months ago.  Use Cycle 24 as an example.
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

deltak = 9

# 19 + 5 = cycle 24
cycle_idx = 19

#------------------------------------------------------------------------------
# read observations

tmon, d, dsm, t1 = u.get_cycles(tstart = True)
print(f"LEN {len(t1)}")
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
# choose years for illustration
years = [3,5]

Ny = len(years)

f = np.zeros((2, Ny, nobs))

ptime = []

for iyear in np.arange(Ny):

  k = 12*years[iyear]

  print(f"Prediction date: {time[k]}")
  ptime.append(time[k])

  afit = curve_fit(u.fpanel,tobs[0:k],ssn[0:k],p0=(170.0,0.0))
  fk = u.fpanel(tobs,afit[0][0],afit[0][1])
  f[0,iyear,:] = fk

  k2 = k - deltak
  afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
  fk2 = u.fpanel(tobs,afit2[0][0],afit2[0][1])
  f[1,iyear,:] = 0.5*(fk+fk2)

#------------------------------------------------------------------------------
# plot

plt.rc("font", weight = 'bold')
plt.rc("font", size = 14)

ssn_sm_nz = np.ma.masked_less(ssn_sm, 0.0)

tmin = np.min(time)
tmax = np.max(time)

sns.set_theme(style={'axes.facecolor': '#F5F5F5'}, palette='colorblind')

fig, ax = plt.subplots(1, 2, figsize = [12,5])

ymax = np.max(ssn) + 30

#------------------------------------------------------------------------------

sns.lineplot(x=time, y=ssn, color='black', linestyle=':', ax=ax[0])
ax[0].set_xlim([tmin,tmax])
ax[0].set_ylim([0,ymax])

#sns.lineplot(x=[time[k],time[k]],y=[0,ymax], color='green', linestyle='--', linewidth = 8, ax=ax[0])
x = np.array([ptime[0],ptime[0]])
y = np.array([0,ymax])
ax[0].plot(x,y, color='green', linestyle='--', linewidth = 4)

sns.lineplot(x=time, y=ssn_sm_nz, color='black', linewidth = 4, ax=ax[0])
sns.lineplot(x=time, y=f[0,0,:], color='red', linewidth = 4, ax=ax[0])
sns.lineplot(x=time, y=f[1,0,:], color='blue', linewidth = 4, ax=ax[0])

ax[0].set_ylabel('SSN', fontweight = "bold")
ax[0].set_xlabel('Date', fontweight = "bold")

#------------------------------------------------------------------------------

sns.lineplot(x=time, y=ssn, color='black', linestyle=':', ax=ax[1])
ax[1].set_xlim([tmin,tmax])
ax[1].set_ylim([0,ymax])

x = np.array([ptime[1],ptime[1]])
ax[1].plot(x,y, color='green', linestyle='--', linewidth = 4)

sns.lineplot(x=time, y=ssn_sm_nz, color='black', linewidth = 4, ax=ax[1])
sns.lineplot(x=time, y=f[0,1,:], color='red', linewidth = 4, ax=ax[1])
sns.lineplot(x=time, y=f[1,1,:], color='blue', linewidth = 4, ax=ax[1])

ax[1].set_xlabel('Date', fontweight = "bold")

#------------------------------------------------------------------------------
# annotate

x1 = .1
y1 = .85

x2 = .59
y2 = y1

ax[0].annotate(f"(a)", (x1,y2), xycoords='figure fraction', weight = "bold", fontsize = 16, family = 'serif', style = "italic")

ax[1].annotate(f"(b)", (x2,y2), xycoords='figure fraction', weight = "bold", fontsize = 16, family = 'serif', style = "italic")

#------------------------------------------------------------------------------

fig.tight_layout()

#------------------------------------------------------------------------------
# save to a file

indir, outdir, valdir = u.get_data_dirs()

dir = valdir + '/output/'

fname = f"justify_avg.png"

plt.savefig(dir+fname, dpi=300)

#------------------------------------------------------------------------------
plt.show()