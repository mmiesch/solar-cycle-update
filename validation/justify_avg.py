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
# choose year for illustration
year = 3
k = 12*year

f = np.zeros((2, nobs))

afit = curve_fit(u.fpanel,tobs[0:k],ssn[0:k],p0=(170.0,0.0))
fk = u.fpanel(tobs,afit[0][0],afit[0][1])
f[0,:] = fk

k2 = k - deltak
afit2 = curve_fit(u.fpanel,tobs[0:k2],ssn[0:k2],p0=(170.0,0.0))
fk2 = u.fpanel(tobs,afit2[0][0],afit2[0][1])
f[1,:] = 0.5*(fk+fk2)

#------------------------------------------------------------------------------
# plot

ssn_sm_nz = np.ma.masked_less(ssn_sm, 0.0)

tmin = np.min(time)
tmax = np.max(time)

sns.set_theme(style={'axes.facecolor': '#FFFFFF'}, palette='colorblind')

fig, ax = plt.subplots(1,2,figsize=[12,4])

ymax = np.max(ssn) + 30

a = ax[0]

a.plot(time,ssn, color='black', linestyle=':')
a.set_xlim([tmin,tmax])
a.set_ylim([0,ymax])

a.plot([time[k],time[k]],[0,ymax], color='red', linestyle='--')

a.plot(time, ssn_sm_nz, color='black', linewidth = 4)
a.plot(time, f[0,:], color='blue')

a.set_ylabel('SSN')
a.set_xlabel('date')

a = ax[1]

a.plot(time,ssn, color='black', linestyle=':')
a.set_xlim([tmin,tmax])
a.set_ylim([0,ymax])

a.plot([time[k],time[k]],[0,ymax], color='red', linestyle='--')

a.plot(time, ssn_sm_nz, color='black', linewidth = 4)
a.plot(time, f[1,:], color='blue')
a.set_xlabel('date')

fig.tight_layout()

#------------------------------------------------------------------------------
# label

x1 = .1
y1 = .9

x2 = .6
y2 = .9

plt.rcParams['text.usetex'] = True

a.annotate("($a$)", (x1,y2), xycoords='figure fraction', weight = "bold", fontsize = 14)

#------------------------------------------------------------------------------
# save to a file

indir, outdir, valdir = u.get_data_dirs()

dir = valdir + '/output/'

fname = f"justify_avg.png"

plt.savefig(dir+fname)

#------------------------------------------------------------------------------
plt.show()