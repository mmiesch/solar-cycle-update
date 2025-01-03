"""
This code reads the cycle N+1 prediction computed by next_cycle_1dkde.py and plots it out.  The plotting is done in different files for display on presentation slides.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys

from astropy.time import Time
from astropy import units
from scipy.io import netcdf_file
from scipy.signal import savgol_filter

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# data directories
indir, outdir, valdir = u.get_data_dirs()

#------------------------------------------------------------------------------
# read Cycle N+1 SSN prediction

pfile = outdir+'/nextcycle/nextcycle_1dkde.nc'

p = netcdf_file(pfile,'r')
print(p.history)

# datetimes are represented as decimal year in the netcdf file
a = p.variables['Cycle']
cycle = a[:].copy()

a = p.variables['Cycle Periods']
per = a[:].copy()

a = p.variables['prediction time']
ptime_decyr = a[:].copy()

a = p.variables['start time']
stime_decyr = a[:].copy()

a = p.variables['Cycle N period']
periodN_decyr = a[:].copy()

a = p.variables['start time pdf']
pdf_tstart = a[:].copy()

a = p.variables['percentiles']
percentiles = a[:].copy()

a = p.variables['probabilistic prediction']
prediction = a[:].copy()

a = p.variables['prediction mode']
mode = a[:].copy()

del a
p.close()

#------------------------------------------------------------------------------
# convert decimal year to datetime

ptime =[]
for pt in ptime_decyr:
  ptime.append(datetime.datetime.fromisoformat(Time(pt,format='decimalyear').iso))
ptime = np.array(ptime)

stime =[]
for st in stime_decyr:
  stime.append(datetime.datetime.fromisoformat(Time(st,format='decimalyear').iso))
stime = np.array(stime)

# also compute the prediction time in months
seconds_per_year = units.year.to(units.second)
seconds_per_month = seconds_per_year / 12.0

ptime_months = np.zeros(len(ptime), dtype = np.float32)
for i in np.arange(len(ptime)):
  ptime_months[i] = (ptime[i]-ptime[0]).total_seconds()/seconds_per_month

#------------------------------------------------------------------------------
# compute average profile for reference

offset = -5.611901599023621
cycleN_tstart = datetime.datetime.fromisoformat(Time(2019.96,format='decimalyear').iso)

ctime, cssn, cssn_sm, cstart_times = u.get_cycles(tstart = True)
per = u.cycle_periods(cstart_times[6:])
camps = u.cycle_amps(cssn_sm[6:])

per_ref = np.mean(per)
smax_ref = np.mean(camps)
amp_ref = u.smax_to_amp(smax_ref)
print(f"reference period, smax: {per_ref} {smax_ref}")

# median cycle period
#per_ref = np.median(per)

print(f"reference period: {per_ref}")
tstart_ref = per_ref*12 - (ptime[0] - cycleN_tstart).total_seconds()/seconds_per_month

#smax_ref, apdf_ref, mu_ref = u.amp_pdf(per_ref, Namp = 1)
#print(f"reference start time, amp: {tstart_ref} {mu_ref}")
#amp_ref = u.smax_to_amp(mu_ref)

fref = u.fpanel(ptime_months, amp_ref, tstart_ref+offset)

ref_start = cycleN_tstart + datetime.timedelta(days = per_ref*365)
print(f"reference start date: {cycleN_tstart} {ref_start}")

ref_date = ptime[np.argmax(fref)]
print(f"reference max date: {ref_date}")

#------------------------------------------------------------------------------
# Plot setup

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

#------------------------------------------------------------------------------
# smooth for plotting

width = 16

p0 = savgol_filter(prediction[:,0], width, 3)
p1 = savgol_filter(prediction[:,1], width, 3)
p2 = savgol_filter(prediction[:,2], width, 3)
p3 = savgol_filter(prediction[:,3], width, 3)
p4 = savgol_filter(prediction[:,4], width, 3)

#------------------------------------------------------------------------------
fig = plt.figure(figsize=[12,5])

plt.fill_between(ptime,y1=p0,y2=p4,color='silver',alpha=0.8)
plt.fill_between(ptime,y1=p1,y2=p3,color='slategrey',alpha=0.8)

#plt.plot(ptime,mode,color='maroon', linewidth=2.0)
plt.plot(ptime,p2,color='black', linewidth=2.0)
plt.plot(ptime,fref,color='blue', linewidth=2.0)

ax = fig.axes

ax[0].grid(visible=True)

# max time to plot
tmax = datetime.datetime(2044,1,1)

ax[0].set_ylim(0,250)
ax[0].set_xlim(ptime[0],tmax)

ax[0].set_ylabel('SSNV2')
ax[0].set_xlabel('Date')

# annotate
bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white")
ax[0].annotate("$P(S|t)$", (.62,.74), xycoords='figure fraction', bbox=bbox_props, fontsize=18, family='serif',style='italic')
ax[0].annotate("95%", (.29,.68), xycoords='figure fraction', weight="bold", fontsize=14, family='serif')
ax[0].annotate("50%", (.35,.32), xycoords='figure fraction', weight="bold", fontsize=14, family='serif')

# optionally overplot test pdf
mtest = 75
#ax[0].axvline(ptime[mtest], color='red', linestyle='--')

fig.tight_layout()

plt.savefig(outdir+'/nextcycle/prediction.png', dpi = 300)
#plt.show()