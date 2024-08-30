"""
The intention of this function is to overplot three curves:
* the smoothed SSN at time t
* the prediction made at t - 1 year
* the prediction made at t - 3 years

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

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# Preliminaries

# the two lead times you want to plot, in months

lead_times = [12, 36]

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
# plot out results.  Show SSN and F10.7 in separate files for 
# inclusion in presentations

plt.rc("font", weight = 'bold')

sns.set_theme(style={'axes.facecolor': '#F5F5F5'}, palette='colorblind')

fig1 = plt.figure(figsize = [6,3], dpi = 300)

ax = sns.lineplot(x = obstime[:-6], y = ssn_sm[:-6], color='black', label='Smoothed SSN')

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.set_ylabel('SSN', fontweight='bold')
ax.set_xlabel('Date', fontweight='bold')

ax.legend().set_visible(False)

fig1.tight_layout()
plt.savefig(outfig_ssn)

plt.show()