"""
This script takes the output json generated by the operational code and plots it.
"""

import datetime
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# define files

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

indir, outdir, valdir = u.get_data_dirs()


# json product
jfile = outdir + "/predicted-solar-cycle.json"

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

ssn = np.array(ssn)
ssn_sm = np.array(ssn_sm)
fobs10 = np.array(fobs10)
fobs10_sm = np.array(fobs10_sm)

for i in np.arange(len(fobs10)):
   print(f"{obstime[i]} {ssn[i]} {ssn_sm[i]}")

#------------------------------------------------------------------------------
# read prediction file

pred_file = open(jfile)

data = json.loads(pred_file.read())
N = len(data)

ptime = []
pssn = []
pf10 = []

for d in data:
    t = np.array(d['time-tag'].split('-'), dtype='int')
    ptime.append(datetime.date(t[0], t[1], 15))
    pssn.append(d['predicted_ssn'])
    pf10.append(d['predicted_f10.7'])

ptime = np.array(ptime)
pssn = np.array(pssn)
pf10 = np.array(pf10)

#------------------------------------------------------------------------------
# copy last smoothed point into prediction for plotting continuity

ptime = np.insert(ptime, 0, obstime[-7])
pssn = np.insert(pssn, 0, ssn_sm[-7])
pf10 = np.insert(pf10, 0, fobs10_sm[-7])

#------------------------------------------------------------------------------
# plot SSN

fig, ax = plt.subplots(2, 1, figsize = [12.8,6.5])
fig.tight_layout(rect=(0.14,0.04,0.9,1.))
ax[0].xaxis.set_tick_params(labelsize=14)
ax[1].xaxis.set_tick_params(labelsize=14)
ax[0].yaxis.set_tick_params(labelsize=12)
ax[1].yaxis.set_tick_params(labelsize=12)

tmax = datetime.date(2032,1,1)
ymax = np.max([np.max(ssn),np.max(pssn)]) * 1.05

print(np.max(pssn))

sns.set_theme(style={'axes.facecolor': '#FFFFFF'}, palette='colorblind')

sns.lineplot(x=obstime,y=ssn, color='black', ax = ax[0])
ax[0].set_xlim([tmin,tmax])
ax[0].set_ylim([0,ymax])

ssn_sm_nz = np.ma.masked_less(ssn_sm, 0.0)

sns.lineplot(x=obstime, y=ssn_sm_nz, color='blue', linewidth = 4, ax = ax[0])

sns.lineplot(x=ptime,y=pssn, color='darkmagenta', ax = ax[0])

#------------------------------------------------------------------------------
# plot f10.7

fobs10_sm_nz = np.ma.masked_less(fobs10_sm, 0.0)

ymax = np.max([np.max(fobs10),np.max(pf10)]) * 1.05

sns.lineplot(x=obstime,y=fobs10, color='black', ax = ax[1])
ax[1].set_xlim([tmin,tmax])
ax[1].set_ylim([50,ymax])

sns.lineplot(x=obstime, y=fobs10_sm_nz, color='blue', linewidth = 4, ax = ax[1])

sns.lineplot(x=ptime,y=pf10, color='darkmagenta', ax = ax[1])

#------------------------------------------------------------------------------
tmin = datetime.date(2022,11,1)
tmax = datetime.date(2023,11,1)

#ax[0].set_xlim([tmin,tmax])
#ax[1].set_xlim([tmin,tmax])

plt.show()