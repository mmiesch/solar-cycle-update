"""
The intention of this function is to overplot three curves:
* the smoothed SSN at time t
* the prediction made at t - 1 year
* the prediction made at t - 2 years

I could add other curves to it, like the unsmoothed obs, but let's start with these as the main goal.
"""
import datetime
import json
import numpy as np
import matplotlib.dates as mdates
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
# Preliminaries

# the two lead times you want to plot, in years
# currently it is assumed that these are integers

lead_times = [1, 2]

#------------------------------------------------------------------------------
# Define files

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

indir, outdir, valdir = u.get_data_dirs()

# Figure product
outfig = outdir + '/leadtime.png'

# official start time of cycle 25 from SIDC, in decimal year
tstart = 2019.96

# minimum date for prediction
mindate = datetime.date(2021,12,15)

# directory containing the archived json files
jsondir = valdir + "/reanalysis/archive"

print(f"obsfile = {obsfile}")

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

# time of available observations in decimal year
nobs = len(obstime)
odect = np.zeros(nobs)
for i in np.arange(nobs):
    dt = datetime.datetime(obstime[i].year,obstime[i].month,obstime[i].day)
    odect[i] = Time(dt).to_value('decimalyear')

#------------------------------------------------------------------------------
# read the predictions

# time in months since cycle beginning
tobs = (odect - tstart)*12

# predictions
nlt = len(lead_times)
Nq = 3
pp = np.zeros((nlt,nobs)) - 1
pmin = np.zeros((nlt,nobs,Nq)) - 1
pmax = np.zeros((nlt,nobs,Nq)) - 1

pp10 = np.zeros((nlt,nobs)) - 1
pmin10 = np.zeros((nlt,nobs,Nq)) - 1
pmax10 = np.zeros((nlt,nobs,Nq)) - 1

cmonth = np.rint(tobs[-1]).astype(np.int32)
print(f"Current month = {cmonth}")

for itime in np.arange(len(obstime)):

  time = obstime[itime]

  tag = f"{time.year}-{time.month:02d}"

  for il in np.arange(nlt):
    pyear = time.year - lead_times[il]
    pmonth = time.month
    if datetime.date(pyear,pmonth,15) < mindate:
      continue
    jfile = jsondir+f'/predicted-solar-cycle_{pyear}_{pmonth:02d}.json'
    print(f"MSM {tag} {jfile}")

    # Read the JSON file
    with open(jfile, 'r') as file:
      data = json.load(file)

      # Find the record with the matching "time-tag"
      for record in data:
        if record.get("time-tag") == tag:
          print(f"Found record: {record}")
          pp[il,itime] = record.get("predicted_ssn")
          pmin[il,itime,0] = record.get("low25_ssn")
          pmin[il,itime,1] = record.get("low_ssn")
          pmin[il,itime,2] = record.get("low75_ssn")
          pmax[il,itime,0] = record.get("high25_ssn")
          pmax[il,itime,1] = record.get("high_ssn")
          pmax[il,itime,2] = record.get("high75_ssn")

          pp10[il,itime] = record.get("predicted_f10.7")
          pmin10[il,itime,0] = record.get("low25_f10.7")
          pmin10[il,itime,1] = record.get("low_f10.7")
          pmin10[il,itime,2] = record.get("low75_f10.7")
          pmax10[il,itime,0] = record.get("high25_f10.7")
          pmax10[il,itime,1] = record.get("high_f10.7")
          pmax10[il,itime,2] = record.get("high75_f10.7")
          break

#------------------------------------------------------------------------------
# plot out results.  Show SSN and F10.7 in separate files for 
# inclusion in presentations
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

#sns.set_theme(style={'axes.facecolor': '#F5F5F5'}, palette='colorblind')

fig, ax = plt.subplots(1, 2, figsize = [12,4])

sns.lineplot(x = obstime[:-6], y = ssn_sm[:-6], color='blue', label='Smoothed SSN', linewidth=4, ax=ax[0])

#sns.lineplot(x = obstime, y = ssn, color='gray', label='SSN', alpha=0.2)

idx = np.where(pp[0,:] > 0)
sns.lineplot(x = obstime[idx], y = pp[0,idx[0]], color='darkmagenta', label='1 year lead time', ax=ax[0])
ax[0].fill_between(obstime[idx], y1 = pmin[0,idx[0],2], y2 = pmax[0,idx[0],2], color='darkmagenta', alpha=0.1)
#ax[0].fill_between(obstime[idx], y1 = pmin[0,idx[0],1], y2 = pmax[0,idx[0],1], color='darkmagenta', alpha=0.2)
#ax[0].fill_between(obstime[idx], y1 = pmin[0,idx[0],0], y2 = pmax[0,idx[0],0], color='darkmagenta', alpha=0.3)

idx = np.where(pp[1,:] > 0)
sns.lineplot(x = obstime[idx], y = pp[1,idx[0]], color='red', label='2 year lead time', ax=ax[0])
ax[0].fill_between(obstime[idx], y1 = pmin[1,idx[0],2], y2 = pmax[1,idx[0],2], color='red', alpha=0.1)
#ax[0].fill_between(obstime[idx], y1 = pmin[1,idx[0],1], y2 = pmax[1,idx[0],1], color='red', alpha=0.2)
#ax[0].fill_between(obstime[idx], y1 = pmin[1,idx[0],0], y2 = pmax[1,idx[0],0], color='red', alpha=0.3)

#------------------------------------------------------------------------------
#  Now F10.7

sns.lineplot(x = obstime[:-6], y = fobs10_sm[:-6], color='blue', label='Smoothed F10.7',linewidth=4, ax=ax[1])

idx = np.where(pp10[0,:] > 0)
sns.lineplot(x = obstime[idx], y = pp10[0,idx[0]], color='darkmagenta', label='F10.7 1 year lead time', ax=ax[1])
ax[1].fill_between(obstime[idx], y1 = pmin10[0,idx[0],2], y2 = pmax10[0,idx[0],2], color='darkmagenta', alpha=0.1)
#ax[1].fill_between(obstime[idx], y1 = pmin10[0,idx[0],1], y2 = pmax10[0,idx[0],1], color='darkmagenta', alpha=0.2)
#ax[1].fill_between(obstime[idx], y1 = pmin10[0,idx[0],0], y2 = pmax10[0,idx[0],0], color='darkmagenta', alpha=0.3)

idx = np.where(pp10[1,:] > 0)
sns.lineplot(x = obstime[idx], y = pp10[1,idx[0]], color='red', label='F10.7 2 year lead time', ax=ax[1])
ax[1].fill_between(obstime[idx], y1 = pmin10[1,idx[0],2], y2 = pmax10[1,idx[0],2], color='red', alpha=0.1)
#ax[1].fill_between(obstime[idx], y1 = pmin10[1,idx[0],1], y2 = pmax10[1,idx[0],1], color='red', alpha=0.2)
#ax[1].fill_between(obstime[idx], y1 = pmin10[1,idx[0],0], y2 = pmax10[1,idx[0],0], color='red', alpha=0.3)

# annotate and write to a file

for a in ax:
  a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
  a.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
  a.tick_params(axis='both', which='major', labelsize=12)
  a.set_xlabel('Date')
  a.set_xlim([datetime.date(2022,12,1),datetime.date(obstime[-6].year,obstime[-6].month,1)])
  a.legend().set_visible(False)

ax[0].set_ylabel('SSN V2')
ax[1].set_ylabel('F10.7 radio flux (sfu)')

ax[0].annotate("(a)", (.12,.84), xycoords='figure fraction', weight="bold", fontsize=16, family = 'serif', style = 'italic')
ax[1].annotate("(b)", (.6 ,.84), xycoords='figure fraction', weight="bold", fontsize=16, family = 'serif', style = 'italic')

fig.tight_layout()

plt.savefig(outfig, dpi=300)

#------------------------------------------------------------------------------

print(obstime[-6])

#plt.show()