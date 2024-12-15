"""
The intention of this function is to make something like my movie_maker movie but with jus a few frames, for the purpose of including in a publication.
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
from matplotlib.ticker import AutoMinorLocator

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# Preliminaries

# the two prediction times you want to plot

prediction_times = ['2021_12', '2022_12']

# set this to false to plot F10.7 instead of SSN
plot_ssn = False

#------------------------------------------------------------------------------
# Define files

obsfile, ssnfile, f10file, resfile = u.ops_input_files()

indir, outdir, valdir = u.get_data_dirs()

# Figure product
outfig_ssn = outdir + '/movieframes_ssn.png'
outfig_f10 = outdir + '/movieframes_f10.png'

# directory containing the archived json files
jsondir = valdir + "/reanalysis/archive"

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
# read the predictions

ptime_list = []
pbase_list = []
plower_list = []
pupper_list = []
pbase10_list = []
plower10_list = []
pupper10_list = []

for pt in prediction_times:

  jfile = jsondir+f'/predicted-solar-cycle_{pt}.json'
  print(f"file {jfile}")

  pdata = json.loads(open(jfile).read())
  Nt = len(pdata)

  ptime = []

  pbase = np.zeros(Nt)
  plower = np.zeros((Nt,3))
  pupper = np.zeros((Nt,3))

  pbase10 = np.zeros(Nt)
  plower10 = np.zeros((Nt,3))
  pupper10 = np.zeros((Nt,3))

  # Read the JSON file
  for idx, d in enumerate(pdata):
    t = np.array(d['time-tag'].split('-'), dtype='int')
    ptime.append(datetime.date(t[0], t[1], 15))

    pbase[idx]    = d['predicted_ssn']
    plower[idx,0] = d['low25_ssn']
    plower[idx,1] = d['low_ssn']
    plower[idx,2] = d['low75_ssn']
    pupper[idx,0] = d['high25_ssn']
    pupper[idx,1] = d['high_ssn']
    pupper[idx,2] = d['high75_ssn']

    pbase10[idx]    = d['predicted_f10.7']
    plower10[idx,0] = d['low25_f10.7']
    plower10[idx,1] = d['low_f10.7']
    plower10[idx,2] = d['low75_f10.7']
    pupper10[idx,0] = d['high25_f10.7']
    pupper10[idx,1] = d['high_f10.7']
    pupper10[idx,2] = d['high75_f10.7']

  ptime_list.append(np.array(ptime))
  pbase_list.append(pbase)
  plower_list.append(plower)
  pupper_list.append(pupper)
  pbase10_list.append(pbase10)
  plower10_list.append(plower10)
  pupper10_list.append(pupper10)

#------------------------------------------------------------------------------
# pick your plot

if plot_ssn:
  pobs = ssn
  pobs_sm = ssn_sm
  pbs  = pbase_list
  plow = plower_list
  pup  = pupper_list
else:
  pobs = fobs10
  pobs_sm = fobs10_sm
  pbs  = pbase10_list
  plow = plower10_list
  pup  = pupper10_list

#------------------------------------------------------------------------------
# plot out results.  Show SSN and F10.7 in separate files for inclusion publications

plt.rcParams.update({'font.size': 12., 'font.weight': 'bold'})

fig, ax = plt.subplots(1,2,figsize=[12,4])

pt = ptime_list[0][5]
idx0 = np.where(obstime < pt)

ax[0].plot(obstime[idx0], pobs[idx0], color='black')
ax[0].plot(obstime[:-6], pobs_sm[:-6], color='blue', linewidth = 4)

ax[0].fill_between(x=ptime_list[0][6:], y1=plow[0][6:,0], y2=pup[0][6:,0], color='darkmagenta', alpha=0.3)
ax[0].fill_between(x=ptime_list[0][6:], y1=plow[0][6:,1], y2=pup[0][6:,1], color='darkmagenta', alpha=0.2)
ax[0].fill_between(x=ptime_list[0][6:], y1=plow[0][6:,2], y2=pup[0][6:,2], color='darkmagenta', alpha=0.1)

ax[0].plot(ptime_list[0], pbs[0], color='darkmagenta')
ax[0].axvline(x=pt, color='black', linestyle='--', linewidth=1)

pt = ptime_list[1][5]
idx1 = np.where(obstime < pt)
ax[1].plot(obstime[idx1], pobs[idx1], color='black')
ax[1].plot(obstime[:-6], pobs_sm[:-6], color='blue', linewidth = 4)
ax[1].plot(ptime_list[1], pbs[1], color='darkmagenta')
ax[1].axvline(x=pt, color='black', linestyle='--', linewidth=1)

ax[1].fill_between(x=ptime_list[1][6:], y1=plow[1][6:,0], y2=pup[1][6:,0], color='darkmagenta', alpha=0.3)
ax[1].fill_between(x=ptime_list[1][6:], y1=plow[1][6:,1], y2=pup[1][6:,1], color='darkmagenta', alpha=0.2)
ax[1].fill_between(x=ptime_list[1][6:], y1=plow[1][6:,2], y2=pup[1][6:,2], color='darkmagenta', alpha=0.1)

tmin = datetime.date(2020,1,15)
tmax = datetime.date(2030,1,1)

if plot_ssn:
  ax[0].set_ylabel('SSN V2', weight = 'bold')
else:
  ax[0].set_ylabel('F10.7 Radio Flux (s.f.u.)', weight = 'bold')

minor_locator = AutoMinorLocator(2)

for a in ax:
  a.yaxis.set_tick_params(labelsize=12)
  a.xaxis.set_major_locator(mdates.YearLocator(2))
  a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
  a.xaxis.set_minor_locator(minor_locator)
  a.yaxis.set_minor_locator(minor_locator)
  a.set_xlabel('year', weight = 'bold')
  a.set_xlim(tmin, tmax)
  if plot_ssn:
    a.set_ylim(0, 200)
  else:
    a.set_ylim(60, 220)


#fig.tight_layout(rect=(0.02,0.18,0.99,.98))
fig.tight_layout()

ax[0].annotate("(a)", (.438,.84), xycoords='figure fraction', weight = "bold")
ax[1].annotate("(b)", (.92,.84), xycoords='figure fraction', weight = "bold")

#------------------------------------------------------------------------------
# save the figure

if plot_ssn:
  plt.savefig(outfig_ssn)
else:
  plt.savefig(outfig_f10)

#------------------------------------------------------------------------------

print("Number of fit points")
print(f'{len(idx0[0])} {len(idx1[0])}')

#plt.show()