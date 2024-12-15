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

prediction_times = ['2022_01', '2023_01']

# set this to false to plot F10.7 instead of SSN
plot_ssn = True

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

    pbase[idx] = d['predicted_ssn']

  ptime_list.append(np.array(ptime))
  pbase_list.append(pbase)
  plower_list.append(plower)
  pupper_list.append(pupper)
  pbase10_list.append(pbase10)
  plower10_list.append(plower10)
  pupper10_list.append(pupper10)

#------------------------------------------------------------------------------
# plot out results.  Show SSN and F10.7 in separate files for inclusion publications

plt.rcParams.update({'font.size': 12., 'font.weight': 'bold'})

fig, ax = plt.subplots(1,2,figsize=[12,4])

ax[0].plot(obstime, ssn, color='black')
ax[0].plot(obstime[:-6], ssn_sm[:-6], color='blue', linewidth = 4)
ax[0].plot(ptime_list[0], pbase_list[0], color='darkmagenta')

ax[1].plot(obstime, ssn, color='black')
ax[1].plot(obstime[:-6], ssn_sm[:-6], color='blue', linewidth = 4)
ax[1].plot(ptime_list[1], pbase_list[1], color='darkmagenta')

tmin = datetime.date(2020,1,15)
tmax = datetime.date(2032,1,1)

ax[0].set_ylabel('SSN', weight = 'bold')

minor_locator = AutoMinorLocator(2)

for a in ax:
  a.yaxis.set_tick_params(labelsize=12)
  a.xaxis.set_minor_locator(minor_locator)
  a.yaxis.set_minor_locator(minor_locator)
  a.set_xlabel('year', weight = 'bold')
  a.set_xlim(tmin, tmax)
  a.set_ylim(0, 200)


fig.tight_layout(rect=(0.02,0.18,0.99,.98))

#------------------------------------------------------------------------------

print(obstime[-6:])

plt.show()