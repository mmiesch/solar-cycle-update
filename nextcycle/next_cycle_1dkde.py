"""
This code is used to make a prediction for the next cycle before it begins
based on statistics of previous cycles.  It computes a 1D KDE for each prediction month.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import os
import sys
from astropy.time import Time
from astropy import units
from matplotlib.ticker import AutoMinorLocator
from sklearn.neighbors import KernelDensity
from scipy.io import netcdf_file

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# start with some definitions

# define current cycle number
this_cycle_number = 25

# Start time for current cycle, Cycle N (assuming N=25)
cycleN_tstart = datetime.datetime.fromisoformat(Time(2019.96,format='decimalyear').iso)

print(f"Cycle N start time: {cycleN_tstart}")

# Optimal offset between tstart and t0
# computed with the optimize_offset.py script
offset = -5.611901599023621

# conversion factors
seconds_per_year = units.year.to(units.second)
days_per_year = seconds_per_year / (3600.*24)
days_per_month = days_per_year / 12.0
seconds_per_month = seconds_per_year / 12.0

# optionally define a minimum cycle amplitude
amp_min = 0

# This runs faster, for development and debugging
dev = False

# illustrative slice
#mtest = 45
#mtest = 55
mtest = 76

# data directories
indir, outdir, valdir = u.get_data_dirs()

#------------------------------------------------------------------------------
# get cycle data to work with

time, ssn, ssn_sm, start_times = u.get_cycles(full = True, tstart = True)

Ncycles = len(ssn_sm)
print(f"Number of cycles: {Ncycles}")

# period and max ssn arrays for all cycles
per = u.cycle_periods(start_times)

#------------------------------------------------------------------------------
# Now estimate the pdf that provides a probabilistic prediction for the start
# date of Cycle N+1, P(tstart)

kde_tstart = KernelDensity(kernel='gaussian', bandwidth = .5).fit(per[:, np.newaxis])

# Sample pdf for computing the 99% confidence interval
# specifying random_state ensures reproducibility
if dev:
  Nsam = 1000
else:
  Nsam = 10000

tstart_sample = kde_tstart.sample(Nsam, random_state = 42)
prange = np.percentile(tstart_sample, [0.5, 99.5])

# minimum and maximum period of cycle N in days (99% confidence)
period_min = prange[0]*days_per_year
period_max = prange[1]*days_per_year

#  expected range of start times for cycle N+1
t1 = cycleN_tstart + datetime.timedelta(days = period_min)
t2 = cycleN_tstart + datetime.timedelta(days = period_max)

print(f"tstart_sample {type(tstart_sample)} {tstart_sample.shape}"  )
print(f"period 99 percentile (years) {prange}")
print(f"tstart range {t1} {t2}")

#------------------------------------------------------------------------------
# define a time grid for the prediction

#  Start and end times for prediction (to nearest month)
pt1 = datetime.datetime(t1.year,t1.month-1,15)
pt2 = datetime.datetime(t2.year,t2.month,15) + datetime.timedelta(days = period_max)

print(f"prediction time range {pt1} to {pt2}")

# prediction time grid in months, starting from pt1
Ntime = np.int32((pt2 - pt1).total_seconds()/seconds_per_month) + 1
ptime_months = np.arange(Ntime, dtype = np.float32)

print(f"Number of months in prediction time grid: {Ntime}")

# construct prediction time grid
ptime = []
m0 = pt1.month
for m in ptime_months:
  year = np.int32(pt1.year + (m + m0 - 1)//12)
  month = np.int32((m + m0 - 1)%12 + 1)
  ptime.append(datetime.datetime(year,month,15))

ptime = np.array(ptime)

print(f"prediction time to nearest month {ptime[0]} to {ptime[-1]}")

#------------------------------------------------------------------------------
# compute P(tstart) for each month in the prediction time grid

# index in ptime for latest likely start date
idx_tstart = np.where(ptime < t2)[0][-1]

print(f"last start date: {ptime[idx_tstart+1]}")

# start date in years since the start of Cycle N
tstart_grid = (ptime_months[:idx_tstart+2] +
              (ptime[0] - cycleN_tstart).total_seconds()/seconds_per_month)/12

pdf_tstart = np.exp(kde_tstart.score_samples(tstart_grid[:,np.newaxis]))

dt = tstart_grid[1] - tstart_grid[0]

# this should be close to the 99% confidence interval defined above
print(f"tstart integrated probability: {np.sum(pdf_tstart*dt)}")

#------------------------------------------------------------------------------
# Now compute the residuals from fits to previous cycles

resid = []

# filter out first 5 cycles for computing residuals
cstart = 5
for c in np.arange(cstart,Ncycles):
    resid.append(u.residual_known_A(time[c], ssn_sm[c], offset = offset))

print(f"Number of cycles for computing residuals: {len(resid)}")

#------------------------------------------------------------------------------
# Now allocate space for the point cloud

# frequency to sample the start time in months
dstart = 2

# number of start times to consider
Nstart = np.int32(len(pdf_tstart)/dstart) + 1

# Number of amplitudes to consider for each start time
Namp = 20

# Maximum number of points for each month
Np = Nstart * Namp * len(resid)

pointcloud = np.zeros([len(ptime), Np], dtype = np.float32)
weights = np.zeros([len(ptime), Np], dtype = np.float32)

print(f"Number of starting times: {Nstart}")
print(f"Number of amplitudes: {Namp}")
print(f"Number of residuals: {len(resid)}")
print(f"Point cloud dimensions: {pointcloud.shape}")

#------------------------------------------------------------------------------
# set up pointcloud figure

plt.rcParams.update({'font.size': 14})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig, ax = plt.subplots(1, 2, figsize = [12,5])

ax[0].set_ylabel('SSN V2')
ax[0].set_xlabel('Date')
ax[0].set_xlim(ptime[0],ptime[-1])

#------------------------------------------------------------------------------
# now loop over start times, spaced by 2 months

if dev:
  nend = 4
else:
  nend = len(pdf_tstart)

# point index
pidx = np.int64(0)

for istart in np.arange(0,nend,dstart):

  # if ptime[istart] is the start time for Cycle N+1, then this means that
  # this is the period for Cycle N (in year)
  periodN = (ptime[istart] - cycleN_tstart).total_seconds()/seconds_per_year

  print(80*'='+f"\nStart Time: {ptime[istart].month}/{ptime[istart].year} (Period N = {periodN:.2f})")

  #------------------------------------------------------------------------------
  # first get the amplitude grid and pdf (weights) for this period
  # the amplitude grid spans the 99% confidence interval based on
  # the period(N) vs amplitude(N+1) correlation

  # mu is the mean of the amplitude pdf (most likely amplitude for this period)
  smax, apdf, mu = u.amp_pdf(periodN, Namp = Namp, ssn_sm = ssn_sm, tstart = start_times)

  # sunspot max converted to amp parameter for fpanel
  amps = u.smax_to_amp(smax)

  print(f"Amplitude range: {smax.min():.2f} to {smax.max():.2f}")

  #------------------------------------------------------------------------------
  # now loop over amplitudes and accumulate points for pdf estimation

  t0 = ptime_months[istart] + offset

  for a in np.arange(len(amps)):

    amp = amps[a]

    if amp < amp_min:
      continue

    # weight for this point
    weights[:,pidx] = pdf_tstart[istart]*apdf[a]*dt

    f = u.fpanel(ptime_months,amp,t0)
    f[np.where(f < 0)] = 0.0

    for c in np.arange(len(resid)):

      # number of residual months available
      nr = len(resid[c])

      # number of available months that are in the prediction time window
      # (to ensure the arrays are the same length)
      i2 = np.min([len(ptime_months), istart + len(resid[c])])
      nrp = i2 - istart
      fr = resid[c][:nrp] + f[istart:i2]
      fr[np.where(fr < 0)] = 0
      pointcloud[istart:i2,pidx] = fr
      ax[0].scatter(ptime[istart:i2],pointcloud[istart:i2,pidx],s=1,facecolor='black')
      pidx += 1

  # plot mean profile for each t0 with points
  f = u.fpanel(ptime_months,mu,t0)
  f[np.where(f < 0)] = 0.0
  ax[0].plot(ptime,f)

print(80*'=')

#----------------------------------------------------------------------------
# Now compute the 1D KDE for each prediction month

# allocate arrays for percentiles
percentiles = [5, 25, 50, 75, 95]

prediction = np.zeros([len(ptime),len(percentiles)], dtype = np.float32)
mode = np.zeros([len(ptime)], dtype = np.float32)

points = np.zeros([Np,1])

for m in np.arange(len(ptime)):

  points[:,0] = pointcloud[m,:]
  kde = KernelDensity(kernel='gaussian', bandwidth = 10).fit(points, sample_weight = weights[m,:])

  # I think Ellery Galvin said 1000 is a typical sample size to use here 
  # for estimating confidence intervals
  sam = kde.sample(1000)
  prediction[m,:] = np.percentile(sam[:,0], percentiles)

  # Estimate mode of the KDE
  ss = np.arange(0, 300, 1)
  log_dens = kde.score_samples(ss[:, np.newaxis])
  mode[m] = ss[np.argmax(log_dens)]

  # plot selected m for debugging purposes
  if m == mtest:
    pdf = np.exp(log_dens)
    print(f"test case: {m} {ptime[m]} {np.sum(pdf)} {prediction[m,:]}")
    ax[1].hist(pointcloud[m,:], color = 'lightsteelblue', bins = 40, density = True)
    ax[1].plot(ss,pdf,'black')
    label = f"{ptime[m].month}/{ptime[m].year}"
    ax[1].annotate(label, xy=(0.7, 0.6), xycoords='axes fraction',
                fontsize=16, color='black', weight = "bold")
    ax[1].set_xlabel('SSN V2')
    ax[1].set_ylabel('pdf')

# make sure prediction is non-negative
prediction[np.where(prediction < 0)] = 0.0

#------------------------------------------------------------------------------
# finish up figure

# Add vertical line at the mean start time for Cycle N+1
ax[0].axvline(ptime[mtest], color='red', linestyle='--')

ax[1].set_xlim(0,320)

ax[0].annotate("(a)", (.4,.84), xycoords='figure fraction', weight="bold", fontsize=16, family='serif',style='italic')
ax[1].annotate("(b)", (.89,.84), xycoords='figure fraction', weight="bold", fontsize=16, family='serif',style='italic')

minor_locator = AutoMinorLocator(2)
ax[0].yaxis.set_tick_params(labelsize=12)
ax[0].xaxis.set_major_locator(mdates.YearLocator(4))
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax[0].xaxis.set_minor_locator(minor_locator)

fig.tight_layout()

plt.savefig(outdir+'/nextcycle/pointcloud.png', dpi = 300)

#------------------------------------------------------------------------------
# write results to a file for further processing and display

outfile = outdir+'/nextcycle/nextcycle_1dkde.nc'

file = netcdf_file(outfile,'w')
file.history = f"Probabilistic prediction for Cycle {this_cycle_number+1}"

# express prediction time in decimal year for netcdf output
ptimed = np.zeros(len(ptime), dtype=np.float32)

for i in np.arange(len(ptimed)):
  ptimed[i] = Time(ptime[i]).decimalyear

file.createDimension('cycle',len(per))
file.createDimension('ptime',len(ptime))
file.createDimension('stime',len(tstart_grid))
file.createDimension('pcent',len(percentiles))

outcycle = file.createVariable('Cycle','int32',('cycle',))
outcycle[:] = np.arange(len(per)) + 1
outcycle.units = 'cycle number'

outptime = file.createVariable('prediction time','float32',('ptime',))
outptime[:] = ptimed
outptime.units = 'decimal year'

outstime = file.createVariable('start time','float32',('stime',))
outstime[:] = ptimed[:idx_tstart+2]
outstime.units = 'decimal year'

outpcent = file.createVariable('percentiles','float32',('pcent',))
outpcent[:] = percentiles
outpcent.units = 'percent'

outper = file.createVariable('Cycle Periods','float32',('cycle',))
outper[:] = per
outper.units = 'years'

outNper = file.createVariable('Cycle N period','float32',('stime',))
outNper[:] = tstart_grid
outNper.units = 'decimal year'

startpdf = file.createVariable('start time pdf','float32',('stime',))
startpdf[:] = pdf_tstart
startpdf.units = 'probability density function'

outprediction = file.createVariable('probabilistic prediction','float32',('ptime','pcent'))
outprediction[:,:] = prediction
outprediction.units = 'SSN'

outmode = file.createVariable('prediction mode','float32',('ptime',))
outmode[:] = mode
outmode.units = 'SSN'

file.close()

#------------------------------------------------------------------------------

