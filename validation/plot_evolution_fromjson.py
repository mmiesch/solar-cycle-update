"""
Plot some of the results from movie_maker.py to highlight evolution
"""

import csv
import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import os
import sys

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# estimate date range of max
def get_date(t, g, gmin, gmax):

  # First see where the mean prediction peaks
  i = np.argmax(g)

  tmean = t[i]

  for m in np.arange(len(gmin)):
     print(f"{t[m]} {g[m]:.1f} {gmin[m]:.1f} {gmax[m]:.1f}")

  # now see where the min and max curves peak on either side 
  # of the mean

  iin = np.where(t <= tmean)
  iip = np.where(t >= tmean)

  tn = t[iin]
  tp = t[iip]

  tmin1 = tn[np.argmax(gmin[iin])]
  tmin2 = tn[np.argmax(gmax[iin])]

  tmax1 = tp[np.argmax(gmin[iip])]
  tmax2 = tp[np.argmax(gmax[iip])]

  tt = np.array([tmin1, tmax1, tmean, tmin2, tmax2])

  amin = int(np.max(gmin))
  amax = int(np.max(gmax))

  return [np.min(tt), np.max(tt), amin, amax]

#------------------------------------------------------------------------------
# program options

# set q = 0, 1, 2 for 25, 50, 75th percentile
q = 1

# set this to false to plot F10.7
plot_ssn = True

#------------------------------------------------------------------------------
indir, outdir, valdir = u.get_data_dirs()

archive_dir = outdir + '/archive'
reanalysis_dir = valdir + '/reanalysis/archive'

start_date = datetime.date(2021,12,15)

year = start_date.year
month = start_date.month



while True:

  if datetime.date(year,month,15) > datetime.date.today():
    break

  fname = f'predicted-solar-cycle_{year}_{month:02d}.json'

  fpath = archive_dir + '/' + fname
  if os.path.exists(fpath):
    with open(fpath, 'r') as json_file:
      data = json.load(json_file)
      if not 'high75_ssn' in data[0]:
        fpath = reanalysis_dir + '/' + fname

  else:
    fpath = reanalysis_dir + '/' + fname

  if not os.path.exists(fpath):
    break

  # Now read the json file

  print(fpath)
  data = json.load(open(fpath))

  ptime = []
  pbase = []
  pmin = []
  pmax = []

  for d in data:
    t = np.array(d['time-tag'].split('-'), dtype='int')
    ptime.append(datetime.date(t[0], t[1], 15))

    if plot_ssn:
      pbase.append(d['predicted_ssn'])

      if q == 0:
        pmin.append(d['low25_ssn'])
        pmax.append(d['high25_ssn'])
      elif q == 2:
        pmin.append(d['low75_ssn'])
        pmax.append(d['high75_ssn'])
      else:
        pmin.append(d['low_ssn'])
        pmax.append(d['high_ssn'])

    else:
      pbase.append(d['predicted_f10.7'])

      if q == 0:
        pmin.append(d['low25_f10.7'])
        pmax.append(d['high25_f10.7'])
      elif q == 2:
        pmin.append(d['low75_f10.7'])
        pmax.append(d['high75_f10.7'])
      else:
        pmin.append(d['low_f10.7'])
        pmax.append(d['high_f10.7'])

  # increment month
  month += 1
  if month > 12:
    month = 1
    year += 1

ptime = np.array(ptime)
pbase = np.array(pbase)
pmin = np.array(pmin)
pmax = np.array(pmax)

exit()

#------------------------------------------------------------------------------
# first read csv file

indir, outdir, valdir = u.get_data_dirs()

csvfile = outdir + '/prediction_evolution.csv'

amps = []
pdates = []
maxdates = []
minamp = []
maxamp = []
drange1 = []
drange2 = []

with open(csvfile, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[0] != 'prediction month':
          pdate = datetime.date(int(row[1]),int(row[0]),15)
          pdates.append(pdate)

          amps.append(int(row[2]))

          mdate = datetime.date(int(row[4]),int(row[3]),15)
          maxdates.append(mdate)

          minamp.append(int(row[5]))
          maxamp.append(int(row[6]))

          mdate = datetime.date(int(row[8]),int(row[7]),15)
          drange1.append(mdate)

          mdate = datetime.date(int(row[10]),int(row[9]),15)
          drange2.append(mdate)


amps = np.array(amps)
pdates = np.array(pdates)
maxdates = np.array(maxdates)
minamp = np.array(minamp)
maxamp = np.array(maxamp)
drange1 = np.array(drange1)
drange2 = np.array(drange2)

#------------------------------------------------------------------------------
# plot

fig = plt.figure(figsize = [12,6])
with sns.axes_style('whitegrid',{'grid.linestyle': ':', 'axes.facecolor': '#FFF8DC'}):
   ax1 = fig.add_subplot(211)
   ax2 = fig.add_subplot(212)

plt.rc("font", weight = 'bold')

fig.tight_layout(rect=(0.08,0.08,0.94,0.98))

sns.set_theme(palette='colorblind')

ax1.xaxis.set_tick_params(labelsize=12)
ax2.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)

sns.lineplot(x=pdates,y=amps, color='black', ax = ax1)
ax1.fill_between(x=pdates, y1 = minamp, y2 = maxamp, color='darkgray', alpha = 0.5)

sns.lineplot(x=pdates,y=maxdates, color='black', ax=ax2)
ax2.fill_between(x=pdates, y2 = drange2, y1 = drange1, color='darkgray', alpha = 0.5)

ax2.invert_yaxis()

#------------------------------------------------------------------------------
# indicate release date and most recent prediction

ax1.axhline(y = amps[-1], linestyle=':', color = 'blue')
ax2.axhline(y = maxdates[-1], linestyle=':', color = 'blue')

release_date = datetime.date(2023,9,15)
ax1.axvline(x = release_date, linestyle=':', color = 'green')
ax2.axvline(x = release_date, linestyle=':', color = 'green')

ax1.annotate(f"  {amps[-1]}", (pdates[-1],amps[-1]), ha='left', annotation_clip = False, 
             fontsize = 10, color = 'blue')

ax2.annotate(f"  {maxdates[-1].month}/{maxdates[-1].year}", (pdates[-1],maxdates[-1]), ha='left', annotation_clip = False, 
             fontsize = 10, color = 'blue')

#------------------------------------------------------------------------------
# axis limits and labels

ax1.set_xlim(np.min(pdates),np.max(pdates))
ax2.set_xlim(np.min(pdates),np.max(pdates))

ax2.set_xlabel("Prediction Date", weight='bold', fontsize=14)

ax1.set_ylabel("Amplitude (SSN)", weight = 'bold', fontsize = 12)
ax2.set_ylabel("Date of Max", weight = 'bold', fontsize = 12)

ax1.tick_params(axis='both', which='major', labelsize = 12)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

ax2.yaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax2.yaxis.set_major_locator(mdates.MonthLocator(interval=3))

# annotate

ax1.annotate("(a)", (.84,.86), xycoords='figure fraction', weight="bold", fontsize=16, family = 'serif', style = 'italic')
ax2.annotate("(b)", (.84,.43), xycoords='figure fraction', weight="bold", fontsize=16, family = 'serif', style = 'italic')

#------------------------------------------------------------------------------
# save figure

outfile = outdir + '/prediction_evolution.png'

plt.savefig(outfile)

#------------------------------------------------------------------------------
#plt.show()