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
# program options

# set q = 0, 1, 2 for 25, 50, 75th percentile
q = 1

# set this to false to plot F10.7
plot_ssn = True

if plot_ssn:
  label = 'SSN'
else:
  label = 'F10'

#------------------------------------------------------------------------------
indir, outdir, valdir = u.get_data_dirs()

archive_dir = outdir + '/archive'
reanalysis_dir = valdir + '/reanalysis/archive'

start_date = datetime.date(2021,12,15)

dt = start_date - datetime.date(2019,12,15)
mstart = round(dt.days * 12 / 365)

print(f"start date: {start_date.month}/{start_date.year} mstart: {mstart}")

year = start_date.year
month = start_date.month

m_ptime = []
m_pbase = []
m_pmin = []
m_pmax = []


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

  m_ptime.append(np.array(ptime))
  m_pbase.append(np.array(pbase))
  m_pmin.append(np.array(pmin))
  m_pmax.append(np.array(pmax))

  # increment month
  month += 1
  if month > 12:
    month = 1
    year += 1

print(f"Number of prediction months: {len(m_ptime)}")

#------------------------------------------------------------------------------
# first read csv file

#amps = []
#pdates = []
#maxdates = []
#minamp = []
#maxamp = []
#drange1 = []
#drange2 = []

#with open(csvfile, 'r') as file:
#    csvreader = csv.reader(file)
#    for row in csvreader:
#        if row[0] != 'prediction month':
#          pdate = datetime.date(int(row[1]),int(row[0]),15)
#          pdates.append(pdate)
#
#          amps.append(int(row[2]))
#
#          mdate = datetime.date(int(row[4]),int(row[3]),15)
#          maxdates.append(mdate)
#
#          minamp.append(int(row[5]))
#          maxamp.append(int(row[6]))
#
#          mdate = datetime.date(int(row[8]),int(row[7]),15)
#          drange1.append(mdate)
#
#          mdate = datetime.date(int(row[10]),int(row[9]),15)
#          drange2.append(mdate)


#amps = np.array(amps)
#pdates = np.array(pdates)
#maxdates = np.array(maxdates)
#minamp = np.array(minamp)
#maxamp = np.array(maxamp)
#drange1 = np.array(drange1)
#drange2 = np.array(drange2)

#------------------------------------------------------------------------------
# loop over prediction months and get ranges for each

# note: after 10/2024 we started only writing the prediction out to the end
# of 2030 instead of 2032.  So, the length of ptime and other arrays
# decreases after that.

amps = []
pdates = []
maxdates = []
arange1 = []
arange2 = []
drange1 = []
drange2 = []

for m in np.arange(len(m_ptime)):
  pmonth = mstart + m
  print(f"ptime: {m_ptime[m][6]} {pmonth} {len(m_ptime[m])}")

  pdates.append(m_ptime[m][6])
  amps.append(np.max(m_pbase[m]))
  maxdates.append(m_ptime[m][np.argmax(m_pbase[m])])

  t1, t2, a1, a2 = u.get_date(m_ptime[m][6:], m_pbase[m][6:], m_pmin[m][6:], 
                              m_pmax[m][6:], tnow = start_date, label = "SSN")
  arange1.append(a1)
  arange2.append(a2)
  drange1.append(t1)
  drange2.append(t2)

amps = np.array(amps)
pdates = np.array(pdates)
maxdates = np.array(maxdates)
arange1 = np.array(arange1)
arange2 = np.array(arange2)
drange1 = np.array(drange1)
drange2 = np.array(drange2)


exit()
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