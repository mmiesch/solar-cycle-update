"""
Plot some of the results from movie_maker.py to highlight evolution
"""

import csv
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import sys

sys.path.append("../utilities")
import cycles_util as u

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
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

ax2.yaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
ax2.yaxis.set_major_locator(mdates.MonthLocator(interval=3))

#------------------------------------------------------------------------------
# save figure

outfile = outdir + '/prediction_evolution.png'

plt.savefig(outfile)

#------------------------------------------------------------------------------
plt.show()