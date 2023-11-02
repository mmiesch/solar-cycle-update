"""
Plot some of the results from movie_maker.py to highlight evolution
"""

import csv
import datetime
import numpy as np
import sys

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# first read csv file

indir, outdir, valdir = u.get_data_dirs()

csvfile = outdir + '/prediction_evolution.csv'

amps = []
pdates = []
mdates = []

with open(csvfile, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[0] != 'prediction month':
          pdate = datetime.date(int(row[1]),int(row[0]),15)
          pdates.append(pdate)
          amps.append(int(row[2]))
          mdate = datetime.date(int(row[4]),int(row[3]),15)
          mdates.append(mdate)

amps = np.array(amps)
pdates = np.array(pdates)
mdates = np.array(mdates)

#------------------------------------------------------------------------------

