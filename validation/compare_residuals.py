"""
The purpose of this function is to compute an updated prediction for the current solar cycle.  It is intended to be applied at least 3 years into the cycle, when there is enough SSN data to make a reasonable projection based on the average cycle progression (formula due to Hathaway et al).  The prediction is done by fitting the current data to the nonlinear function that approximates an average cycle.
"""

import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.io import netcdf_file

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# choose comparison

comp = 6

if comp == 2:
   lab1 = 'panel2'
   lab2 = 'panel2_d9'
elif comp == 3:
   lab1 = 'panel2_d9'
   lab2 = 'panel2_d6'
elif comp == 4:
   lab1 = 'panel2_d9'
   lab2 = 'panel2_d12'
elif comp == 5:
   lab1 = 'uh'
   lab2 = 'uh_d9'
elif comp == 6:
   lab1 = 'panel2'
   lab2 = 'uh'
else:
   lab1 = 'panel2_d9'
   lab2 = 'uh_d9'

# choose quartile to plot (typically 1 for median or 3 for full range)
qplot = 1

#------------------------------------------------------------------------------

indir, outdir, valdir = u.get_data_dirs()

file1 = valdir + f'/residuals/quartiles_{lab1}.nc'
file2 = valdir + f'/residuals/quartiles_{lab2}.nc'

#---
r = netcdf_file(file1,'r')
a = r.variables['time']
rtime1 = a[:].copy()
a = r.variables['prediction month']
kmon1 = a[:].copy()
a = r.variables['quartile']
quartiles1 = a[:].copy()
a = r.variables['positive quartiles']
presid1 = a[:,:,:].copy()
a = r.variables['negative quartiles']
nresid1 = a[:,:,:].copy()
del a
r.close()

r = netcdf_file(file2,'r')
a = r.variables['time']
rtime2 = a[:].copy()
a = r.variables['prediction month']
kmon2 = a[:].copy()
a = r.variables['quartile']
quartiles2 = a[:].copy()
a = r.variables['positive quartiles']
presid2 = a[:,:,:].copy()
a = r.variables['negative quartiles']
nresid2 = a[:,:,:].copy()
del a
r.close()

#------------------------------------------------------------------------------
# plot positions

p = []
p.append((0,0))
p.append((0,1))
p.append((1,0))
p.append((1,1))
p.append((2,0))
p.append((2,1))
p.append((3,0))
p.append((3,1))
p.append((4,0))
p.append((4,1))

#------------------------------------------------------------------------------
# plot
Nsam = 10
klist = 12*np.arange(Nsam, dtype = 'int') + 11

fig, ax = plt.subplots(5,2,figsize=[12,12])

for iframe in np.arange(Nsam):

  a = ax[p[iframe][0],p[iframe][1]]

  k = klist[iframe]

  a.plot(presid1[:,k,qplot],color='blue')
  a.plot(-nresid1[:,k,qplot],color='blue')
  a.fill_between(x=rtime1, y1=presid1[:,k,qplot], y2=-nresid1[:,k,qplot],color='blue', alpha = 0.3)

  a.plot(presid2[:,k,qplot],color='red')
  a.plot(-nresid2[:,k,qplot],color='red')
  a.fill_between(x=rtime2, y1=presid2[:,k,qplot], y2=-nresid2[:,k,qplot],color='red', alpha = 0.3)

#------------------------------------------------------------------------------
# save to a file
dir = valdir + '/output/'

if qplot == 0:
   lab3 = 'q25'
elif qplot == 1:
   lab3 = 'q50'
elif qplot == 2:
   lab3 = 'q75'
else:
   lab3 = 'q100'

fname = f"compare_{lab1}_vs_{lab2}_{lab3}.png"

plt.savefig(dir+fname)

#------------------------------------------------------------------------------
plt.show()