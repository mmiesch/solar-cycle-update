"""
The purpose of this function is to compute an updated prediction for the current solar cycle.  It is intended to be applied at least 3 years into the cycle, when there is enough SSN data to make a reasonable projection based on the average cycle progression (formula due to Hathaway et al).  The prediction is done by fitting the current data to the nonlinear function that approximates an average cycle.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from astropy.time import Time
from scipy.optimize import curve_fit
from scipy.io import netcdf_file

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# choose comparison

comp = 2

if comp == 2:
   lab1 = 'panel2'
   lab2 = 'panel2_d9'
   title = 'panel fit: no avg (blue) vs -9 month avg (red)'
elif comp == 3:
   lab1 = 'panel2_d9'
   lab2 = 'panel2_d6'
   title = 'panel fit: -9 (blue) vs -6 month (red) avg'
elif comp == 4:
   lab1 = 'panel2_d9'
   lab2 = 'panel2_d12'
   title = 'panel fit: -9 (blue) vs -12 month (red) avg'
elif comp == 5:
   lab1 = 'uh'
   lab2 = 'uh_d9'
   title = 'UH 2023 fit: no avg (blue) vs -9 month (red) avg'
elif comp == 6:
   lab1 = 'panel2'
   lab2 = 'uh'
   title = 'Panel fit (blue) vs UH 2023 fit (red): no avg'
else:
   lab1 = 'panel2_d9'
   lab2 = 'uh_d9'
   title = 'Panel fit (blue) vs UH 2023 fit (red): -9 month avg'

# choose quartile to plot (typically 1 for median or 3 for full range)
qplot = 1

#------------------------------------------------------------------------------
# label with quartile choice

if qplot == 0:
   lab3 = 'q25'
   plab3 = '25th percentile'
elif qplot == 1:
   lab3 = 'q50'
   plab3 = 'median'
elif qplot == 2:
   lab3 = 'q75'
   plab3 = '75th percentile'
else:
   lab3 = 'q100'
   plab3 = 'full range'

title = f"Residuals ({plab3}): " + title

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

#------------------------------------------------------------------------------
# plot - choose four years for illustration
#years = np.array([3.6, 5, 7, 9])
years = np.array([3, 5, 7, 9])
klist = np.int64(12*years)
Nsam = len(klist)

sns.set_theme(style={'axes.facecolor': '#FFFFFF'}, palette='colorblind')

fig, ax = plt.subplots(2,2,figsize=[12,6])
fig.suptitle(title,fontsize=16,x=0.05,y=.97,fontweight='bold', horizontalalignment = 'left')

ytime1 = rtime1 / 12.0
ytime2 = rtime2 / 12.0

for iframe in np.arange(Nsam):

  i1 = np.where(ytime1 >= years[iframe])
  i2 = np.where(ytime2 >= years[iframe])

  a = ax[p[iframe][0],p[iframe][1]]

  k = klist[iframe] - kmon1[0]
  a.plot(ytime1[i1[0]],presid1[i1[0],k,qplot],color='blue')
  a.plot(ytime1[i1[0]],-nresid1[i1[0],k,qplot],color='blue')
  a.fill_between(x=ytime1[i1[0]], y1=-nresid1[i1[0],k,qplot], y2=presid1[i1[0],k,qplot],color='blue', alpha = 0.3)

  k = klist[iframe] - kmon1[0]
  a.plot(ytime2[i2[0]],presid2[i2[0],k,qplot],color='red')
  a.plot(ytime2[i2[0]],-nresid2[i2[0],k,qplot],color='red')
  a.fill_between(x=ytime2[i2[0]], y1=-nresid2[i2[0],k,qplot], y2=presid2[i2[0],k,qplot],color='red', alpha = 0.3)

  #a.set_xlim([years[iframe] - 2, 13])
  a.set_xlim([2.5, 13])

  if iframe == 2 or iframe == 3:
     a.set_xlabel('years since cycle beginning')

  if iframe == 0 or iframe == 2:
     a.set_ylabel('SSN residual')

#------------------------------------------------------------------------------
# label

x1 = .09
y1 = .16

x2 = .57
y2 = .59

ax[0,0].annotate(f"{years[0]} years", (.39,.84), xycoords='figure fraction', weight = "bold")

ax[0,1].annotate(f"{years[1]} years", (x2,y2), xycoords='figure fraction', weight = "bold")

ax[1,0].annotate(f"{years[2]} years", (x1,y1), xycoords='figure fraction', weight = "bold")

ax[1,1].annotate(f"{years[3]} years", (x2,y1), xycoords='figure fraction', weight = "bold")


fig.tight_layout()

#------------------------------------------------------------------------------
# save to a file
dir = valdir + '/output/'

fname = f"compare_{lab1}_vs_{lab2}_{lab3}.png"

plt.savefig(dir+fname)

#------------------------------------------------------------------------------
# compute a quantitative measure of predictive skill
# start at three years in (kidx = 25 corresponds to k = 36).

Nk1 = presid1.shape[1]
kmin = kmon1[0]

median_err1 = 0.0
full_err1 = 0.0
for i in np.arange(25,Nk1):
   k = i + kmin
   median_err1 += 0.5 * np.mean(presid1[k:,i,1])
   median_err1 += 0.5 * np.mean(nresid1[k:,i,1])
   full_err1 += 0.5 * np.mean(presid1[k:,i,3])
   full_err1 += 0.5 * np.mean(nresid1[k:,i,3])

Nk2 = presid2.shape[1]
kmin = kmon2[0]

median_err2 = 0.0
full_err2 = 0.0
for i in np.arange(25,Nk2):
   k = i + kmin
   median_err2 += 0.5 * np.mean(presid2[k:,i,1])
   median_err2 += 0.5 * np.mean(nresid2[k:,i,1])
   full_err2 += 0.5 * np.mean(presid2[k:,i,3])
   full_err2 += 0.5 * np.mean(nresid2[k:,i,3])

print("Avg resid")
print(f"{os.path.basename(file1)} {median_err1/Nk1} {full_err1/Nk1}")
print(f"{os.path.basename(file2)} {median_err2/Nk2} {full_err2/Nk2}")

#------------------------------------------------------------------------------
#plt.show()