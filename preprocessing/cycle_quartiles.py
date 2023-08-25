"""
This script computes residuals from cycle fits computed at different points in the cycle.  The intention is to use these residuals to provide error bars on cycle predictions based on curve fits.

An implicit assumption is that there is no missing data that is, for a given cycle, index i represents i months since the beginning of the cycle

"""

import os
import sys
import numpy as np

sys.path.append("../utilities")
import cycles_util as u

from scipy.optimize import curve_fit
from scipy.io import netcdf_file

#------------------------------------------------------------------------------
# choose fit type
# 1: 2-parameter panel fit (A, t0)
# 2: 2-parameter uh fit (A, t0)

ftype = 1

if ftype == 2:
  Np = 2
  name = "2 parameter uh fit: a, t0"
  lab = "uh"
else:
  Np = 2
  name = "2 parameter panel fit: amp, t0"
  lab = "panel2"

#------------------------------------------------------------------------------
# optionally average an earlier fit for stability
# units are months.  Set to -1 to disable

deltak = -1

if deltak > 0:
  lab = lab+f"_d{deltak}"

#------------------------------------------------------------------------------
# get cycle data

time, d, dsm = u.get_cycles()

Nc = len(d)

print(f"Number of cycles: {Nc}")

#------------------------------------------------------------------------------
# define minimum number of points needed to make a prediction.
#  Setting kmin = 11 means that the cycle update starts one year into the cycle, at month 12.  So a prediction is not made for k=0,10 (the first 11 months of the cycle).

kmin = 11

#------------------------------------------------------------------------------
# Allocate array for the result

length = []

for ssn in d:
  length.append(len(ssn))

length = np.array(length)

Nm = np.max(length)
Nk = Nm - kmin

print(f"Array size: {Nm} {Nk}")

# residual results for all cycles and lags
res = np.zeros((Nm,Nk,Nc))

#------------------------------------------------------------------------------

for i in np.arange(Nc):
  t = time[i]
  ssn = d[i]
  ssn_sm = dsm[i]

  N = len(ssn)

  print(f"Cycle {i+6} {N} {kmin}")

  for k in np.arange(kmin,N):

    if ftype == 2:
      afit = curve_fit(u.fuh,t[0:k],ssn[0:k],p0=(178.,-4.))
      f = u.fuh(t, afit[0][0], afit[0][1])
    else:
      afit = curve_fit(u.fpanel,t[0:k],ssn[0:k],p0=(178.,-4.))
      f = u.fpanel(t, afit[0][0], afit[0][1])

    if (deltak > 0) and (k > (deltak + 23)):
      k2 = k - deltak
      if ftype == 2:
        afit2 = curve_fit(u.fuh,t[0:k2],ssn[0:k2],p0=(178.,-4.))
        f2 = u.fuh(t, afit2[0][0], afit2[0][1])
      else:
        afit2 = curve_fit(u.fpanel,t[0:k2],ssn[0:k2],p0=(178.,-4.))
        f2 = u.fpanel(t, afit2[0][0], afit2[0][1])
      f = 0.5*(f+f2)

    diff = f - ssn_sm
    kidx = k - kmin

    res[:N,kidx,i] = diff

    if k == 36:
      m = 48
      print(f"check {f[m]} {ssn_sm[m]} {res[m,25,i]}")

#------------------------------------------------------------------------------
# Now compute positive and negative quartiles

pqt = np.zeros((Nm,Nk,4), dtype='float32')
nqt = np.zeros((Nm,Nk,4), dtype='float32')

for m in np.arange(Nm):
  for kidx in np.arange(Nk):

    pos = []
    neg = []
    for c in np.arange(Nc):
      if res[m,kidx,c] < 0.0:
        neg.append(np.absolute(res[m,kidx,c]))
      elif res[m,kidx,c] > 0.0:
        pos.append(np.absolute(res[m,kidx,c]))

    if len(pos) > 0:
      xp = np.array(pos)
      pqt[m,kidx,:] = np.percentile(xp, [25,50,75,100])
      #print(f"pqt {np.max(xp)} {pqt[m,kidx,3]}")

    if len(neg) > 0:
      xn = np.absolute(np.array(neg))
      nqt[m,kidx,:] = np.percentile(xn, [25,50,75,100])

    if (np.absolute(np.min(res[m,kidx,:]) + nqt[m,kidx,3]) > .01) and len(neg) > 0:
      print(f"min is outside quartile! {np.min(res[m,kidx,:])} {-nqt[m,kidx,3]}")

    if (np.absolute(np.max(res[m,kidx,:]) - pqt[m,kidx,3]) > .01) and len(pos) > 0:
      print(f"max is outside quartile! {np.max(res[m,kidx,:])} {pqt[m,kidx,3]}")

#------------------------------------------------------------------------------
# write quartiles to a file

indir, outdir, valdir = u.get_data_dirs()

os.makedirs(valdir + '/residuals', exist_ok = True)
outfile = valdir + '/residuals/quartiles_'+lab+'.nc'

file = netcdf_file(outfile,'w')
file.history = "quartiles computed from cycle fits."+name

file.createDimension('time',Nm)
file.createDimension('prediction month',Nk)
file.createDimension('quartile',4)

time = file.createVariable('time','i4',('time',))
time[:] = np.arange(Nm)
time.units = 'months'

kmon = file.createVariable('prediction month','i4',('prediction month',))
kmon[:] = np.arange(Nk) + kmin
kmon.units = 'months'

q = file.createVariable('quartile','i4',('quartile',))
q[:] = (np.arange(4)+1) * 25
q.units = 'percentile'

pres = file.createVariable('positive quartiles','float32',('time','prediction month','quartile'))
pres[:,:,:] = pqt
pres.units = 'ssn'

nres = file.createVariable('negative quartiles','float32',('time','prediction month','quartile'))
nres[:,:,:] = nqt
nres.units = 'ssn'

file.close()

print(f"Check {pres[48,25,3]}")