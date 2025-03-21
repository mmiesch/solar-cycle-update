"""
compute quality of fit for each cycle curve
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import curve_fit
from scipy.stats import chisquare, chi2

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# Define files

tmon, d, dsm, t1 = u.get_cycles(tstart = True)

#------------------------------------------------------------------------------
R_squared = []
chi_prob = []
redchi = []

MAE = []
bias = []
MAE_metric = []
bias_metric = []
sigma = []

MAE_sm = []
MAE_metric_sm = []

for cycle in range(6,25):

  tobs = tmon[cycle-7]
  ssn = d[cycle-7]
  ssn_sm = dsm[cycle-7]

  afit = curve_fit(u.fuh,tobs,ssn,p0=(170.0,0.0))
  fk = u.fuh(tobs,afit[0][0],afit[0][1])

  #------------------------------------------------------------------------------
  # R squared value

  resid = ssn - fk
  ssres = np.sum(resid**2)
  sstot = np.sum((ssn - np.mean(ssn))**2)
  r2 = 1.0 - ssres/sstot
  R_squared.append(r2)

  #------------------------------------------------------------------------------
  # chi squared test

  idx = np.where(ssn > 0.0)
  y = ssn[idx]

  # The stats.chisquare function requres the observed and expected means to be 
  # the same.  There used to be an option to override this, but apparently 
  # that has been deprecated

  # The standard chi2 adjusts the fit so the means to be the same
  fexp = fk[idx] - np.mean(fk[idx]) + np.mean(y)

  # but we can do a manual calculation without the adustment
  fexp_nocheck = fk[idx]

  # this is the manual calculation of the chi2 statistic
  chitest = np.sum((y-fexp)**2/fexp)
  chitest_nocheck = np.sum((y-fexp_nocheck)**2/fexp_nocheck)

  # the degrees of freedom are the number of data points minus the number 
  # of fitting parameters
  dof = len(y) - 2

  # this is the critical value of the chi2 distribution for 75% confidence
  # ppf is the percent point function
  ccrit = chi2.ppf(0.75, df=dof)

  # This calculates the chi2 statistic and the p-value from scipy
  csq = chisquare(y, fexp)
  chi_prob.append(csq.pvalue)

  #------------------------------------------------------------------------------
  # reduced chi squared test

  # esitmate error by subracting off smoothed ssn
  err = ssn - ssn_sm
  csigma = np.mean(np.abs(err))
  rchi = np.sum(resid**2/csigma**2)/dof

  redchi.append(rchi)

  #------------------------------------------------------------------------------
  # MAE measure

  mymae = np.mean(np.abs(resid))
  mybias = np.mean(resid)

  eidx = np.where(ssn_sm > 0.0)
  err = ssn[eidx] - ssn_sm[eidx]
  ref = np.mean(np.abs(err))

  MAE.append(mymae)
  bias.append(mybias)
  MAE_metric.append(mymae/ref)
  bias_metric.append(mybias/ref)
  sigma.append(ref)

  resid_sm = ssn_sm - fk
  mymae_sm = np.mean(np.abs(resid_sm))
  MAE_sm.append(mymae_sm)
  MAE_metric_sm.append(mymae_sm/ref)

  #------------------------------------------------------------------------------
  # print the results

  # chitest should be the same as csq.statistic
  print(f'cycle {cycle} ; R^2 = {r2:.2f} rchi = {rchi:.2f} dof = {dof} sigma = {csigma:.2f}')
  print(f'chi2: {chitest_nocheck:.2f} {chitest:.2f} {csq.statistic:.2f} crit = {ccrit:.2f} p = {csq.pvalue:.2f}')

  # plot the data and the fit
  fig,ax = plt.subplots(1,2,figsize=(12,6))
  ax[0].plot(tobs, ssn, color='black', linewidth=2)
  ax[0].plot(tobs[idx], fexp, color='blue', linewidth=2)
  ax[0].plot(tobs[idx], fexp_nocheck, color='red', linewidth=2)

  # plot the chi2 distribution
  #x = np.linspace(chi2.ppf(0.01,dof),chi2.ppf(0.99,dof))
  #ax[1].plot(x, chi2.pdf(x,dof), color='black', linewidth=2)
  #ax[1].axvline(chitest, color='blue', linestyle='--')

  # residual plot
  yy = ssn[idx] - fk[idx]
  xx = fk[idx]
  #ss = ssn_sm[idx]
  ax[1].scatter(xx, yy, color='black')

  #plt.show()

#------------------------------------------------------------------------------
for i in range(0,len(R_squared)):
  print(f'cycle {i+6} R^2 = {R_squared[i]:.2f} redchi = {redchi[i]:.2f} chi2 p = {chi_prob[i]:.2f}')

print('\nMAE')

for i in range(0,len(MAE)):
  print(f'cycle {i+6} MAE = {MAE[i]:.1f} {MAE_metric[i]:.1f}, bias = {bias[i]:.1f} {bias_metric[i]:.1f}, sigma = {sigma[i]:.1f}, MAE_sm = {MAE_sm[i]:.1f} {MAE_metric_sm[i]:.1f}') 
