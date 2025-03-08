"""
compute quality of fit for each cycle curve
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats

from scipy.optimize import curve_fit
from scipy.stats import chisquare

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# Define files

tmon, d, dsm, t1 = u.get_cycles(tstart = True)

#------------------------------------------------------------------------------

cycle = 24

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
ccrit = stats.chi2.ppf(0.75, df=dof)

print(f'dof = {dof}')

# This calculates the chi2 statistic and the p-value from scipy
chi2 = chisquare(y, fexp)

# chitest should be the same as chi2.statistic
print(f'chi2: {chitest_nocheck:.2f} {chitest:.2f} {chi2.statistic:.2f} crit = {ccrit:.2f} p = {chi2.pvalue:.2f}')

# plot the data and the fit
plt.plot(tobs, ssn, color='black', linewidth=2)
plt.plot(tobs[idx], fexp, color='blue', linewidth=2)
plt.plot(tobs[idx], fexp_nocheck, color='red', linewidth=2)

print(f"R^2 = {r2:.2f}")

plt.show()