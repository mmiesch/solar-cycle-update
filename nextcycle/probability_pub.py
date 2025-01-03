"""
This plots P(t0) and P(A|t0) for the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.neighbors import KernelDensity

sys.path.append("../utilities")
import cycles_util as u

#------------------------------------------------------------------------------
# data directories
indir, outdir, valdir = u.get_data_dirs()

#------------------------------------------------------------------------------
# get cycle data to work with

time, ssn, ssm, tstart = u.get_cycles(full = True, tstart = True)

N = len(ssn)
print(f"Number of cycles: {N}")

per = u.cycle_periods(tstart)
amp = u.cycle_amps(ssm)

per1 = per[:N-1]
amp1 = amp[1:]

#------------------------------------------------------------------------------
# try scikit-learn kde for P(t0)

x0 = per[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth = .5).fit(x0)

x_plot = np.linspace(8,15,100)[:, np.newaxis]
pdf_t0 = np.exp(kde.score_samples(x_plot))

print(f"pdf shape {type(pdf_t0)} {pdf_t0.ndim} {pdf_t0.shape}")

dx = x_plot[1,0] - x_plot[0,0]
print(f"Normalization: {np.sum(pdf_t0)*dx}")

#------------------------------------------------------------------------------
# compute regression for P(A|t0)

from sklearn import linear_model

x = per1[:, np.newaxis]
y = amp1[:, np.newaxis]

reg = linear_model.LinearRegression()

reg.fit(x,y)

xnew = np.linspace(8,14,100)[:, np.newaxis]

ynew = reg.predict(xnew)

#------------------------------------------------------------------------------
# compute standard deviation with respect to regression

yfit = reg.predict(x)

diff = (yfit[:,0] - amp1)**2
var = np.sum(diff)/len(diff)
sigma = np.sqrt(var)

print(f"stdev: {sigma}")

#------------------------------------------------------------------------------
# define pdf with gaussians for each value of the period

nx = 100
ny = 100

xx = np.linspace(8,14,nx)
yy = np.linspace(50,300,ny)

xxfit = xx[:, np.newaxis]
yyfit = reg.predict(xxfit)

pdf = np.zeros([nx,ny])

norm = 1.0 / (sigma*np.sqrt(2*np.pi))
enorm = -0.5 / var

for i in np.arange(nx):
    ee = enorm * (yy - yyfit[i,0])**2
    pdf[i,:] = norm * np.exp(ee)

# normalize so the sum of all probabilities is 1
dx = xx[1] - xx[0]
dy = yy[1] - yy[0]
pdf = pdf / np.sum(pdf*dx*dy)

print(f"pdf range: {np.min(pdf)} {np.max(pdf)}")

#------------------------------------------------------------------------------

plt.rcParams.update({'font.size': 14})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig, ax = plt.subplots(1, 2, figsize = [12,5])

## P(t0)
ax[0].fill_between(x_plot[:, 0], y1=0, y2=pdf_t0)
ax[0].plot(x0[:,0], np.full(x0.shape[0], -0.01), "*k")
ax[0].set_xlabel('Period (yrs)')
ax[0].set_ylabel('pdf')

## P(A|t0)

xgrid, ygrid = np.meshgrid(xx,yy,indexing='ij')

ax[1].contourf(xgrid,ygrid,pdf)
ax[1].plot(xxfit[:,0],yyfit[:,0])
ax[1].scatter(per1,amp1)

ax[1].set_ylabel('Cycle n+1 S$_{max}$ (SSNV2)')
ax[1].set_xlabel('Cycle n Period (yr)')

## Annotate
ax[0].annotate("(a)", (.42,.84), xycoords='figure fraction', weight="bold", fontsize=16, family='serif',style='italic')
ax[1].annotate("(b)", (.89,.84), xycoords='figure fraction', weight="bold", fontsize=16, family='serif',style='italic',color='navajowhite')

fig.tight_layout()

fname = outdir+"/nextcycle/pt_pat.png"

#plt.savefig(fname)
plt.savefig(fname, dpi=300)

#plt.show()

