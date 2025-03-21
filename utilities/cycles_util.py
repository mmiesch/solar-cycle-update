"""
This module defines general purpose functions that are used throughout the application
"""

import datetime
import json
import numpy as np

from astropy.time import Time
from astropy import units

#------------------------------------------------------------------------------
# define colors for output
red = '\033[91m'
yellow = '\033[93m'
cend = '\033[0m'

#------------------------------------------------------------------------------
month = {
   1:"Jan",
   2:"Feb",
   3:"Mar",
   4:"Apr",
   5:"May",
   6:"Jun",
   7:"Jul",
   8:"Aug",
   9:"Sep",
   10:"Oct",
   11:"Nov",
   12:"Dec"
}

#------------------------------------------------------------------------------
"""
This function is used to define the directories where input and output (product) files are stored for operations.

It also optionally returns the directory used to store validation data. If you are only interested in running this operationally, then you can ignore this directory by setting it to None.
"""
def get_data_dirs():

  basedir = '/home/mark.miesch/data/solar-cycle-update'

  # put monthly observations and residual files here
  indir = basedir + '/input'

  # put generated product files here
  outdir = basedir + '/output'

  # put validation data here.  If you are only interested in 
  # operations this can be set to None
  valdir = basedir + '/validation'

  return indir, outdir, valdir

#------------------------------------------------------------------------------
"""
This function identifies the input files needed for operational execution.

The obsfile, panel_ssn_prediction, and panel_f10_prediction are obtained from the public SWPC services page:

https://services.swpc.noaa.gov/json/solar-cycle/

"""
def ops_input_files():

  dirs = get_data_dirs()
  indir = dirs[0] + '/'

  # SSN and F10.7 observations (updated monthly)
  obsfile = indir + 'observed-solar-cycle-indices.json'

  # SSN prediction from 2019 panel, with uncertainties
  panel_ssn_prediction = indir + 'solar-cycle-25-ssn-predicted-range.json'

  # F10.7 prediction from 2019 panel, with uncertainties
  panel_f10_prediction = indir + 'solar-cycle-25-f10-7-predicted-range.json'

  # Residual file used to compute error bars
  # set this to None for default naming convention
  # based on fit type

  # for operational use
  residual_file = indir + 'quartiles_panel2_d9.nc'

  # for validation runs
  #residual_file = dirs[2] + '/residuals/quartiles_panel2_d9.nc'

  return obsfile, panel_ssn_prediction, panel_f10_prediction, residual_file

#------------------------------------------------------------------------------
"""
This function reads in sunspot number (ssn) data from a swpc services file and returns the series broken up into cycles.
"""


def get_cycles(full = False, tstart = False):

  #----------------------------------------------------------------------------
  # Cycle begin dates according to SIDC

  t1 = []

  t1.append(datetime.datetime(1700,1,15))
  t1.append(datetime.datetime(1755,2,15))
  t1.append(datetime.datetime(1766,6,15))
  t1.append(datetime.datetime(1775,6,15))
  t1.append(datetime.datetime(1784,9,15))
  t1.append(datetime.datetime(1798,4,15))
  t1.append(datetime.datetime(1810,7,15))
  t1.append(datetime.datetime(1823,5,15))
  t1.append(datetime.datetime(1833,11,15))
  t1.append(datetime.datetime(1843,7,15))
  t1.append(datetime.datetime(1855,12,15))
  t1.append(datetime.datetime(1867,3,15))
  t1.append(datetime.datetime(1878,12,15))
  t1.append(datetime.datetime(1890,3,15))
  t1.append(datetime.datetime(1902,1,15))
  t1.append(datetime.datetime(1913,7,15))
  t1.append(datetime.datetime(1923,8,15))
  t1.append(datetime.datetime(1933,9,15))
  t1.append(datetime.datetime(1944,2,15))
  t1.append(datetime.datetime(1954,4,15))
  t1.append(datetime.datetime(1964,10,15))
  t1.append(datetime.datetime(1976,3,15))
  t1.append(datetime.datetime(1986,9,15))
  t1.append(datetime.datetime(1996,8,15))
  t1.append(datetime.datetime(2008,12,15))
  t1.append(datetime.datetime(2019,12,15))
  t1.append(datetime.datetime(2040,12,15))

  Nc = len(t1) - 3

  #----------------------------------------------------------------------------
  # read data

  indir, outdir, valdir = get_data_dirs()

  obsfile = open(indir+'/observed-solar-cycle-indices.json')

  obsdata = json.loads(obsfile.read())

  ssn = []
  ssn_sm = []

  cycles = []
  cycles_sm = []
  tmon = []
  Nm = []

  cycle = 0
  tthis = Time(t1[cycle]).to_value('decimalyear')
  tnext = Time(t1[cycle+1]).to_value('decimalyear')
  ssn = []
  ssn_sm = []
  tm = []
  for d in obsdata:
      year, month = np.array(d['time-tag'].split('-'), dtype='int')
      t = Time(datetime.datetime(year,month,15)).to_value('decimalyear')
      if t >= tnext:
          cycles.append(np.array(ssn))
          cycles_sm.append(np.array(ssn_sm))
          tmon.append((np.array(tm) - tthis)*12)
          Nm.append(len(tm))
          cycle += 1
          print(f"cycle {cycle} {year} {month}")
          tthis = tnext
          tnext = Time(t1[cycle+1]).to_value('decimalyear')
          ssn = []
          ssn_sm = []
          tm = []
      else:
          ssn.append(d['ssn'])
          ssn_sm.append(d['smoothed_ssn'])
          tm.append(t)

  obsfile.close()

  n = Nc + 1

  if full:
    i = 1
  else:
    i = 6

  if tstart:
    return tmon[i:n], cycles[i:n], cycles_sm[i:n], t1[i:n]
  else:
    return tmon[i:n], cycles[i:n], cycles_sm[i:n]

#----------------------------------------------------------------------------
def fpanel(t,amp,t0):
  """
  I believe this is the functional form used for the panel prediction.
  Note that, relative to Hathaway's LRSP form (see below), effectivel
  A has been redefined to absorb a factor of 1/b^3
  times t and t0 should be in months since minimum
  You can define a month as a year / 12
  """
  tt = t - t0
  a =  0.000300057 + amp*(-7.12917e-6) + (amp**2)*(1.29379e-7)
  b = 15.6 + 8.18 / (a**0.25)
  c = 0.42
  return a * tt**3 / (np.exp((tt/b)**2) - c)

#----------------------------------------------------------------------------
def fuh(t,a,t0):
  """
  A new parameterization from Upton & Hathway 2023
  """
  tt = t - t0
  b = 36.3 + 0.72 / (a**0.5)
  c = 0.7
  return a * tt**3 / (np.exp((tt/b)**2) - c)

#----------------------------------------------------------------------------
def f10_from_ssn_2019(f):

  # Numbers used by Doug for panel prediction
  c0 = 6.77e1
  c1 = 3.368e-1
  c2 = 3.690e-3
  c3 = - 1.517e-5
  c4 = 1.974e-8

  return c0 + c1*f + c2*f**2 + c3*f**3 + c4*f**4

#----------------------------------------------------------------------------
def f10_from_ssn_2021(f):

  # Numbers from F. Clette (J. Space Weather Space Clim. 2021, 11, 2)
  c0 = 67.85
  c1 = 3.845e-1
  c2 = 2.881e-3
  c3 = - 7.429e-6
  c4 = 2.694e-10

  return c0 + c1*f + c2*f**2 + c3*f**3 + c4*f**4

#----------------------------------------------------------------------------
def fpanel10(t,amp,t0):
  """
  fpanel fit applied to F10.7 flux
  """

  tt = t - t0
  a =  0.000300057 + amp*(-7.12917e-6) + (amp**2)*(1.29379e-7)
  b = 15.6 + 8.18 / (a**0.25)
  c = 0.42
  s = a * tt**3 / (np.exp((tt/b)**2) - c)

  return f10_from_ssn_2019(s)

#----------------------------------------------------------------------------
def fclette10(t,amp,t0):
  """
  fpanel fit applied to F10.7 flux with Clette (2021) conversion to f10.7
  """

  tt = t - t0
  a =  0.000300057 + amp*(-7.12917e-6) + (amp**2)*(1.29379e-7)
  b = 15.6 + 8.18 / (a**0.25)
  c = 0.42
  s = a * tt**3 / (np.exp((tt/b)**2) - c)

  return f10_from_ssn_2021(s)

#------------------------------------------------------------------------------
# estimate date range of max
def get_date(t, g, gmin, gmax, tnow = None, label = None):

  # First see where the mean prediction peaks
  i = np.argmax(g)

  tmean = t[i]

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

  if tnow is None:
     tnow = datetime.date.today()

  ttmin = np.min(tt)
  #if ttmin < tnow:
  #   ttmin = tnow

  # now find amplitude range for the future
  idx = np.where(t > tnow)
  amin = int(np.max(gmin[idx]))
  amax = int(np.max(gmax[idx]))

  msg = f"{g[i]} {month[t[i].month]}/{t[i].year}"

  if label is not None:
     msg = label + ': ' + msg

  print(80*'*')
  print(yellow+"Mean prediction:"+cend)
  print(msg)
  print(80*'*')

  return [ttmin, np.max(tt), amin, amax]


#------------------------------------------------------------------------------
# determine whether or not you are in the declining phase

def declining_phase(tp, p, pmin, pmax, obstime, data, tnow = None, label = 'SSN'):
  # input parameters
  # tp = time axis for p, pmin, and pmax
  # p = mean prediction
  # pmin = lower median quartile for p
  # pmax = upper median quartile for p
  # tdata = time axis for data
  # data = smoothed observations

  if tnow is None:
     tnow = datetime.date.today()

  # these indices correspond to future time, beginning now
  pif = np.where(tp > tnow)

  # First check to see if the max observed SSN is greater than
  # the max of the positive median quartile.  If yes, then the
  # prediction is that this is the declining phase.
  if np.max(data) > np.max(pmax[pif]):
     dec = True
     tpeak = obstime[np.argmax(data)]
     print(red+f"{label} declining phase {np.max(data)} {np.max(pmax[pif])}"+cend)

  else:
     # This may or may not be the declining phase
     # So, we should still quote a range for possible max
     dec = False

     # if the observed smoothed SSN is larger than the lower median quartile
     # then there is a possibility that the max has already passed. Shift the
     # estimated range to a value earlier than today
     if np.max(data) > np.max(pmin[pif]):
        tpeak = obstime[np.argmax(data)]
        print(red+f"{label} max may have passed: {np.max(data)} {np.max(pmin[pif])}"+cend)
     else:
        # set this to a large value so it doesn't change the time range determined
        # by get_date()
        tpeak = np.max(tp)

  print(80*'*')
  print(yellow+f"Max observed {label} : {np.max(data)}"+cend)
  print(f"low  prediction {label} : {np.max(pmin[pif])}")
  print(f"mean prediction {label} : {np.max(p[pif])}")
  print(f"high prediction {label} : {np.max(pmax[pif])}")
  print(80*'*')

  return dec, tpeak

#----------------------------------------------------------------------------
def cycle_amps(ssn):
  """
  Given a list of ssn for each cycle, this returns their amplitudes
  """

  N = len(ssn)
  amps = np.zeros(N)

  for c in np.arange(N):
    amps[c] = np.max(ssn[c])

  return amps

#----------------------------------------------------------------------------
def cycle_periods(tstart):
  """
  Given the start dates for cycles, compute the periods
  """

  N = len(tstart)

  per = np.zeros(N)

  for c in np.arange(N-1):
    per[c] = (tstart[c+1] - tstart[c]).days

  # hardwire in the start date of the most recent cycle
  this_t1 = Time(2019.96,format='decimalyear').iso
  per[N-1] = (datetime.datetime.fromisoformat(this_t1) - tstart[N-1]).days

  days_per_year = units.year.to(units.second)/(3600.0*24)

  return per / days_per_year

#----------------------------------------------------------------------------
"""
Given the (smoothed) ssn and time arrays, compute the residual for the panel fit as if you know ahead of time what the amplitude and start time are.
"""
def residual_known_A(t, ssn, offset = -4.0):

  smax = np.max(ssn)
  amp = smax_to_amp(smax)

  f = fpanel(t,amp,offset)

  resid = ssn - f

  return resid

#----------------------------------------------------------------------------
"""
Return the value of amp needed to achieve a desired max sunspot number of smax
Based on a curve fit valid for amp between 60 and 300
"""
def smax_to_amp(smax):

  c = np.flip(np.array([ 2.34925573e-06, -1.99589707e-03,  1.41056910e+00, -2.50466776e+01]))

  amp = c[0]
  for n in np.arange(1,len(c)):
    amp += c[n] * np.power(smax,n)

  return amp

#----------------------------------------------------------------------------
"""
Given the period of the previous cycle, return a pdf of amplitudes and an amplitude grid spanning that pdf
Note the ampgrid that is returned corresponds to the max ssn, not the amp parameter in the fpanel function
"""
def amp_pdf(period, Namp=20, ssn_sm = None, tstart = None):

  # get cycle data
  if ssn_sm is None or tstart is None:
    time, ssn, ssn_sm, tstart = get_cycles(tstart = True)

  per = cycle_periods(tstart)
  amp = cycle_amps(ssn_sm)

  N = len(ssn_sm)
  per1 = per[:N-1]
  amp1 = amp[1:]

  #----------------------------------------------------------------------------
  # compute regression in period(N) vs amplitude (N+1) relation

  from sklearn import linear_model

  x = per1[:, np.newaxis]
  y = amp1[:, np.newaxis]

  reg = linear_model.LinearRegression()
  reg.fit(x,y)

  #----------------------------------------------------------------------------
  # compute standard deviation with respect to regression

  yfit = reg.predict(x)

  diff = (yfit[:,0] - amp1)**2
  var = np.sum(diff)/len(diff)
  sigma = np.sqrt(var)

  #----------------------------------------------------------------------------
  # assume Gaussian pdf in amplitudes

  # center on results from regression
  xnew = np.array([period])[:,np.newaxis]
  mu = reg.predict(xnew)[0][0]
  print(f"amp_pdf mean prediction {period:.2f} {mu:.2f} {sigma:.2f}")

  # span 95% confidence interval
  #amax = 1.96 * sigma

  # 99% confidence interval
  amax = 2.576 * sigma

  amin = mu - amax
  if amin < 0:
    amin = 0

  ampgrid = np.linspace(amin,mu+amax,Namp)

  ee = -0.5 * (ampgrid - mu)**2 / var
  pdf = np.exp(ee) / (sigma*np.sqrt(2*np.pi))

  return ampgrid, pdf, mu
