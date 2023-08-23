"""
This module defines general purpose functions that are used by  
"""

import datetime
import json
import numpy as np

from astropy.time import Time

#------------------------------------------------------------------------------
"""
This function is used to define the base data directory where input, calibration, and validation data are stored, along with the generated output products.
"""
def get_base_data_dir():

  return '/home/mark.miesch/data/solar-cycle-update'

#------------------------------------------------------------------------------
"""
This function reads in sunspot number (ssn) data from a swpc services file and returns the series broken up into cycles.
"""


def get_cycles(tstart = False):

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

  dir = 'data/swpc_web_page/'
  obsfile = open(dir+'observed-solar-cycle-indices.json')

  obsdata = json.loads(obsfile.read())

  ssn = []
  ssn_sm = []

  cycles = []
  cycles_sm = []
  tmon = []
  Nm = []

  yref = t1[1].year
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

  if tstart:
    return tmon[1:n], cycles[1:n], cycles_sm[1:n], t1[1:n]
  else:
    return tmon[1:n], cycles[1:n], cycles_sm[1:n]

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
def fpanel3(t,a,t0,c):
  """
  Three-parameter version of fpanel for use with fitting full cycles
  """
  tt = t - t0
  b = 15.6 + 8.18 / (a**0.25)
  return a * tt**3 / (np.exp((tt/b)**2) - c)

#----------------------------------------------------------------------------
def fbase(t,a,b,c,t0):
  """
  This is the form suggested by Hathaway (2015, LRSP)
  t and t0 are months since minimum
  b is in months
  Hathaway quotes these as optimal parameters for the average cycle
  A = 195
  b = 56
  c = 0.8
  t0 = -4 months (prior to min)
  Hathaway et al. (1994) found that good fits to most cycles could be obtained with a fixed value for the parameter c and a parameter b that is allowed to vary with the amplitude — leaving a function of just two parameters (amplitude and starting time) that were well determine early in each cycle.
  """
  tau = (t - t0)/b
  return a * tau**3 / (np.exp(tau**2) - c)

#----------------------------------------------------------------------------
def fhath(t,a,t0):
  """
  This is a two parameter version of fhath, for predictive purposes
  based on curve fitting.
  t and t0 are in decimal year.
  Compare this with curve fits from fpanel
  """
  # placeholder - re-calibrate these as functions of A
  b = 56
  c = 0.8
  return fbase(t,a,b,c,t0)

#----------------------------------------------------------------------------
def fhath3(t,a,t0,c):
  """
  A three-parameter version of fpred, for use in fitting full cycles
  """
  # placeholder - re-calibrate these as functions of A
  b = 56
  return fbase(t,a,b,c,t0)

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

  # Numbers used by Doug
  c0 = 6.77e1
  c1 = 3.368e-1
  c2 = 3.690e-3
  c3 = - 1.517e-5
  c4 = 1.974e-8

  return c0 + c1*s + c2*s**2 + c3*s**3 + c4*s**4
