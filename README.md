# solar-cycle-update

This is the operational implementation of software that is used to update the solar cycle prediction that appears on [SWPC's public website](https://www.swpc.noaa.gov/products/solar-cycle-progression) 



## Instructions for Operational use

The following steps describe how to implement the operational code located in the `operations` directory.

#### Prerequisite: Define data directory and provide preprocessed data files

The base data directory is a directory on the filesystem that is used to store:

* input observational data (monthly sunspot number and F10.7)
* residual data (used to display error bars)
* generated output products (images and json)

The path to this directory must be specified in the `get_base_data_dir` function in the [cycles_util module](./utilities/cycles_util.py)

The input observational data is updated monthly as described in [Step 1](#step-1-acquire-monthly-observation-filesstep-1).

The residual data files do not change over the course of a given cycle.  So, they can be provided once and left alone.  If it is necessary to recompute them because they have been lost or corrupted, this can be done as described in the [Instructions for Preprocessing](#instructions-for-preprocessing) section.

The proper structure for the residual files relative to the base directory is as follows:

This is used to compute the error bars that are displayed in the operational product.

The base data directory is also used to store validation data but this is not needed for operational use.

#### Step 1: Acquire monthly observation files


## Instructions for Preprocessing