# solar-cycle-update

This is the operational implementation of software that is used to update the solar cycle prediction that appears on [SWPC's public website](https://www.swpc.noaa.gov/products/solar-cycle-progression) 



## Instructions for Operational use

The following steps describe how to implement the operational code located in the `operations` directory.

#### Prerequisite: Define data directories and provide residual data file

In order to run the code operationally, the user must provide the requisite input files, including:

1. An observations file containing monthly sunspot and F10.7 data: see [Step 1](#step-1-acquire-monthly-observation-files)
2. json files from the SWPC services web site containing the 2019 panel predictions and associated uncertainties
3. A residuals file that is used to compute the error bars on the operational products

Items 1 and 2 are obtained from the [SWPC Services web site](https://services.swpc.noaa.gov/json/solar-cycle/).

The prediction and residual data files (Items 2 and 3) do not change over the course of a given cycle.  So, they can be provided once and left alone.

If it is necessary to recompute the residuals file because it has been lost or corrupted, this can be done as described in the [Instructions for Preprocessing](#instructions-for-preprocessing) section.

The user must specify the directory on the filesystem where the input files are located and a directory to place the output files (products).   This is done by editing the `get_data_dirs` function in the [cycles_util module](./utilities/cycles_util.py).

If non-standard filenames are used for Items 1 and 2, these can be specified by editing the `ops_input_files` function in the [cycles_util module](./utilities/cycles_util.py).

#### Step 1: Acquire monthly observation files

## Instructions for Preprocessing