# solar-cycle-update

This is the operational implementation of software that is used to update the solar cycle prediction that appears on [SWPC's public website](https://www.swpc.noaa.gov/products/solar-cycle-progression) 

## Instructions for Operational use

The following steps describe how to implement the operational code located in the `operations` directory.

#### Prerequisite: Define data directories and provide residual data file

In order to run the code operationally, the user must provide the requisite input files, including:

1. An observations file containing monthly sunspot and F10.7 data
2. json containing the uncertainty range of the 2019 panel predictions
3. A residuals file that is used to compute the error bars on the operational products

Item 1 is obtained from the [SWPC Services web site](https://services.swpc.noaa.gov/json/solar-cycle/) and must be updated monthly as described in [Step 1](#step-1-acquire-monthly-observation-files).

Items 2 and 3 are provided once and do not change over the course of a cycle.  They do not need to be updated.

If it is necessary to recompute the residuals file (Item 3) because it has been lost or corrupted, this can be done as described in the [Instructions for Preprocessing](#instructions-for-preprocessing) section.

The user must specify the directory on the filesystem where the input files are located and a directory to place the output files (products).   This is done by editing the `get_data_dirs` function in the [cycles_util module](./utilities/cycles_util.py).

If non-standard filenames are used for Items 1 and 2, these can be specified by editing the `ops_input_files` function in the [cycles_util module](./utilities/cycles_util.py).

#### Prerequisite: Install python dependencies

The python dependencies can be obtained by installing the requirements file:

```python
pip install -r requirements.txt
```

This only has to be done once.

#### Step 1: Acquire monthly observation file

The observation file must be downloaded every month from the [SWPC Services web site](https://services.swpc.noaa.gov/json/solar-cycle/) and placed in the input data directory.  This can be done either manually or automatically (automation is not yet implemented).  The name of the file in question is `observed-solar-cycle-indices.json`.

#### Step 2: Compute the updated prediction

The updated prediction is obtained by running the `update_cycle_prediction.py` python script in the `operations` directory:

```bash
cd operations
python update_cycle_prediction.py
```

## Instructions for Preprocessing

The residuals file containing error bars as quartiles is computed by applying the prediction procedure to previous cycles.  Cycles 1-5 are generally excluded because the sparsity of observations does not allow for an accurate determination of the shape and amplitude of those cycles.

The required input file is the observations file from the SWPC services website.  This is the same file, in the same location, as Item 1 in the [operational execution](#prerequisite-define-data-directories-and-provide-residual-data-file).

If this input observations file is in place, you can compute the residual file by running the `cycle_quartiles` python script in the `preprocessing` directory:

```bash
cd preprocessing
python cycle_quartiles.py
```

The quartiles file is written to the `validation/residuals` data directory.  The appropriate file must then be copied to the input data directory.  Filenames reflect the type of fit used (`panel2` for 2-parameter fit to the function used by the 2019 panel, or `uh` for the function used by Upton & Hathaway 2023) and the (optional) averaging of different lead times (e.g. `d9` to average the current prediction with the prediction made 9 months prior).  

The default operational configuration is `quartiles_panel2_d9.nc`.  Other configurations for validation can be specified by changing the `ftype` and `deltak` parameters in the [cycle_quartiles](preprocessing/cycle_quartiles.py) script.


## Validation

The first step in the validation process is to compute residuals (quartiles) for different fit types and dual lead times as described in the [Preprocessing](#instructions-for-preprocessing) section.  These are stored in the `validation/residuals` data directory.

