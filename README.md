# Tull Coudé Echelle Spectrograph Reduction Pipeline
The Robert G. Tull Coudé Spectrograph Data Reduction Pipeline

## Overview
The Tull Coudé Echelle Spectrograph Reduction Pipeline is designed to process and analyze spectral data from the Tull Coudé Echelle Spectrograph at the Harlan J. Smith Telescope. This pipeline facilitates produces order extracted spectra for the following setups:

- TS21
- TS23

## Features
- **Data Preprocessing:** Automatic handling of raw data files, including calibration and noise reduction.
- **Wavelength Calibration:** Accurate mapping of pixel coordinates to wavelength.
- **Trace Determination:** Accurate mapping of pixel coordinates for each fiber as function of column.
- **Spectral Extraction:** Extraction of one-dimensional spectra from two-dimensional images.
- **Continuum Normalization:** Removal of sky background from the spectral data.
- **Spectral Stapling:** We rectify the spectra to a flexible grid and combine overlapping orders.
- **Output Formats:** We output all of the products to one multi-extension fits frame for users.

## Installation
```bash
git clone https://github.com/grzeimann/TSP.git
```
## Usage
```bash
python WHEREVER/TSP/pipeline.py FOLDER BASEDIR
```

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required libraries (listed below)

### Required Libraries
```bash
pip install numpy scipy matplotlib astropy scikit-image seaborn

### Optional but Suggested Libraries
```bash
pip install cosmic_conn
```
