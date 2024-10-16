# The Robert G. Tull Coudé Spectrograph Data Reduction Pipeline

## Overview
The Tull Spectrograph Data Reduction Pipeline (TSDRP) is designed to process and analyze spectral data from the Tull Coudé Echelle Spectrograph at the Harlan J. Smith Telescope. This pipeline produces order extracted spectra for the following setups:

- TS21
- TS23

## Key Tasks

1. **Configuration Values**:
   - Image dimensions (`Nrows`, `Ncols`) set, bias section size, and instrument parameters such as `gain` and `readnoise`.

2. **Master Bias Creation**:
   - Build a master bias frame from bias calibration files.
   - Save the master bias frame as a FITS file.

3. **Master Flat Creation**:
   - Create a master flat-field frame from flat-field calibration files, applying bias corrections.
   - Save the master flat frame as a FITS file.

4. **Mask Frame Creation**:
   - Generate a mask frame to identify bad pixels and masking known bad columns.
   - Optionally, you can mask the picket effect although the location may be different than the hard coded location.
   - Save the mask frame as a FITS file.

5. **Trace Measurement**:
   - Measure the pixel trace from the flat-field frame and identify the spectral orders.
   - Save the trace information as a FITS file.

6. **Flat-Field Correction**:
   - Create a 2D flat-field correction model using the measured trace normalized by the spectra from the flat field.
   - Save the flat-field correction model as a FITS file.

7. **Master Arc Image and Spectra Creation**:
   - Build the master arc frame from arc calibration files.
   - Extract arc spectra.
   - Save the master arc frame and arc spectra.

8. **Load Wavelength Solution**:
   - Load the wavelength solution for the input setup from an archived file.
   - Adjust the wavelength solution based on extracted arc spectra.

9. **Combined Wavelength for Rectification**:
    - Compute a combined wavelength grid for rectification using logarithmic steps across the wavelength range.

10. **Deblazing and Combining Orders**:
    - Calculate the blaze function to correct for the instrument’s spectral response from the flat-field spectra.
    - Generate weights for combining spectral orders using blaze-corrected flat-field spectra.

11. **Science Frame Reduction**:
    - Reduce science frames by applying bias, scattered light, and flat-field corrections.
    - Optionally, detect cosmic rays and fill masked pixels.
    - Extract spectra, correct the trace, re-extract, deblaze, and combine spectral orders.
    - Save the reduced spectra to a FITS file.

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
```

### Optional but Suggested Libraries
```bash
pip install cosmic_conn, maskfill
```
