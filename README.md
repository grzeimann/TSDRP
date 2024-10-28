# The Robert G. Tull Coudé Spectrograph Data Reduction Pipeline

## Overview
The Tull Spectrograph Data Reduction Pipeline (TSDRP) is designed to process and analyze spectral data from the Tull Coudé Echelle Spectrograph at the Harlan J. Smith Telescope. This pipeline produces order extracted spectra for the following setups:

- TS23

## Key Tasks

1. **Configuration Values**:
   - The code defines several command-line argument options for configuring spectrum extraction and data processing. The folder argument specifies the folder for reduction and is required, accepting a string value. Similarly, the rootdir argument, also required and a string, designates the base directory for raw data. For customizing the extraction aperture size, the -ea or --extraction_aperture argument takes an integer with a default value of 11. To extract spectra using fiber profile weights, the -we or --weighted_extraction option can be used.  Another option, -fae or --full_aperture_extraction, returns the full aperture of the spectrum interpolated to a uniform grid rather than collapsing it into a 1D array for each order. For wavelength solution fitting, -fw or --fit_wave can be applied. Lastly, the -cr or --cosmic_rejection option provides cosmic rejection during wavelength fitting. Together, these options allow for detailed control over data processing and spectrum extraction within the specified directory and folder structure.
   - Image dimensions (`Nrows`, `Ncols`) set, bias section size, and instrument parameters such as `gain` and `readnoise`.

3. **Master Bias Creation**:
   - The build_master_bias function creates a master bias image by averaging multiple bias frames while subtracting overscan values from a overscan region of each image. The function takes in a list of file paths to bias images (bias_files), the number of rows (Nrows), columns (Ncols), and the size of the overscan section to exclude (Bias_section_size). The function loads each image, removes the bias calculated from the specified section, and retains only the relevant columns. It then computes the average bias image using a robust biweight averaging method and trims the edges. The result is a 2D array (avg_bias) representing the cleaned and averaged master bias image.
   - The master bias frame is saved as a FITS file (bias_image.fits).

4. **Master Flat Creation**:
   - The build_master_ff function generates a master flat-field image by averaging multiple flat-field frames, after subtracting a master bias and removing a designated bias section. It takes as inputs a list of flat-field file paths (ff_files), the number of rows (Nrows), columns (Ncols), the size of the bias section to exclude (Bias_section_size), and the precomputed master bias image (avg_bias). Each flat-field image is loaded, the bias is removed from the specified section, and only the relevant columns are retained. The images are then edge-trimmed, the master bias is subtracted, and the function uses robust biweight averaging to create the final master flat-field image (avg_ff).
   - The master flat frame is saved as a FITS file (ff_image.fits).

5. **Mask Frame Creation**:
   - The make_mask function generates a binary mask for an input image, marking specified columns and an optional "picket fence" region for exclusion, based on given height and bias parameters.
   - The mask frame is saved as a FITS file (mask_image.fits).

6. **Trace Measurement**:
   - The get_trace function computes the trace for each fiber in the input image by detecting fiber peaks in column chunks and aligning them based on a reference peak pattern. It uses convolution and biweight filtering to enhance and locate peaks, then refines the peak alignment by fitting a polynomial model. Finally, it outputs a high-resolution full_trace across all columns, along with the trace data per chunk and the averaged x-coordinates.
   - The trace information is saved as a FITS file (trace_image.fits).

7. **Scattered Light**:
   - The get_scattered_light function estimates and removes scattered light from an image by performing percentile filtering, smoothing, and interpolation, followed by row-wise polynomial fitting. It begins by scanning each column to extract low-level background values through percentile filtering, which are then smoothed with a Gaussian kernel. After applying a mask from task 5, the function uses interpolation and a polynomial fit to create a smooth model of scattered light across the image, producing a refined scattered light profile (scattered_light) and a raw version (S) before the polynomial fit.

8. **Flat-Field Correction**:
   - This section of the code generates a 2D flat-field correction image by estimating and removing scattered light, modeling fiber spectra, dividing that smooth 2D model, normalizing the resultant product to 1 due to mismatch between model image and the actually data, and masking regions outside the trace to 1.
   - The code starts by retrieving the scattered light background get_scattered_light().
   - Next, it creates a fiber model image (model) after subtracting the scattered light from the average flat field (avg_ff). The make_fiber_model_image function generates a model image of smoothed fiber spectra for flat-fielding purposes by extracting and fitting fiber profiles across the input image. It uses fiber trace locations, a specified extraction window, and calculated weights to assemble a "smooth" 2D model image where fiber patterns are smoothed, excluding edge fibers to avoid boundary artifacts. This output model image can then serve as a flat-field correction reference.
   - A mask is then generated to identify regions outside the trace (outside_trace_sel). The flat field is computed by dividing avg_ff minus the background by the model image and then normalized using the biweight of the selected regions. Masked and non-relevant areas are set to 1 (outside of the trace) or NaN (masked regions), and the final flat field is saved as a FITS file named ff_model.fits.

9. **Master Arc Image and Spectra Creation**:
   - Build the master arc frame from arc calibration files.
   - Extract arc spectra.
   - Save the master arc frame and arc spectra.

10. **Fit Wavelength Solution**
   - The fit_wavelength_solution function computes and saves a wavelength calibration for a given 2D array of spectra by matching observed arc lines to known reference wavelengths. It begins by normalizing the spectra using a continuum and then iteratively analyzes each spectral order (excluding the edges) to locate arc peaks, adjusting for offsets based on reference data. A polynomial fit is applied to these offsets to derive a wavelength solution for each order. The function further refines this solution by applying corrections based on observed peak locations. Finally, it saves the resulting wavelength solution along with the original spectra as a FITS file (arc_spectra.fits)
   - The is run if the option --fit_wave is set.  This task is really for generating the arc_spectra.fits for relevant setups and probably won't be run by users unless the wavelength solution starting points are just too far off.

11. **Load Wavelength Solution**:
   - Load the wavelength solution for the input setup from an archived file (arc_spectra.fits).

12. **Adjust Wavelength Solution**:
   - Since step 10 is not often run by the user, the starting wavelength solution may be off from the night's reduction.  Here we adjust the wavelenth solution for the night.
   - The get_wavelength_offset_from_archive function calculates and applies wavelength offsets to a new set of arc spectra by comparing them with archived arc spectra and their corresponding wavelength solutions. It begins by computing the continuum for both archived and new spectra, followed by a loop that calculates pixel shifts between the spectra using phase cross-correlation. A polynomial fit is then applied to the computed offsets, and these offsets are added to the archived wavelength solution. The function also interpolates the archived spectra to match the corrected wavelengths and computes wavelength offsets in chunks to refine the corrections. Finally, it visualizes the offsets and returns the corrected wavelength solution as a 2D array. The plot of the wavelenth correction is in the reduction folder.

12. **Combined Wavelength for Rectification**:
    - Compute a combined wavelength grid for rectification using logarithmic steps across the wavelength range.
    - This code computes a new wavelength array by first taking the logarithm of the input wavelength array, determining the minimum and maximum logarithmic values, calculating the median step size between the logarithmic values, and then generating a specified number of steps based on these values; finally, it exponentiates the linearly spaced logarithmic values to obtain the combined wavelength array.

13. **Deblazing and Combining Orders**:
    - Calculate the blaze function to correct for the instrument’s spectral response from the flat-field spectra.
    - Generate weights for combining spectral orders using blaze-corrected flat-field spectra.
    - The combine_spectrum function combines multiple spectra into a single spectrum through weighted averaging. It first interpolates the individual spectra and their associated errors onto a specified new wavelength grid using piecewise cubic Hermite interpolation. Then, it calculates the combined spectrum by summing the weighted interpolated spectra and propagates the uncertainties to compute the combined error, returning both the resulting spectrum and its associated error.
   
14. **Cosmic Ray Detection**
   - We use a new but robust approach for cosmic ray detection.
   - This code first computes a smoothed version of the error array using a median filter of width ~10 for each row. It then creates a binary model indicating where the error frame significantly exceeds the smoothed error frame by normalizing the difference and setting values outside a specific range to zero. The model is convolved with a 2D box kernel to expand the detection of outliers, which are cosmic rays, and updates the total mask by combining the existing mask with newly identified problematic pixels.

14. **Science Frame Reduction**:
    - The script reduces science frames by applying bias, scattered light, and flat-field corrections.
    - It then extracts spectra and corrects the trace.
    - Optionally, the script detects cosmic rays and fills masked pixels.
    - Then, it re-extracts the spectra with the modified trace, deblazes, and combines spectral orders.
    - Finally, it save the reduced spectra to a FITS file (see below for the extensions)
      
```python
    # Create a list of FITS HDUs (Header Data Units) for each data component
    L = [fits.PrimaryHDU(header=header),   # Primary HDU with header
         fits.ImageHDU(original),          # Original image
         fits.ImageHDU(image),             # Bias-subtracted image
         fits.ImageHDU(image_e),           # Error image
         fits.ImageHDU(np.array(image_m > 0., dtype=float)),  # Mask as binary
         fits.ImageHDU(spectra),           # Extracted spectra
         fits.ImageHDU(error),             # Spectra errors
         fits.ImageHDU(wave),              # Wavelength solution for each order
         fits.ImageHDU([combined_wave, combined_spectrum, 
                        combined_error])]  # Combined data   
```

## Installation
```bash
git clone https://github.com/grzeimann/TSP.git
```
## Usage
```python
usage: python reduction.py [-h] [-we] [-fae] [-fw] folder rootdir

positional arguments:
  folder                folder for reduction
  rootdir               base directory for raw data

options:
  -h, --help            show this help message and exit
  -we, --weighted_extraction
                        Extract spectra using fiber profile weights
  -fae, --full_aperture_extraction
                        Extract the full aperture for each spectrum
  -fw, --fit_wave       Fit the wavelength solution
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
pip install maskfill
```
