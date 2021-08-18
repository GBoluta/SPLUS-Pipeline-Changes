# -*- coding: utf-8 -*-

# ******************************************************************************
#                   S-PLUS CALIBRATION PIPELINE
#                           Get_fits
#   March, 2020
#   last edited: July 2020
#
#   by Felipe Almeida-Fernandes, felipefer42@gmail.com
#      Laura Sampedro,
#      Alberto Molino,
#      André Zamorano Vitorelli
#
# ******************************************************************************


"""
This file includes all the functions used by the different scripts of the
calibration pipeline.

--------------------------------------------------------------------------------
   FUNCTIONS:
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------
   COMMENTS:
--------------------------------------------------------------------------------
A casual user of the calibration pipeline should never have to worry about the
content of this file.

--------------------------------------------------------------------------------
   USAGE:
--------------------------------------------------------------------------------
in python:
import utils

----------------
"""

import os
import sys
from time import time
import warnings
import copy

import numpy as np
import pandas as pd

import importlib

try:
    from matplotlib import pyplot as plt
    import matplotlib.colors
    import matplotlib.gridspec as gridspec
except:
    warnings.warn("cannot import matplotlib")

from collections import OrderedDict

try:
    import scipy.stats
    from scipy.stats import gaussian_kde as kde
    from scipy.stats import linregress
    from scipy.ndimage import gaussian_filter
    from scipy import stats
except:
    warnings.warn("cannot import scipy")

try:
    from astropy.io import fits, ascii
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.table import Table
except:
    warnings.warn("cannot import astropy")

try:
    from astroquery.vizier import Vizier
except:
    warnings.warn("cannot import astroquery")

try:
    from sklearn.neighbors import KernelDensity
except:
    warnings.warn("cannot import sklearn")


# Used for photo_z estimation
try:
    import tensorflow as tf # v2.3.0
    import tensorflow_probability as tfp; tfd = tfp.distributions

    # SKLearn
    import joblib # To save/load SkLearn Scalers

    # Others
    from tqdm import tqdm # Progress bar

except:
    warnings.warn("Cannot import libraries for photo-z estimation")

#try:
#    import astroalign as aa
#except:
#    warnings.warn("cannot import astroalign")

# ******************************************************************************
#
# GENERAL
#
# ******************************************************************************

# **********************************************
#    Import config file
# **********************************************

def import_config(config_file):
    # Split the path
    config_file = config_file.split("/")

    # Get config_file_name and, if present, remove the file extension
    config_file_name = "".join(config_file[-1].split('.')[:-1])

    # Get config_file_path
    config_file_path = "/".join(config_file[:-1])

    print("Importing configuration {name} from {path}".format(name = config_file_name,
                                                               path = config_file_path))

    # Include the config path in the import paths
    sys.path.insert(1, config_file_path)

    # Import the path
    conf = importlib.import_module(config_file_name)

    return conf


# **********************************************
#    make a directory
# **********************************************
def makeroot(path):
    """
It creates a new folder in the place that I want
    """
    if not os.path.exists(path):
        cmd = ""
        cmd += '/bin/mkdir -m 777 %s' % (path)
        os.system(cmd)

        try:
            print("Created folder: %s" % path)
        except:
            pass


# *************************************************************
#   Load data files
# *************************************************************
def load_data(data_file):

    ref_data = pd.read_csv(data_file, delim_whitespace=True, escapechar='#',
                           skipinitialspace=True)
    ref_data.columns = ref_data.columns.str.replace(' ', '')

    return ref_data


# *************************************************************
#   Load fields files
# *************************************************************
def load_field_list(data_file):

    fields = []
    with open(data_file, 'r') as f:
        file_lines = f.readlines()

    for i in range(len(file_lines)):
        fields.append(file_lines[i].replace("\n", ""))

    return fields


# *************************************************************
#   Load fields files for final catalogs
# *************************************************************
def load_field_list_final_catalogs(data_file):

    read_file = pd.read_csv(data_file,
                            delim_whitespace=True,
                            header = None,
                            names = ["field", "reference"])

    fields     = list(read_file.loc[:,"field"].values)
    references = list(read_file.loc[:,"reference"].values)

    return fields, references


# *************************************************************
#   Use kde to estimate mode of a distribution
# *************************************************************
def get_mode_x(x):
    x = np.array(x)
    # Transform to kde
    y_kde = kde(x)

    x_kde = np.arange(-10, 10, 0.001)
    y = y_kde(x_kde)

    # get mode
    mode = x_kde[y == np.max(y)][0]

    return mode


# *************************************************************
#   Estimate mean robust
# *************************************************************
def mean_robust(x, low=3, high=3):
    x = np.array(x)

    mean_x = np.nanmean(x)
    std_x = np.nanstd(x)

    x = x[(x > (mean_x - low * std_x)) & (x < (mean_x + high * std_x))]

    return np.mean(x)


# *************************************************************
#   Estimate mode from discrete distribution
# *************************************************************
def discrete_mode(x):
    x = np.array(x)
    values = list(set(x))

    mode = values[0]
    len_mode = len(x[x == values[0]])

    for i in range(1, len(values)):

        if len(x[x == values[i]]) > len_mode:

            mode = values[i]
            len_mode = len(x[x == values[i]])

    return mode


# ******************************************************************************
#
# FITS HANDLING
#
# ******************************************************************************

def fz2fits(image):
    """
    It converts SPLUS images
    from .fz to .fits
    """
    datos = fits.open(image)[1].data
    heada = fits.open(image)[1].header
    imageout = image[:-2] + 'fits'
    fits.writeto(imageout, datos, heada, clobber=True)


def fits2fz(image):
    """
    It converts SPLUS images
    from .fits to .fz
    """

    cmd = 'fpack %s' % image
    os.system(cmd)


def splus_detection_image(scimas, wscimas, scimageout, wimageout):
    """
    It creates a detection image and a weighted-detection image.
--
listimas    -- list of images to be combined (root+name)
listwimas   -- list of weighted images to be combined (root+name)
scimageout  -- root+name output detection image
wimageou    -- root+name output weighted-detection image

usage:
-------
listimas = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/scimas.list'
listwimas = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/wimas.list'
imageout = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/det.swp.fits'
wimageout = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/det.weight.fits'
splus_detection_image(listimas,listwimas,imageout,wimageout)

    """

    # Defining variables
    n_scimas = len(scimas)
    n_wscimas = len(wscimas)
    if n_scimas != n_wscimas:
        print('Dimensions mismatch. Impossible to combine')
        sys.exit()
    else:
        print('Number of images to be combined: ', n_scimas)
        print('')

        # Defining the image size
    if scimas[0][:-2] == "fz":
        datos_scima = fits.open(scimas[0])[1].data
        heada_scima = fits.open(scimas[0])[1].header
        heada_wima = fits.open(wscimas[0])[1].header

    else:
        datos_scima = fits.open(scimas[0])[0].data
        heada_scima = fits.open(scimas[0])[0].header
        heada_wima = fits.open(wscimas[0])[0].header

    dimx = len(datos_scima[0, :])
    dimy = len(datos_scima[:, 0])
    matrix_scima = np.zeros((dimx, dimy), float)  # New matrix for scimageout.
    matrix_wima = np.zeros((dimx, dimy), float)  # New matrix for wimageout.

    # Combining data
    for ss in range(n_scimas):
        if scimas[ss][:-2] == "fz":
            temp_scima = fits.open(scimas[ss])[1].data
            temp_wscima = fits.open(wscimas[ss])[1].data
        else:
            temp_scima = fits.open(scimas[ss])[0].data
            temp_wscima = fits.open(wscimas[ss])[0].data

        if ss < 1:
            matrix_scima = temp_scima * (temp_wscima / np.median(temp_wscima))
            matrix_wima = temp_wscima / np.median(temp_wscima)
        else:
            matrix_scima += temp_scima * (temp_wscima / np.median(temp_wscima))
            # matrix_wima  += temp_wscima/np.median(temp_wscima)
            matrix_wima *= temp_wscima / np.median(temp_wscima)

    # Saving new image.
    print('Creating files: ')
    print(scimageout)
    fits.writeto(scimageout, matrix_scima, heada_scima, clobber=True)
    print(scimageout)
    # fits.writeto(wimageout,matrix_wima/(1.*n_scimas),heada_wima,clobber=True)
    fits.writeto(wimageout, matrix_wima / (1. * np.median(temp_wscima)), heada_wima, clobber=True)


def splus_detection_image_wo_weight(scimas, scimageout):
    """
    It creates a detection image and a weighted-detection image.
    --
    listimas    -- list of images to be combined (root+name)
    scimageout  -- root+name output detection image

    usage:
    -------
    listimas = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/scimas.list'
    listwimas = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/wimas.list'
    imageout = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/det.swp.fits'
    wimageout = '/Volumes/Madalena/SPLUS/Hydra_F0049/images/det.weight.fits'
    splus_detection_image(listimas,listwimas,imageout,wimageout)

    """

    # Defining variables
    n_scimas = len(scimas)

    print('Number of images to be combined: ', n_scimas)
    print('')

    datos_scima = fits.open(scimas[0])[0].data
    heada_scima = fits.open(scimas[0])[0].header

    dimx = len(datos_scima[0, :])
    dimy = len(datos_scima[:, 0])
    matrix_scima = np.zeros((dimx, dimy), float)  # New matrix for scimageout.

    # Combining data
    for ss in range(n_scimas):
        temp_scima = fits.open(scimas[ss])[0].data

        if ss < 1:
            matrix_scima = temp_scima
        else:
            matrix_scima += temp_scima

    # Saving new image.
    print('Creating file: %s' % scimageout)
    fits.writeto(scimageout, matrix_scima, heada_scima, clobber=True)


# *************************************************************
#   GET THE GAIN PARAMETER FROM THE IMAGES HEADER
# *************************************************************
def SPLUS_image_gain(image):
    """
    It reads the ixmage's header and
    look for the parameter
    which accounts for the gain value.
    """
    head = fits.open(image)[0].header
    # head = fits.open(image)[1].header    #para archivos .fz
    return float(head['GAIN'])


# *************************************************************
#   GET THE SEEING PARAMETER FROM THE IMAGES HEADER
# *************************************************************
def SPLUS_image_seeing(image, fmt = 'fits'):
    """
    It reads the ixmage's header and
    look for the parameter
    which accounts for the seeing value.
    """

    if fmt == 'fits':
        head = fits.open(image)[0].header

    elif fmt == 'fz':
        head = fits.open(image)[1].header    #para archivos .fz

    return float(head['HIERARCH OAJ PRO FWHMSEXT'])


# *************************************************************
#   GET THE GAIN PARAMETER FROM THE IMAGES HEADER
# *************************************************************
def SPLUS_image_satur_level(image, fmt = 'fits'):
    """
    It reads the ixmage's header and
    look for the parameter
    which accounts for the exposure time value.
    and calculates the Saturation Level
    """
    if fmt == 'fits':
        head = fits.open(image)[0].header

    elif fmt == 'fz':
        head = fits.open(image)[1].header

    else:
        raise ValueError("fmt needs to be 'fits' or 'fz'")

    #Saturation_Level = 50000.0 / head['TEXPOSED']

    #return Saturation_Level

    return float(head['SATURATE'])


# ******************************************************************************
#
# PHOTOMETRY
#
# ******************************************************************************

# *************************************************************
#   GET CONFIG FILE
# *************************************************************
def get_config_file(save_file, sex_config_model, catalog_name, field, filter,
                    mag_zp, apertures, path_to_sex, path_to_images, sex_param,
                    mode = None, use_weight = False, check_aperima = None,
                    check_segima = None):

    # Read general configuration file
    with open(sex_config_model, 'r') as f:
        sex_config = f.readlines()
        sex_config = "".join(sex_config)


    # Get image path
    if filter == 'detection':
        image = path_to_images + '/{field}_det_scimas.fits'.format(field = field)
    else:
        image = path_to_images + '/{field}/{field}_{filter}_swp.fits'.format(field=field, filter=filter)


    # Get image properties
    satur  = SPLUS_image_satur_level(image)
    seeing = SPLUS_image_seeing(image)
    gain   = SPLUS_image_gain(image)


    # Update sex_config
    sex_config = sex_config.format(catalog_name = catalog_name,
                                   param_file   = sex_param,
                                   path_to_sex  = path_to_sex,
                                   mag_zp       = mag_zp,
                                   apertures    = apertures,
                                   satur        = satur,
                                   seeing       = seeing,
                                   gain         = gain)

    # Check images. Used only for the detection image
    if filter == 'detection':
        if check_aperima is not None:
            sex_config += ("\nCHECKIMAGE_TYPE APERTURES,SEGMENTATION\n"
                           "CHECKIMAGE_NAME {aperima}, {segima}").format(aperima=check_aperima,
                                                                         segima=check_segima)

    # If using weight image, add lines
    if use_weight and (filter != 'detection'):

        wimage = path_to_images + '/{field}/{field}_{filter}_swpweight.fits'.format(field=field, filter=filter)

        if mode == 'single':
            sex_config += ("\nWEIGHT_TYPE MAP_WEIGHT\n"
                           "WEIGHT_IMAGE {wimage}").format(wimage = wimage)

        if mode == 'dual':
            det_wimage = path_to_images + '/{field}/{field}_det_wimas.fits'.format(field = field)

            sex_config += ("\nWEIGHT_TYPE MAP_WEIGHT, MAP_WEIGHT\n"
                           "WEIGHT_IMAGE {wimage}, {det_wimage}").format(wimage     = wimage,
                                                                         det_wimage = det_wimage)

    if use_weight and (filter == 'detection'):
        det_wimage = path_to_images + '/{field}/{field}_det_wimas.fits'.format(field=field)

        sex_config += ("\nWEIGHT_TYPE MAP_WEIGHT\n"
                       "WEIGHT_IMAGE {det_wimage}").format(det_wimage=det_wimage)

    # Save configuration file
    with open(save_file, 'w') as f:
        f.write(sex_config)


# *************************************************************
#   Master Sex Catalog
# *************************************************************
def master_sex_catalog(detection_catalog, detection_columns, filters,
                       individual_catalog_list, apertures, source_mag_total, save_path,
                       flux_error_cut = None, R_columns = None, R_catalog = None):
    """

    Parameters
    ----------
    detection_catalog str
        detection catalog

    detection_columns list
        list of columns to be taken from the detection catalog

    filters list
        list of filters corresponding to the individual catalogs list

    individual_catalog_list list
        list of individual catalogs to get the magnitudes from

    apertures   list
        list of aperture types to be included in the master catalog

    calib_aper_id   int
        the id of the aperture used for calibration is the one used to estimate the det S2N

    save_path str
        where to save the resulting master catalogs

    Returns
    -------
    Saves the master catalog to save_path

    """

    master_data = []

    # Load data from detection catalog
    det_cat = fits.open(detection_catalog)
    det_data = det_cat[1].data

    # Add columns from detection catalog
    for col in detection_columns:
        master_data.append(det_data.columns[col])


    if R_columns is not None:
        R_cat = fits.open(R_catalog)
        R_data = R_cat[1].data

        # Add columns from detection catalog
        for col in R_columns:
            master_data.append(R_data.columns[col])

    # ***********************************************
    # Estimating the detection image signal to noise

    if source_mag_total == 'auto':
        flux_S2N = det_data['FLUX_AUTO']
        fluxerr_S2N = det_data['FLUXERR_AUTO']

    elif isinstance(source_mag_total, int):
        flux_S2N = det_data['FLUX_APER']
        fluxerr_S2N = det_data['FLUXERR_APER']

        # If multiple apertures are estimated, use only the one used for calibration
        if len(flux_S2N.shape) > 1:
            flux_S2N = flux_S2N[:, source_mag_total]
            fluxerr_S2N = fluxerr_S2N[:, source_mag_total]

    S2N = flux_S2N / fluxerr_S2N
    S2N = np.where(S2N > 0., S2N, -1.00)

    master_data.append(fits.Column(name='s2nDET',
                                   format='1E',
                                   array=S2N))
    # ***********************************************

    # Loading and adding columns from individual catalogs
    for catalog, filter in zip(individual_catalog_list, filters):

        # Loading individual catalog
        filter_cat = fits.open(catalog)
        filter_data = filter_cat[1].data

        # Add columns:
        for aperture in apertures:

            for col in ('FLUX_%s' % aperture, 'FLUXERR_%s' % aperture,
                        'MAG_%s' % aperture, 'MAGERR_%s' % aperture):
                # Get column format from detection data
                name = col + '_%s' % filter
                fmt = det_data.columns[col].format
                data = filter_data.columns[col].array

                master_data.append(fits.Column(name=name,
                                               format=fmt,
                                               array=data))

            # ***********************************************
            # Estimating individual s2n

            fmt = det_data.columns['FLUX_%s' % aperture].format

            flux_S2N = filter_data['FLUX_%s' % aperture]
            fluxerr_S2N = filter_data['FLUXERR_%s' % aperture]

            S2N = flux_S2N / fluxerr_S2N
            S2N = np.where(S2N > 0., S2N, -1.00)

            master_data.append(fits.Column(name='MAGs2n_{aperture}_{filter}'.format(aperture=aperture, filter=filter),
                                           format=fmt,
                                           array=S2N))
            # ***********************************************

    # Generate master HDU from columns
    master_hdu = fits.BinTableHDU.from_columns(master_data)

    # Save master HDU
    master_hdu.writeto(save_path)
    print('Created file %s' % save_path)


# *************************************************************
#   Master Sex Catalog
# *************************************************************
def master_calibration_sex_catalog(master_catalog, filters, source_mag_total, save_path, frac_flux_lost = 0, aper_correction = None):
    calib_data = []

    # Load data from detection catalog
    master_cat = fits.open(master_catalog)
    master_data = master_cat[1].data

    # Load aperture correction file
    if aper_correction is not None:
        aper_correction = zp_read(aper_correction)

    # Add additional info necessary in the catalog, changing col names
    calib_cols_master = ['NUMBER', 'ALPHA_J2000', 'DELTA_J2000', 'X_IMAGE', 'Y_IMAGE']
    calib_cols_new = ['SEX_NUMBER', 'RA', 'DEC', 'X', 'Y']

    for col_master, col_new in zip(calib_cols_master, calib_cols_new):
        calib_data.append(fits.Column(name=col_new,
                                      format=master_data.columns[col_master].format,
                                      array=master_data.columns[col_master].array))

    # Add calibration photometry to the catalog
    for filter in filters:

        # Mag values
        name = "SPLUS_%s" % filter

        if source_mag_total == 'auto':
            data = master_data.columns["MAG_AUTO_%s" % filter].array

        elif source_mag_total == 'petro':
            data = master_data.columns["MAG_PETRO_%s" % filter].array

        elif isinstance(source_mag_total, int):
            data = master_data.columns["MAG_APER_%s" % filter].array

            if len(data.shape) > 1:
                data = data[:, source_mag_total]

        # Apply aperture corrections
        if aper_correction is not None:
            print("Applying {:.3f} aperture correction for filter {:s}".format(aper_correction["SPLUS_%s" % filter], filter))
            data = data + aper_correction["SPLUS_%s" % filter]

        # Apply missing flux corrections
        if frac_flux_lost != 0:
            print("Applying {:.2f}% missing flux correction for filter {:s}\n".format(frac_flux_lost*100, filter))
            flux_correction = 2.5 * np.log10(1-frac_flux_lost)
            data = data + flux_correction

        calib_data.append(fits.Column(name=name,
                                      format='1E',
                                      array=data))

        # Mag error values
        # Mag values
        name = "SPLUS_%s_err" % filter

        if source_mag_total == 'auto':
            data = master_data.columns["MAGERR_AUTO_%s" % filter].array

        elif source_mag_total == 'petro':
            data = master_data.columns["MAGERR_PETRO_%s" % filter].array

        elif isinstance(source_mag_total, int):
            data = master_data.columns["MAGERR_APER_%s" % filter].array

            if len(data.shape) > 1:
                data = data[:, source_mag_total]

        calib_data.append(fits.Column(name=name,
                                      format='1E',
                                      array=data))

    # Generate calibration HDU from columns
    master_hdu = fits.BinTableHDU.from_columns(calib_data)

    # Save master HDU
    master_hdu.writeto(save_path)
    print('Created file %s' % save_path)


# ******************************************************************************
#
# GAIA REFERENCE
#
# ******************************************************************************

# *************************************************************
#   Get catalog coordinates and fov from data
# *************************************************************
def get_catalog_coords(catalog, RA_col='RA', DEC_col='DEC'):
    cat = fits.open(catalog)
    cat_data = cat[1].data

    RA_array = cat_data.columns[RA_col].array
    DEC_array = cat_data.columns[DEC_col].array

    #RA_array[RA_array > 180] = RA_array[RA_array > 180] - 360

    RAmin, RAmax = RA_array.min(), RA_array.max()
    DECmin, DECmax = DEC_array.min(), DEC_array.max()

    if (RAmax > 270) & (RAmin < 90):
        RAmax = RAmax - 360

    RA = (RAmax + RAmin) / 2.0
    DEC = (DECmax + DECmin) / 2.0

    #RA = np.mean(RA_array)
    #DEC = np.mean(DEC_array)

    # rad = np.max((RA - RAmin,
    #              RAmax - RA,
    #              DEC - DECmin,
    #              DECmax - DEC))
    rad = 1.2

    return RA, DEC, rad


# *************************************************************
#   Download Gaia
# *************************************************************
def download_gaia(splus_catalog, save_path):
    # Get RA, DEC, rad
    RA, DEC, rad = get_catalog_coords(catalog=splus_catalog, RA_col='RA_1', DEC_col='DEC_1')

    print("Field centered around RA, DEC = {:.2f} {:.2f} deg, with radius {:.2f} deg".format(RA, DEC, rad))
    Vizier.ROW_LIMIT = -1
    #query_gaia = Vizier.query_region(coord.SkyCoord(ra=RA, dec=DEC,
    #                                                unit=(u.deg, u.deg)),
    #                                 radius=rad * u.deg,
    #                                 catalog=["gaia"])

    query = Vizier.query_region(coord.SkyCoord(ra=RA, dec=DEC,
                                               unit=(u.deg, u.deg)),
                                radius=rad * u.deg,
                                catalog=["gaia"],
                                cache = False)

    gaia_data = query['I/345/gaia2']
    table = Table(gaia_data)
    print('Saving Gaia data to %s' % save_path)
    table.write(save_path, format='fits')


# *************************************************************
#   SAVE GAIA AB photometry
# *************************************************************
def gaia_photometry_AB(gaia_catalog, save_path):
    gaia_AB_data = []

    cat = fits.open(gaia_catalog)
    cat_data = cat[1].data

    selection = ~np.isnan(cat_data.columns['Gmag'].array)
    selection = selection & ~np.isnan(cat_data.columns['BPmag'].array)
    selection = selection & ~np.isnan(cat_data.columns['RPmag'].array)
    selection = selection & (cat_data.columns['Gmag'].array < 19)

    original_cols_names = ['Source', 'RA_ICRS', 'DE_ICRS']

    new_cols_names = ['GAIA_SOURCE', 'RA', 'DEC']

    for original_col, new_col in zip(original_cols_names, new_cols_names):
        format = cat_data.columns[original_col].format
        array = cat_data.columns[original_col].array

        gaia_AB_data.append(fits.Column(name=new_col,
                                        format=format,
                                        array=array[selection]))

    # Transforming GAIA mag to AB mag

    filters = ['G', 'BP', 'RP']
    VEGA2AB_corrs = [0.138, 0.061, 0.389]

    for filter, corr in zip(filters, VEGA2AB_corrs):
        # Magnitudes
        name = 'GAIA_%s' % filter
        array = cat_data.columns['{}mag'.format(filter)].array + corr

        gaia_AB_data.append(fits.Column(name=name,
                                        format='1E',
                                        array=array[selection]))

        # Magnitude errors
        name = 'GAIA_%s_err' % filter
        array = cat_data.columns['e_{}mag'.format(filter)].array

        gaia_AB_data.append(fits.Column(name=name,
                                        format='1E',
                                        array=array[selection]))

    # Generate calibration HDU from columns
    master_hdu = fits.BinTableHDU.from_columns(gaia_AB_data)

    # Save master HDU
    master_hdu.writeto(save_path)
    print('Created file %s' % save_path)


# ******************************************************************************
#
# MODEL FITTING CALIBRATION
#
# ******************************************************************************

# *************************************************************
#   Calculate colors from magnitudes
# *************************************************************
def mags_to_colors(mag_array):
    if len(mag_array.shape) == 2:
        mag_blue = mag_array[:, :-1]
        mag_red = mag_array[:, 1:]

    else:
        mag_blue = mag_array[:-1]
        mag_red = mag_array[1:]

    color_array = mag_blue - mag_red

    return color_array


# *************************************************************
#   Calculate models chi2
# *************************************************************
def get_chi2(model_ref_mag_array, ref_mag_array, ref_magerr_array):
    """
    Explanation


    Parameters
    ----------
    name: type
        description


    Returns
    -------
    type
        description
    """

    # Get the arrays of colors from the magnitudes arrays
    model_ref_colors_array = mags_to_colors(model_ref_mag_array)
    ref_colors_array = mags_to_colors(ref_mag_array)

    # Get array of propagated squared uncertainties
    sigma_colors_2 = ref_magerr_array[1:] ** 2 + ref_magerr_array[:-1] ** 2

    # Calculate chi2
    chi2 = (model_ref_colors_array - ref_colors_array) ** 2 / sigma_colors_2

    # Sum over all colors
    chi2 = chi2.sum(axis=1)

    return chi2


# Version 2 of get_chi2 --> using shifted models instead of colors
def get_chi2_v2(model_ref_mag_array, ref_mag_array, ref_magerr_array):
    """
    Explanation


    Parameters
    ----------
    name: type
        description


    Returns
    -------
    type
        description
    """

    # Shift the models
    shift_array = ref_mag_array - model_ref_mag_array
    shift_array = shift_array.mean(axis = 1).reshape(-1,1)

    shifted_model_ref_mag_array = model_ref_mag_array + shift_array

    chi2 = (shifted_model_ref_mag_array - ref_mag_array) ** 2 / ref_magerr_array

    # Sum over all colors
    chi2 = chi2.sum(axis=1)

    return chi2


# *************************************************************
#   Absolute shift for model mags
# *************************************************************
def get_mag_shift(model_ref_mag_array, best_model_id, ref_mag_array):
    bestmod_ref_mag_array = model_ref_mag_array[best_model_id, :]
    delta_mags = ref_mag_array - bestmod_ref_mag_array

    mag_shift = np.mean(delta_mags)

    return mag_shift


# *************************************************************
#   Evaluate best model
# *************************************************************
def get_best_model(models, data, ref_mag_cols, splus_mag_cols=None):
    # Slice models array to get only reference magnitudes
    model_ref_mag_array = models.loc[:, ref_mag_cols].values

    # Slice data array to get only reference magnitudes
    ref_mag_array = data.loc[ref_mag_cols].values

    # Slice data array to get only reference magnitudes errors
    ref_magerr_cols = get_ref_magerr_cols(ref_mag_cols)
    ref_magerr_array = data.loc[ref_magerr_cols].values

    # Only used for the output
    if splus_mag_cols is not None:
        # Slice data array to get only S-PLUS magnitudes
        splus_mag_array = data.loc[splus_mag_cols].values

        # Slice data array to get only S-PLUS magnitudes errors
        splus_magerr_cols = get_ref_magerr_cols(splus_mag_cols)
        splus_magerr_array = data.loc[splus_magerr_cols].values


    # Calculate chi2 for each model
    chi2 = get_chi2_v2(model_ref_mag_array=model_ref_mag_array,
                    ref_mag_array=ref_mag_array,
                    ref_magerr_array=ref_magerr_array)

    # Get the best model id
    best_model_id = np.argmin(chi2)

    # Calculate chi2 for each model
    mag_shift = get_mag_shift(model_ref_mag_array=model_ref_mag_array,
                              best_model_id=best_model_id,
                              ref_mag_array=ref_mag_array)

    # Get best model
    best_model = models.iloc[best_model_id, :]

    # Generate the output list
    output_list = [best_model.loc['model_id']]
    output_list += [best_model.loc['EB_V']]
    output_list += [chi2[best_model_id]]
    output_list += [mag_shift]

    # Add reference magnitudes to the output
    for mag_ref, magerr_ref in zip(ref_mag_array, ref_magerr_array):
        output_list += [mag_ref]
        output_list += [magerr_ref]

    if splus_mag_cols is not None:
        # Add splus magnitudes to the output
        for mag_ref, magerr_ref in zip(splus_mag_array, splus_magerr_array):
            output_list += [mag_ref]
            output_list += [magerr_ref]

    # Add model predicted magnitudes to the output, applying the shift
    for model_mag_ref in ref_mag_cols:
        output_list += [best_model.loc[model_mag_ref] + mag_shift]

    if splus_mag_cols is not None:
        for model_mag_splus in splus_mag_cols:
            output_list += [best_model.loc[model_mag_splus] + mag_shift]

    # turn output into array
    output = np.array(output_list)

    return output


def get_best_model_v2(models, data, ref_mag_cols, splus_mag_cols=None, bayesian_flag=False):
    # Slice models array to get only reference magnitudes
    model_ref_mag_array = models.loc[:, ref_mag_cols].values

    # Slice data array to get only reference magnitudes
    ref_mag_array = data.loc[ref_mag_cols].values

    # Slice data array to get only reference magnitudes errors
    ref_magerr_cols = get_ref_magerr_cols(ref_mag_cols)
    ref_magerr_array = data.loc[ref_magerr_cols].values

    # Only used for the output
    if splus_mag_cols is not None:
        # Slice data array to get only S-PLUS magnitudes
        splus_mag_array = data.loc[splus_mag_cols].values

        # Slice data array to get only S-PLUS magnitudes errors
        splus_magerr_cols = get_ref_magerr_cols(splus_mag_cols)
        splus_magerr_array = data.loc[splus_magerr_cols].values


    # Calculate chi2 for each model
    chi2 = get_chi2_v2(model_ref_mag_array=model_ref_mag_array,
                    ref_mag_array=ref_mag_array,
                    ref_magerr_array=ref_magerr_array)

    # Get the best model id
    if bayesian_flag:
        prior = models['prior'].values
        posterior = [prior[i] * np.exp(-chi2[i]/2) for i in range(len(chi2))] 
        best_model_id = np.argmax(posterior)
    else:
        best_model_id = np.argmin(chi2)

    # Calculate chi2 for each model
    mag_shift = get_mag_shift(model_ref_mag_array=model_ref_mag_array,
                              best_model_id=best_model_id,
                              ref_mag_array=ref_mag_array)

    # Get best model
    best_model = models.iloc[best_model_id, :]

    # Generate the output list
    output_list = [best_model.loc['model_id']]
    output_list += [best_model.loc['Teff']]
    output_list += [best_model.loc['logg']]
    output_list += [best_model.loc['FeH']]
    output_list += [best_model.loc['aFe']]
    output_list += [best_model.loc['EB_V']]
    output_list += [chi2[best_model_id]]
    output_list += [mag_shift]

    # Add reference magnitudes to the output
    for mag_ref, magerr_ref in zip(ref_mag_array, ref_magerr_array):
        output_list += [mag_ref]
        output_list += [magerr_ref]

    if splus_mag_cols is not None:
        # Add splus magnitudes to the output
        for mag_ref, magerr_ref in zip(splus_mag_array, splus_magerr_array):
            output_list += [mag_ref]
            output_list += [magerr_ref]

    # Add model predicted magnitudes to the output, applying the shift
    for model_mag_ref in ref_mag_cols:
        output_list += [best_model.loc[model_mag_ref] + mag_shift]

    if splus_mag_cols is not None:
        for model_mag_splus in splus_mag_cols:
            output_list += [best_model.loc[model_mag_splus] + mag_shift]

    # turn output into array
    output = np.array(output_list)

    return output


# *************************************************************
#   Load models
# *************************************************************
def load_models(models_file):
    # must load the file with the models and put it in a pd dataframe
    models = pd.read_csv(models_file, delim_whitespace=True, escapechar='#')
    models.columns = models.columns.str.replace(' ', '')

    return models


# *************************************************************
#   Get colnames for the errors
# *************************************************************
def get_ref_magerr_cols(ref_mag_cols):
    ref_magerr_cols = []

    for col in ref_mag_cols:
        ref_magerr_cols.append(col + '_err')

    return ref_magerr_cols


# *************************************************************
#   Fit model mags and save to a file
# *************************************************************
def get_model_mags(models_file, data_file, save_file,
                   ref_mag_cols, pred_mag_cols=None):
    """
    Fit model mags to a reference catalog and predict the values of magnitudes
    for another filter system


    Parameters
    ----------
    models_file: str
        path to file with model magnitudes

    data_file: str
        path to file with reference magnitudes

    save_file: str
        path to the output file

    ref_mag_cols: tuple
        list of magnitudes names to be used to fit the models

    pred_mag_cols: tuple
        list of magnitudes names to be predicted by the models.


    Returns
    -------
    Generates output file with predicted magnitudes

    """

    t0 = time()

    # load models
    print('Loading models from file %s' % models_file)
    models = load_models(models_file)

    # load data
    print('Loading data from file %s' % data_file)
    data = load_data(data_file)

    print('\nReference magnitudes being used are:')
    print(ref_mag_cols)

    ref_magerr_cols = get_ref_magerr_cols(ref_mag_cols)

    if pred_mag_cols is not None:
        pred_magerr_cols = get_ref_magerr_cols(pred_mag_cols)
        print('\nMagnitudes being predicted are:')
        print(ref_mag_cols + pred_mag_cols)

    # Create output array

    Nlines = data.shape[0]
    Ncols = 2  # RA, DEC
    Ncols += 4  # model_id, EB_V, chi2, mag_shift
    Ncols += 3 * len(ref_mag_cols)  # mag_ref, mag_ref_err, mag_ref_model

    if pred_mag_cols is not None:
        Ncols += 3 * len(pred_mag_cols)  # splus_mag_model

    output = np.full((Nlines, Ncols), np.nan)

    print('\n\nStarting to fit best model for each star ')

    # Obtain model mags for each star in data
    for i in range(Nlines):
        sys.stdout.write('\rFinding best model for star {0} of {1}'.format(i + 1,
                                                                           Nlines))
        sys.stdout.flush()

        # Put Model data in the output array
        output[i, 2:] = get_best_model(models=models,
                                       data=data.iloc[i, :],
                                       ref_mag_cols=ref_mag_cols,
                                       splus_mag_cols=pred_mag_cols)

    print('\n\nFinished estimating best model for {0} stars'.format(Nlines))

    output[:, 0] = data.loc[:, 'RA_1'].values
    output[:, 1] = data.loc[:, 'DEC_1'].values

    # save to file
    print('\nSaving results to file %s' % save_file)

    with open(save_file, 'w') as f:
        f.write('# RA_1 DEC_1 model_id EB_V chi2 model_mag_shift')
        fmt = ['%.6f', '%.6f', '%d', '%.3f', '%.3f', '%.3f']

        for mag_name, mag_err_name in zip(ref_mag_cols, ref_magerr_cols):
            fmt += ['%.3f', '%.3f']
            f.write(' {0} {1}'.format(mag_name, mag_err_name))

        if pred_mag_cols is not None:
            for mag_name, mag_err_name in zip(pred_mag_cols, pred_magerr_cols):
                fmt += ['%.3f', '%.3f']
                f.write(' {0} {1}'.format(mag_name, mag_err_name))

        for mag_name in ref_mag_cols:
            fmt += ['%.3f']
            f.write(' %s_mod' % mag_name)

        if pred_mag_cols is not None:
            for mag_name in pred_mag_cols:
                fmt += ['%.3f']
                f.write(' %s_mod' % mag_name)

        f.write('\n')
        np.savetxt(f, output, fmt=fmt)

    print('Results are saved in file %s' % save_file)

    dt = time() - t0
    print("\nThe minchi2 model magnitudes script took "
          "{0} seconds to find the best model for {1} stars".format(dt,
                                                                    Nlines))

    return output


def get_model_mags_v2(models_file, data_file, save_file,
                      ref_mag_cols, pred_mag_cols=None,
                      bayesian_flag=False, cut=None):
    """
    Fit model mags to a reference catalog and predict the values of magnitudes
    for another filter system


    Parameters
    ----------
    models_file: str
        path to file with model magnitudes

    data_file: str
        path to file with reference magnitudes

    save_file: str
        path to the output file

    ref_mag_cols: tuple
        list of magnitudes names to be used to fit the models

    pred_mag_cols: tuple
        list of magnitudes names to be predicted by the models.


    Returns
    -------
    Generates output file with predicted magnitudes

    """

    t0 = time()

    # load models
    print('Loading models from file %s' % models_file)
    models = load_models(models_file)

    # Limit the models to the desired E(B-V)
    if cut != None:
        filt   = models['EB_V'] == cut
        models = models[filt]

    # load data
    print('Loading data from file %s' % data_file)
    data = load_data(data_file)

    print('\nReference magnitudes being used are:')
    print(ref_mag_cols)

    ref_magerr_cols = get_ref_magerr_cols(ref_mag_cols)

    if pred_mag_cols is not None:
        pred_magerr_cols = get_ref_magerr_cols(pred_mag_cols)
        print('\nMagnitudes being predicted are:')
        print(ref_mag_cols + pred_mag_cols)

    # Create output array

    Nlines = data.shape[0]
    Ncols = 2  # RA, DEC
    Ncols += 8  # model_id, Teff, logg, FeH, aFe, EB_V, chi2, mag_shift
    Ncols += 3 * len(ref_mag_cols)  # mag_ref, mag_ref_err, mag_ref_model

    if pred_mag_cols is not None:
        Ncols += 3 * len(pred_mag_cols)  # splus_mag_model

    output = np.full((Nlines, Ncols), np.nan)

    print('\n\nStarting to fit best model for each star ')

    # Obtain model mags for each star in data
    for i in range(Nlines):
        sys.stdout.write('\rFinding best model for star {0} of {1}'.format(i + 1,
                                                                           Nlines))
        sys.stdout.flush()

        # Put Model data in the output array
        output[i, 2:] = get_best_model_v2(models=models,
                                          data=data.iloc[i, :],
                                          ref_mag_cols=ref_mag_cols,
                                          splus_mag_cols=pred_mag_cols,
                                          bayesian_flag=bayesian_flag)

    print('\n\nFinished estimating best model for {0} stars'.format(Nlines))

    output[:, 0] = data.loc[:, 'RA_1'].values
    output[:, 1] = data.loc[:, 'DEC_1'].values

    # save to file
    print('\nSaving results to file %s' % save_file)

    with open(save_file, 'w') as f:
        f.write('# RA_1 DEC_1 model_id Teff logg FeH aFe EB_V chi2 model_mag_shift')
        fmt = ['%.6f', '%.6f', '%d', '%d', '%.1f', '%.2f', "%.2f", '%.3f', '%.3f', '%.3f']

        for mag_name, mag_err_name in zip(ref_mag_cols, ref_magerr_cols):
            fmt += ['%.3f', '%.3f']
            f.write(' {0} {1}'.format(mag_name, mag_err_name))

        if pred_mag_cols is not None:
            for mag_name, mag_err_name in zip(pred_mag_cols, pred_magerr_cols):
                fmt += ['%.3f', '%.3f']
                f.write(' {0} {1}'.format(mag_name, mag_err_name))

        for mag_name in ref_mag_cols:
            fmt += ['%.3f']
            f.write(' %s_mod' % mag_name)

        if pred_mag_cols is not None:
            for mag_name in pred_mag_cols:
                fmt += ['%.3f']
                f.write(' %s_mod' % mag_name)

        f.write('\n')
        np.savetxt(f, output, fmt=fmt)

    print('Results are saved in file %s' % save_file)

    dt = time() - t0
    print ("\nThe minchi2 model magnitudes script took "
           "{0} seconds to find the best model for {1} stars".format(dt,
                                                                     Nlines))

    return output


# *************************************************************
#   Get zeropoints
# *************************************************************
def get_zeropoint(splus_mag_array, model_mag_array, cut=[14,19], zp_type='mode'):
    try:
        if len(cut) == 2:
            f = (model_mag_array >= cut[0]) & (model_mag_array <= cut[1])

    except TypeError:
        f = (model_mag_array >= cut[0]) & (model_mag_array <= cut[1])

    delta_array = model_mag_array[f] - splus_mag_array[f]

    #return mean_robust(delta_array)
    return get_mode(obs_mag_array=splus_mag_array, model_mag_array=model_mag_array, cut=cut)
    # Transform to kde
    # y_kde = kde(delta_array)

    # x = np.arange(-10, 10, 0.001)
    # y = y_kde(x)

    # get mode
    # mode = x[y == np.max(y)][0]

    # return mode


def get_zeropoint_v2(obs_mag_array, model_mag_array, cut=[14,19]):

    f = (model_mag_array >= cut[0]) & (model_mag_array <= cut[1])
    delta_array = model_mag_array[f] - obs_mag_array[f]

    delta_array = delta_array.values.reshape(-1, 1)

    kde_dens = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(delta_array)

    # Transform to kde
    x = np.arange(-10, 10, 0.001)
    y = np.exp(kde_dens.score_samples(x.reshape(-1, 1)))

    # get mode
    mode = x[y == np.max(y)][0]

    return mode


# *************************************************************
#   Get mode
# *************************************************************
def get_mode(obs_mag_array, model_mag_array, cut = [14,19]):

    f = (model_mag_array >= cut[0]) & (model_mag_array <= cut[1])
    delta_array = model_mag_array[f] - obs_mag_array[f]

    delta_array = delta_array[~np.isnan(delta_array)]

    # Transform to kde
    y_kde = kde(delta_array)

    x = np.arange(-10, 10, 0.001)
    y = y_kde(x)

    # get mode
    mode = x[y == np.max(y)][0]

    return mode


# *************************************************************
#   save model fitting data
# *************************************************************
def save_data(output, save_file, ref_mag_cols, ref_magerr_cols,
              splus_mag_cols, splus_magerr_cols):
    with open(save_file, 'w') as f:
        f.write('# RA DEC model_id EB_V chi2')
        fmt = ['%.6f', '%.6f', '%d', '%.3f', '%.3f']

        for mag_name, mag_err_name in zip(ref_mag_cols, ref_magerr_cols):
            fmt += ['%.3f', '%.3f']
            f.write(' {0} {1}'.format(mag_name, mag_err_name))

        if splus_mag_cols is not None:
            for mag_name, mag_err_name in zip(splus_mag_cols, splus_magerr_cols):
                fmt += ['%.3f', '%.3f']
                f.write(' {0} {1}'.format(mag_name, mag_err_name))

        for mag_name in ref_mag_cols:
            fmt += ['%.3f']
            f.write(' %s_mod' % mag_name)

        if splus_mag_cols is not None:
            for mag_name in splus_mag_cols:
                fmt += ['%.3f']
                f.write(' %s_mod' % mag_name)

        f.write('\n')

        np.savetxt(f, output, fmt=fmt)


# ******************************************************************************
#
# ZP FILES
#
# ******************************************************************************

def zp_write(zp_dict, save_file, mag_cols=None):
    print('\nSaving results to file %s' % save_file)

    if mag_cols is None:
        mag_cols = zp_dict.keys()

    with open(save_file, 'w') as f:

        if type(mag_cols) is str:
            f.write("{:s} {:.5f}\n".format(mag_cols, zp_dict[mag_cols]))

        else:
            for mag in mag_cols:
                f.write("{:s} {:.5f}\n".format(mag, zp_dict[mag]))

    print('Results are saved in file %s' % save_file)


def zp_read(load_file, zp_col = 1):
    mags = np.genfromtxt(load_file, dtype=str, usecols=[0])
    ZPs = np.genfromtxt(load_file, dtype=float, usecols=[zp_col])

    zp_dict = {}

    try:
        for i in range(len(mags)):
            zp_dict[mags[i]] = ZPs[i]

    except TypeError:
        zp_dict[str(mags)] = float(ZPs)

    return zp_dict


def obtain_ZPs(data_file, save_file, mag_cols, mag_cut = 19, only_half_best_chi2 = False):
     data = load_data(data_file)
     print("\n\nStarting to apply ZeroPoints\n\n")

     print("Obtaining zero point for magnitudes:")
     print(mag_cols)

     print("Using {0} stars to estimate ZPs".format(data.shape[0]))

     # Estimating and applying ZP
     zp_dict = {}

     for mag_name in mag_cols:
         print("\nEstimating ZP for mag {0}".format(mag_name))

         # Cut logg
         dwarfs = data.loc[:, 'logg'].values > 3

         if only_half_best_chi2:

             chi2_half = get_half_array_value(data.loc[:, 'chi2'].values)

             condition = data.loc[:, 'chi2'].values < chi2_half

             obs_mag_array = data.loc[condition & dwarfs, mag_name]
             mod_mag_array = data.loc[condition & dwarfs, mag_name + '_mod']

         else:
             obs_mag_array = data.loc[dwarfs, mag_name]
             mod_mag_array = data.loc[dwarfs, mag_name + '_mod']


         mag_zp = get_zeropoint_v2(obs_mag_array=obs_mag_array,
                                   model_mag_array=mod_mag_array,
                                   cut = mag_cut)

         zp_dict[mag_name] = mag_zp

         data.loc[:, mag_name] = data.loc[:, mag_name] + mag_zp

         print("{} ZP = {:.3f}".format(mag_name, mag_zp))

     # Saving data
     zp_write(zp_dict = zp_dict, save_file = save_file, mag_cols = mag_cols)


def get_half_array_value(x):
    x = np.sort(x)

    half_id = int(len(x) / 2)

    return x[half_id]


def obtain_absolute_ZPs_v2(gaia_zp, save_file, splus_mag_cols):

    gaia_zps = zp_read(gaia_zp)

    absolute_zp = -np.mean(gaia_zps.values())

    splus_absolute_zps = {}

    for mag in splus_mag_cols:
        splus_absolute_zps[mag] = absolute_zp

    zp_write(splus_absolute_zps, save_file, splus_mag_cols)


def obtain_absolute_ZPs(data_file, save_file_splus, save_file_gaia, mag_cols, mag_ref,
                        mag_cut=19, only_half_best_chi2=False):
    data = load_data(data_file)
    print("\n\nStarting to apply Absolute ZeroPoints\n\n")

    print("Obtaining zero point for magnitudes:")
    print(mag_cols)

    print("Using {0} stars to estimate ZPs".format(data.shape[0]))

    # Estimating and applying ZP
    zp_dict_splus = {}
    zp_dict_gaia = {}

    if only_half_best_chi2:

        chi2_half = get_half_array_value(data.loc[:, 'chi2'].values)

        condition = data.loc[:, 'chi2'].values < chi2_half

        obs_mag_array = data.loc[condition, mag_ref]
        mod_mag_array = data.loc[condition, mag_ref + '_mod']

    else:
        obs_mag_array = data.loc[:, mag_ref]
        mod_mag_array = data.loc[:, mag_ref + '_mod']

    abs_zp = get_zeropoint_v2(obs_mag_array=obs_mag_array,
                              model_mag_array=mod_mag_array,
                              cut=mag_cut)

    # need to invert the sign
    abs_zp = -abs_zp

    print("absolute ZP = {:.3f}".format(abs_zp))
    zp_dict_gaia[mag_ref] = abs_zp

    for mag_name in mag_cols:
        zp_dict_splus[mag_name] = abs_zp

    # Saving data
    zp_write(zp_dict=zp_dict_splus, save_file=save_file_splus, mag_cols=mag_cols)
    zp_write(zp_dict=zp_dict_gaia, save_file=save_file_gaia, mag_cols=mag_ref)


def apply_ZPs(data_file, save_file, zp_file, model_file=None, mag_cols=None, sex_mag_zp = 0):
    ZPs = zp_read(zp_file)

    data = load_data(data_file)

    # Calculate delta ZPs
    if model_file is not None:
        model = load_data(model_file)

        # Include EB_V in the data
        data['EB_V'] = model['EB_V']

        for mag_name in mag_cols:
            zp_individual = model.loc[:, mag_name + '_mod'] - data.loc[:, mag_name]
            delta_zp_mag = zp_individual - ZPs[mag_name] + sex_mag_zp

            data['DZP_%s' % mag_name] = delta_zp_mag

    # Apply Zero Points
    for mag_name in ZPs.keys():
        data.loc[:, mag_name] = data.loc[:, mag_name] + ZPs[mag_name] - sex_mag_zp

    with open(data_file, 'r') as f:
        first_line = f.readline()

    if model_file is not None:
        first_line = first_line[:-1] + '   EB_V\n'

        for mag_name in mag_cols:
            first_line = first_line[:-1] + '   DZP_%s\n' % mag_name

    with open(save_file, 'w') as f:
        f.write(first_line)
        np.savetxt(f, data, fmt="%.5f")


def combine_zp_files(zp_file1, zp_file2, save_file, mags):
    zp_dict_combined = {}

    zp_dict1 = zp_read(zp_file1)

    try:
        zp_dict2 = zp_read(zp_file2)
    except:
        pass

    for mag in mags:
        try:
            zp_dict_combined[mag] = zp_dict1[mag]

        except KeyError:
            zp_dict_combined[mag] = zp_dict2[mag]

    zp_write(zp_dict=zp_dict_combined,
             save_file=save_file,
             mag_cols=mags)


def sum_zp_files(zp_file1, zp_file2, save_file, mags, mag_zp=None):
    zp_dict_sum = {}

    zp_dict1 = zp_read(zp_file1)
    zp_dict2 = zp_read(zp_file2)

    for mag in mags:
        zp_dict_sum[mag] = zp_dict1[mag] + zp_dict2[mag]

        if mag_zp is not None:
            zp_dict_sum[mag] = zp_dict_sum[mag] + mag_zp

    zp_write(zp_dict=zp_dict_sum,
             save_file=save_file,
             mag_cols=mags)


def sum_multiple_zp_files(zp_file_list, save_file, mags, mag_zp=None):
    zp_dict_sum = {}

    zp_dict_list = []

    for zp_file in zp_file_list:
        zp_dict_list.append(zp_read(zp_file))

    for mag in mags:
        zp_sum = 0
        for zp_dict in zp_dict_list:
            if mag in zp_dict.keys():
                zp_sum += zp_dict[mag]

        zp_dict_sum[mag] = zp_sum

        if mag_zp is not None:
            zp_dict_sum[mag] = zp_dict_sum[mag] + mag_zp

    zp_write(zp_dict=zp_dict_sum,
             save_file=save_file,
             mag_cols=mags)


def average_zp_files(zp_file_list, save_file, mags):

    zp_dicts = []

    mean_zp_dict = {}

    for i in range(len(zp_file_list)):
        zp_dicts.append(zp_read(zp_file_list[i]))

    for mag in mags:
        zp_mag_list = []

        for i in range(len(zp_file_list)):
            zp_mag_list.append(zp_dicts[i][mag])


        mean_zp_dict[mag] = np.nanmean(zp_mag_list)

    zp_write(zp_dict=mean_zp_dict, save_file=save_file, mag_cols=mags)


# ******************************************************************************
#
# STELLAR LOCUS CALIBRATION
#
# ******************************************************************************


def get_field_EB_V(field, EB_V_file):
    extinction = pd.read_csv(EB_V_file, delimiter=',', escapechar='#',
                             skipinitialspace=True)

    select_field_EBV = extinction['NAME'] == field

    return extinction.loc[select_field_EBV, 'sfb_ebv'].values[0]

def mode_for_discrete(x):
    """
    Returns the mode of a discrete distribution
    Parameters
    ----------
    x

    Returns
    -------

    """
    x = np.array(x)

    values = np.array(list(set(x)))
    probs = np.zeros(len(values))

    for i in range(len(values)):
        probs[i] = 1.*len(x[x == values[i]])/len(x)

    mode = values[probs == probs.max()][0]
    return mode


def apply_EBV_dependent_offset(model_file, zp_file, save_file, offset_file, mags, mag_bright_selection = "SPLUS_G_mod"):

    """
    Offset = C + alpha * EB_V
    :param zp_file:
    :param save_path:
    :param offset_file:
    :param mags:
    :return:
    """

    zp_dict_corrected = {}

    # Load zp and offset correction data
    zp_dict = zp_read(zp_file)
    alpha   = zp_read(offset_file, zp_col = 1)
    C       = zp_read(offset_file, zp_col = 2)

    # Load data to estimate the field EB_V
    data = load_data(model_file)

    # Select only the bright sources
    selection = (data[mag_bright_selection].values > 13) & (data[mag_bright_selection].values < 17)

    # Estimate the mode EB_V of the field
    EBV = mean_robust(data['EB_V'][selection])
    print("Field EB_V is {}".format(EBV))

    # Apply offset correction for each magnitude
    for mag in mags:
        zp_dict_corrected[mag] = zp_dict[mag] + (C[mag] + alpha[mag] * EBV)

    # Save offset corrected zp file
    zp_write(zp_dict = zp_dict_corrected,
             save_file = save_file,
             mag_cols = mags)



def get_stellar_locus_ZP(data_file, save_file, stellar_locus_reference_file, mag_color_ref, mag_ref, mags_to_get_zp,
                         color_range, Nbins, savefig_path):

    # Load Reference and data
    reference = load_data(stellar_locus_reference_file)
    data = load_data(data_file)

    # Select only EB-V of the field
    EB_V_field = discrete_mode(data["EB_V"])

    #select_reference = (reference["EB_V"] >= EB_V_field - 0.05) & (reference["EB_V"] <= EB_V_field + 0.05)
    #select_reference = reference["EB_V"] == EB_V_field
    #reference = reference[select_reference]

    #select_data = (data["EB_V"] >= EB_V_field - 0.05) & (data["EB_V"] <= EB_V_field + 0.05)
    #select_data = data["EB_V"] == EB_V_field
    #data = data[select_data]

    # define x axis
    reference_x = reference.loc[:, mag_color_ref[0]] - reference.loc[:, mag_color_ref[1]]
    data_x = data.loc[:, mag_color_ref[0]] - data.loc[:, mag_color_ref[1]]

    zp_dict = {}

    bins = np.linspace(color_range[0], color_range[1], Nbins)
    delta_bin = bins[1] - bins[0]

    # Obtain zero points
    for mag in mags_to_get_zp:
        print("Estimating ZP for mag %s using the stellar locus" % mag)

        delta_mag = []

        ####
        reference_bin_y_list = []
        data_bin_y_list = []
        ####

        # Remove mag = 99 or -99
        remove_bad_data = (data.loc[:, mag].values != -99) & \
                          (data.loc[:, mag].values != 99) & \
                          (data.loc[:, mag_ref].values != -99) & \
                          (data.loc[:, mag_ref].values != 99) & \
                          (data.loc[:, mag_color_ref[0]].values != -99) & \
                          (data.loc[:, mag_color_ref[0]].values != 99) & \
                          (data.loc[:, mag_color_ref[1]].values != -99) & \
                          (data.loc[:, mag_color_ref[1]].values != 99)

        for bin in bins[:-1]:

            reference_bin_cut = (reference_x >= bin) & (reference_x < bin + delta_bin)
            data_bin_cut = (data_x >= bin) & (data_x < bin + delta_bin) & remove_bad_data

            reference_bin_y = reference.loc[reference_bin_cut, mag] - reference.loc[reference_bin_cut, mag_ref]
            data_bin_y = data.loc[data_bin_cut, mag] - data.loc[data_bin_cut, mag_ref]

            mean_reference_bin_y = mean_robust(reference_bin_y)
            mean_data_bin_y = mean_robust(data_bin_y[(data_bin_y > -5) & (data_bin_y < 5)], 0.5, 0.5)

            delta_mag.append(mean_reference_bin_y - mean_data_bin_y)

            ####
            reference_bin_y_list.append(mean_reference_bin_y)
            data_bin_y_list.append(mean_data_bin_y)
            ####

        # Calculate ZP
        delta_mag = np.array(delta_mag)

        reference_bin_y_list = np.array(reference_bin_y_list)
        data_bin_y_list = np.array(data_bin_y_list)

        # Get order to remove max and min values
        o = np.argsort(delta_mag)

        zp_dict[mag] = mean_robust(delta_mag[o][1:-1])
        print("{:s} ZP = {:.3f}".format(mag, zp_dict[mag]))

        #######
        x = np.array(bins[:-1]) + delta_bin/2

        plt.scatter(reference_x, reference.loc[:, mag] - reference.loc[:, mag_ref], zorder = 1, alpha = 0.02)

        plt.scatter(x[o][1:-1], reference_bin_y_list[o][1:-1], s = 100, c = "#000066", zorder = 3)
        plt.scatter(x[o][0], reference_bin_y_list[o][0], s = 100, c = "#000000", marker = 'x', zorder = 3)
        plt.scatter(x[o][-1], reference_bin_y_list[o][-1], s = 100, c = "#000000", marker = 'x', zorder = 3)

        plt.plot(x, reference_bin_y_list, color = "#000000", zorder = 4)


        plt.scatter(data_x[remove_bad_data], data.loc[remove_bad_data, mag] - data.loc[remove_bad_data, mag_ref], c = "#FF0000", zorder=2, alpha=0.2)

        plt.scatter(x[o][1:-1], data_bin_y_list[o][1:-1], s = 100, c = "#660000", zorder = 5)
        plt.scatter(x[o][0], data_bin_y_list[o][0], s = 100, c = "#000000", marker = 'x', zorder = 5)
        plt.scatter(x[o][-1], data_bin_y_list[o][-1], s = 100, c = "#000000", marker = 'x', zorder = 5)

        plt.plot(x, data_bin_y_list, color = "#000000", zorder = 6)


        plt.gca().set_xlabel("{} - {}".format(mag_color_ref[0], mag_color_ref[1]))
        plt.gca().set_ylabel("{} - {}".format(mag, mag_ref))
        plt.gca().set_ylim((-2,4))
        plt.gca().set_xlim((0.3,1.2))
        plt.savefig(savefig_path + "/%s.png" % mag)
        plt.clf()
        plt.close()
        #######

    zp_write(zp_dict=zp_dict, save_file=save_file, mag_cols=mags_to_get_zp)


def get_stellar_locus_ZP_v2(data_file, save_file, model_fitted_file, models_file,
                            mag_color_ref, mag_ref, mags_to_get_zp, color_range, Nbins):


    # Load model fitted file to get the most fitted EB-V
    models_fitted = load_data(model_fitted_file)
    EB_V = scipy.stats.mode(models_fitted.loc[:,'EB_V'].values)[0][0]

    # Load models
    models = load_data(models_file)

    # Select only models for the right EB_V
    models = models[models.loc[:,'EB_V'].values == EB_V]

    # Load

    models_x = models.loc[:, mag_color_ref[0]] - models.loc[:, mag_color_ref[1]]

    # Load data
    data = load_data(data_file)
    data_x = data.loc[:, mag_color_ref[0]] - data.loc[:, mag_color_ref[1]]

    zp_dict = {}

    bins = np.linspace(color_range[0], color_range[1], Nbins)
    delta_bin = bins[1] - bins[0]

    # Obtain zero points
    for mag in mags_to_get_zp:
        print("Estimating ZP for mag %s using the stellar locus" % mag)

        delta_mag = []

        ####
        models_bin_y_deleteme = []
        data_bin_y_deleteme = []
        ####

        # Remove mag = 99 or -99
        remove_bad_data = (data.loc[:, mag].values != -99) & (data.loc[:, mag].values != 99)

        for bin in bins[:-1]:

            models_bin_cut = (models_x >= bin) & (models_x < bin + delta_bin)
            data_bin_cut = (data_x >= bin) & (data_x < bin + delta_bin) & remove_bad_data

            models_bin_y = models.loc[models_bin_cut, mag] - models.loc[models_bin_cut, mag_ref]
            data_bin_y = data.loc[data_bin_cut, mag] - data.loc[data_bin_cut, mag_ref]

            mean_models_bin_y = mean_robust(models_bin_y)
            mean_data_bin_y = mean_robust(data_bin_y, 0.5, 0.5)

            delta_mag.append(mean_models_bin_y - mean_data_bin_y)

            ####
            models_bin_y_deleteme.append(mean_models_bin_y)
            data_bin_y_deleteme.append(mean_data_bin_y)
            ####

        # Calculate ZP
        zp_dict[mag] = mean_robust(delta_mag)
        print("{:s} ZP = {:.3f}".format(mag, zp_dict[mag]))

        #######
        x = np.array(bins[:-1]) + delta_bin/2

        plt.scatter(models_x, models.loc[:, mag] - models.loc[:, mag_ref], zorder = -2, alpha = 0.2)
        plt.scatter(x, models_bin_y_deleteme, s = 100, c = "#000066")
        plt.plot(x, models_bin_y_deleteme, color = "#000000")

        plt.scatter(data_x, data.loc[:, mag] - data.loc[:, mag_ref], c = "#FF0000", zorder=-1, alpha=0.2)
        plt.scatter(x, data_bin_y_deleteme, s = 100, c = "#660000")
        plt.plot(x, data_bin_y_deleteme, color = "#000000")

        plt.show()
        #######

    zp_write(zp_dict=zp_dict, save_file=save_file, mag_cols=mags_to_get_zp)



# ******************************************************************************
#
# FIT FINAL OFFSETS
#
# ******************************************************************************

def get_broad_band_offset_from_models(data_file, model_file, save_file,
                                      color_range = [-0.3, 0.3], Nbins = 4,
                                      mag_color_ref = ['SPLUS_G', 'SPLUS_I'],
                                      savefig_path = ""):

    filters = ['SPLUS_G', 'SPLUS_R', 'SPLUS_I', 'SPLUS_Z']

    # Load data file
    data = load_data(data_file)
    mag_cut = data.loc[:,"SPLUS_R"] <= 17.5
    mag_cut = mag_cut & (data.loc[:,"SPLUS_R"] >= 13)

    mag_cut = mag_cut & (data.loc[:,"SPLUS_G"] <= 17.5)
    mag_cut = mag_cut & (data.loc[:,"SPLUS_G"] >= 13)

    mag_cut = mag_cut & (data.loc[:,"SPLUS_I"] <= 17.5)
    mag_cut = mag_cut & (data.loc[:,"SPLUS_I"] >= 13)

    mag_cut = mag_cut & (data.loc[:,"SPLUS_Z"] <= 17.5)
    mag_cut = mag_cut & (data.loc[:,"SPLUS_Z"] >= 13)

    data = data[mag_cut]

    data_x = data.loc[:, mag_color_ref[0]] - data.loc[:, mag_color_ref[1]]

    # Load models file
    reference = load_data(model_file)
    reference_x = reference.loc[:, mag_color_ref[0]] - reference.loc[:, mag_color_ref[1]]

    # Coefficient matrix
    a = np.array([[1, -1, 0, 0],
                  [1, 0, -1, 0],
                  [1, 0, 0, -1],
                  [0, 1, -1, 0],
                  [0, 1, 0, -1],
                  [0, 0, 1, -1]])

    # Colors
    a_colors = [['SPLUS_G', 'SPLUS_R'],
                ['SPLUS_G', 'SPLUS_I'],
                ['SPLUS_G', 'SPLUS_Z'],
                ['SPLUS_R', 'SPLUS_I'],
                ['SPLUS_R', 'SPLUS_Z'],
                ['SPLUS_I', 'SPLUS_Z']]

    b = []

    # Prepare bins
    bins = np.linspace(color_range[0], color_range[1], Nbins)
    delta_bin = bins[1] - bins[0]

    for y_color in a_colors:

        ####
        reference_bin_y_plot = []
        data_bin_y_plot = []
        ####

        # Remove mag = 99 or -99
        remove_bad_data = (data.loc[:, y_color[0]].values != -99) & \
                          (data.loc[:, y_color[0]].values != 99) & \
                          (data.loc[:, y_color[1]].values != -99) & \
                          (data.loc[:, y_color[1]].values != 99) & \
                          (data.loc[:, mag_color_ref[0]].values != -99) & \
                          (data.loc[:, mag_color_ref[0]].values != 99) & \
                          (data.loc[:, mag_color_ref[1]].values != -99) & \
                          (data.loc[:, mag_color_ref[1]].values != 99)

        delta_mag = []

        for bin in bins[:-1]:

            reference_bin_cut = (reference_x >= bin) & (reference_x < bin + delta_bin)
            data_bin_cut = (data_x >= bin) & (data_x < bin + delta_bin)

            reference_bin_y = reference.loc[reference_bin_cut, y_color[0]] - reference.loc[reference_bin_cut, y_color[1]]
            data_bin_y = data.loc[data_bin_cut, y_color[0]] - data.loc[data_bin_cut, y_color[1]]

            mean_reference_bin_y = mean_robust(reference_bin_y)
            mean_data_bin_y = mean_robust(data_bin_y)

            delta_mag.append(mean_reference_bin_y - mean_data_bin_y)

            ####
            reference_bin_y_plot.append(mean_reference_bin_y)
            data_bin_y_plot.append(mean_data_bin_y)
            ####


        # Calculate ZP
        delta_mag = np.array(delta_mag)
        reference_bin_y_plot = np.array(reference_bin_y_plot)
        data_bin_y_plot = np.array(data_bin_y_plot)

        o = np.argsort(delta_mag)

        b.append(mean_robust(delta_mag[o][1:-1]))

        #######
        x = np.array(bins[:-1]) + delta_bin/2

        plt.scatter(reference_x, reference.loc[:, y_color[0]] - reference.loc[:, y_color[1]], zorder = 2, alpha = 0.05)
        plt.scatter(x[o][1:-1], reference_bin_y_plot[o][1:-1], s = 100, c = "#000066", zorder = 3)
        plt.scatter(x[o][0], reference_bin_y_plot[o][0], s = 100, c = "#000000", zorder = 3, marker = 'x')
        plt.scatter(x[o][-1], reference_bin_y_plot[o][-1], s = 100, c = "#000000", zorder = 3, marker = 'x')
        plt.plot(x, reference_bin_y_plot, color = "#000000", zorder = 4)

        plt.scatter(data_x[remove_bad_data], data.loc[remove_bad_data, y_color[0]] - data.loc[remove_bad_data, y_color[1]], c = "#FF0000", zorder=1, alpha=0.2)
        plt.scatter(x[o][1:-1], data_bin_y_plot[o][1:-1], s = 100, c = "#660000", zorder = 5)
        plt.scatter(x[o][0], data_bin_y_plot[o][0], s = 100, c = "#000000", zorder = 5, marker = 'x')
        plt.scatter(x[o][-1], data_bin_y_plot[o][-1], s = 100, c = "#000000", zorder = 5, marker = 'x')
        plt.plot(x, data_bin_y_plot, color = "#000000", zorder = 6)

        plt.gca().set_xlabel("{} - {}".format(mag_color_ref[0], mag_color_ref[1]))
        plt.gca().set_ylabel("{} - {}".format(y_color[0], y_color[1]))
        plt.gca().set_ylim((-0.8,0.8))
        plt.gca().set_xlim((color_range[0]-0.1, color_range[1]+0.1))
        plt.savefig(savefig_path + "{}-{}.png".format(y_color[0], y_color[1]))
        plt.clf()
        plt.close()
        #######


    b = np.array(b)

    offsets, residuals, rank, s = np.linalg.lstsq(a, b)

    offsets_dict = {}

    for filter, offset in zip(filters, offsets):
        offsets_dict[filter] = offset


    zp_write(offsets_dict, save_file, filters)


def get_narrow_band_offsets(calibrated_catalog, save_file, models_file, mag_color_ref,
                            mag_ref, mags_to_get_zp, color_range, Nbins, fig_path, remove_minmax = True,
                            min_mag = 13.5, max_mag = 17.5):


    # Load models
    models = load_data(models_file)

    # Select only models in the right range
    models_color = models.loc[:, mag_color_ref[0]].values - models.loc[:, mag_color_ref[1]].values

    model_selection = (models_color >= color_range[0]) & (models_color <= color_range[1])
    model_low_ebv = models.loc[:, 'EB_V'].values <= 0.12

    models = models[model_selection & model_low_ebv]

    # Load data
    data = load_data(calibrated_catalog)

    mag_cut = data.loc[:,mag_color_ref[0]] <= max_mag
    mag_cut = mag_cut & (data.loc[:,mag_color_ref[0]] >= min_mag)

    mag_cut = mag_cut & (data.loc[:,mag_color_ref[1]] <= max_mag)
    mag_cut = mag_cut & (data.loc[:,mag_color_ref[1]] >= min_mag)

    mag_cut = mag_cut & (data.loc[:,mag_ref] <= max_mag)
    mag_cut = mag_cut & (data.loc[:,mag_ref] >= min_mag)

    data = data[mag_cut]

    # select only data in the right range
    data_color = data.loc[:, mag_color_ref[0]].values - data.loc[:, mag_color_ref[1]].values

    data_selection_range = (data_color >= color_range[0]) & (data_color <= color_range[1])

    data_selection_mag_cut = data.loc[:, 'SPLUS_R'].values <= 17.5

    data = data[data_selection_range & data_selection_mag_cut]


    # prepare data
    models_x = models.loc[:, mag_color_ref[0]] - models.loc[:, mag_color_ref[1]]
    data_x = data.loc[:, mag_color_ref[0]] - data.loc[:, mag_color_ref[1]]


    zp_dict = {}

    bins = np.linspace(color_range[0], color_range[1], Nbins)
    delta_bin = bins[1] - bins[0]

    # Obtain zero points
    for mag in mags_to_get_zp:
        print("Estimating ZP for mag %s using the stellar locus" % mag)

        if mag == mag_ref:
            zp_dict[mag] = np.nan
            continue

        delta_mag = []

        ####
        models_bin_y_list = []
        data_bin_y_list = []
        ####

        # Remove mag = 99 or -99
        remove_bad_data = (data.loc[:, mag].values >= min_mag) & (data.loc[:, mag].values <= max_mag)

        for bin in bins[:-1]:
            models_bin_cut = (models_x >= bin) & (models_x < bin + delta_bin)
            data_bin_cut = (data_x >= bin) & (data_x < bin + delta_bin) & remove_bad_data

            models_bin_y = models.loc[models_bin_cut, mag] - models.loc[models_bin_cut, mag_ref]
            data_bin_y = data.loc[data_bin_cut, mag] - data.loc[data_bin_cut, mag_ref]

            mean_models_bin_y = mean_robust(models_bin_y)
            mean_data_bin_y = mean_robust(data_bin_y, 0.5, 0.5)

            delta_mag.append(mean_models_bin_y - mean_data_bin_y)

            ####
            models_bin_y_list.append(mean_models_bin_y)
            data_bin_y_list.append(mean_data_bin_y)
            ####

        # Calculate ZP
        delta_mag = np.array(delta_mag)
        models_bin_y_list = np.array(models_bin_y_list)
        data_bin_y_list = np.array(data_bin_y_list)

        # Get order to remove max and min values
        o = np.argsort(delta_mag)

        zp_dict[mag] = mean_robust(delta_mag[o][1:-1])
        print("{:s} ZP = {:.3f}".format(mag, zp_dict[mag]))

        #######
        x = np.array(bins[:-1]) + delta_bin / 2

        zorder = -2
        alpha = 0.2
        if len(models_x) < len(data_x):
            zorder = 0
            alpha = 1

        plt.scatter(models_x, models.loc[:, mag] - models.loc[:, mag_ref], label = 'model', zorder=zorder, alpha=alpha)
        plt.scatter(x[o][1:-1], models_bin_y_list[o][1:-1], s=100, c="#000066")
        plt.scatter(x[o][0], models_bin_y_list[o][0], s=100, c="#000000", marker = 'x')
        plt.scatter(x[o][-1], models_bin_y_list[o][-1], s=100, c="#000000", marker = 'x')

        plt.plot(x, models_bin_y_list, color="#000000")

        plt.scatter(data_x, data.loc[:, mag] - data.loc[:, mag_ref], label = 'data', c="#FF0000", zorder=-1, alpha=0.2)
        plt.scatter(x[o][1:-1], data_bin_y_list[o][1:-1], s=100, c="#660000")
        plt.scatter(x[o][0], data_bin_y_list[o][0], s=100, c="#000000", marker = 'x')
        plt.scatter(x[o][-1], data_bin_y_list[o][-1], s=100, c="#000000", marker = 'x')
        plt.plot(x, data_bin_y_list, color="#000000")

        plt.gca().set_xlim(color_range)
        plt.gca().set_ylim(-1,1)

        plt.gca().set_xlabel("{} - {}".format(mag_color_ref[0], mag_color_ref[1]))
        plt.gca().set_ylabel("{} - {}".format(mag, mag_ref))

        if mag in ["SPLUS_U", "SPLUS_F378", "SPLUS_F395"]:
            plt.gca().set_ylim(-0.5, 2.5)

        plt.legend()

        plt.savefig(fig_path+"/{}_{}.png".format(mag, mag_ref))
        plt.clf()
        plt.close()
        #######

    zp_write(zp_dict=zp_dict, save_file=save_file, mag_cols=mags_to_get_zp)



# ******************************************************************************
#
# PLOTS
#
# ******************************************************************************

filters_lambda_eff = {'SDSS_U': 3587.16, 'SDSS_G': 4769.82, 'SDSS_R': 6179.99,
              'SDSS_I': 7485.86, 'SDSS_Z': 8933.86,
              'SPLUS_U': 3511.74, 'SPLUS_F378': 3784.04, 'SPLUS_F395': 3951.82,
              'SPLUS_F410': 4104.50, 'SPLUS_F430': 4303.83, 'SPLUS_G': 4821.55,
              'SPLUS_F515': 5153.59, 'SPLUS_R': 6255.93, 'SPLUS_F660': 6603.23,
              'SPLUS_I': 7696.31, 'SPLUS_F861': 8615.71, 'SPLUS_Z': 9609.69,
              'GAIA_G': 6437.70, 'GAIA_BP': 5309.57, 'GAIA_RP': 7709.85}

# def wavelength_to_hex(wavelength, gamma=0.8):
#     '''This converts a given wavelength of light to an
#     approximate hex color value. The wavelength must be given
#     in angstrons in the range from 3000 A through 10000 A
#
#     Based on code by Dan Bruton
#     http://www.physics.sfasu.edu/astro/color/spectra.html
#     '''
#
#     wavelength = float(wavelength) / 10.
#
#     if wavelength >= 300 and wavelength <= 440:
#         attenuation = 0.3 + 0.7 * (wavelength - 300) / (440 - 300)
#         R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
#         G = 0.0
#         B = (1.0 * attenuation) ** gamma
#     elif wavelength >= 440 and wavelength <= 490:
#         R = 0.0
#         G = ((wavelength - 440) / (490 - 440)) ** gamma
#         B = 1.0
#     elif wavelength >= 490 and wavelength <= 510:
#         R = 0.0
#         G = 1.0
#         B = (-(wavelength - 510) / (510 - 490)) ** gamma
#     elif wavelength >= 510 and wavelength <= 580:
#         R = ((wavelength - 510) / (580 - 510)) ** gamma
#         G = 1.0
#         B = 0.0
#     elif wavelength >= 580 and wavelength <= 645:
#         R = 1.0
#         G = (-(wavelength - 645) / (645 - 580)) ** gamma
#         B = 0.0
#     elif wavelength >= 645 and wavelength <= 1000:
#         attenuation = 0.3 + 0.7 * (1000 - wavelength) / (1000 - 645)
#         R = (1.0 * attenuation) ** gamma
#         G = 0.0
#         B = 0.0
#     else:
#         R = 0.0
#         G = 0.0
#         B = 0.0
#     R *= 255
#     G *= 255
#     B *= 255
#
#     color = '#%02x%02x%02x' % (R, G, B)
#     return color


def wavelength_to_hex(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an
    approximate hex color value. The wavelength must be given
    in angstrons in the range from 3000 A through 10000 A

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength) / 10.

    if wavelength >= 200 and wavelength < 300:
        attenuation = 0.3 + 0.7 * (wavelength - 200) / (440 - 200)
        R = ((-(wavelength - 440) / (440 - 200)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 300 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 300) / (440 - 300)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 1000:
        attenuation = 0.3 + 0.7 * (1000 - wavelength) / (1000 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255

    color = '#%02x%02x%02x' % (R, G, B)
    return color


def get_catalog_offsets(crossmatched_catalog, save_file, filters, fig_path):

    # load data
    data = load_data(crossmatched_catalog)

    # get offsets for each band
    zp_dict = {}

    for mag in filters:
        mag_cut = data.loc[:,mag+"_2"].values <= 17.5

        offset = data.loc[mag_cut,mag+"_1"] - data.loc[mag_cut,mag+"_2"]

        offset = offset[~np.isnan(offset)]

        zp_dict[mag] = np.mean(offset)

        print("Offset for mag {}: mean {:.3f} | std {:.3f}".format(mag,
                                                                   offset.mean(),
                                                                   offset.std()))

        #########
        plt.hist(offset, bins = np.linspace(-0.05, 0.05, 30))
        plt.gca().set_xlim(-0.05, 0.05)

        plt.savefig(fig_path+"/{}.png".format(mag))
        plt.clf()
        plt.close()
        #########

    zp_write(zp_dict, save_file, filters)

# ZP comparisons ###############################################################

def plot_ZP_fitting(model_fitting_file, zp_file, save_file, filter_name,
                    mag_cut, data_file=None,
                    only_half_best_chi2=False):
    if data_file is None:
        data_file = model_fitting_file

    model_data = load_data(model_fitting_file)
    obs_data = load_data(data_file)

    if zp_file is not None:
        zp = zp_read(zp_file)

    residual = model_data.loc[:, filter_name + '_mod'] - obs_data.loc[:, filter_name]

    # Getting the limits of the plot #######

    mode = get_mode(obs_mag_array=obs_data.loc[:, filter_name],
                    model_mag_array=model_data.loc[:, filter_name + '_mod'])
    around_mode = (residual >= mode - 1) & (residual <= mode + 1)
    sigma = np.std(residual[around_mode])

    lim = np.max((np.abs(mode + 5 * sigma), np.abs(mode - 5 * sigma)))

    ########################################

    # Making the plot ######################

    x = model_data.loc[:, filter_name + '_mod']
    y = residual

    # Plot residual data

    plt.scatter(x, y, c="#2266ff", s=20, alpha=0.5, zorder=2)

    # If using only half best chi2 to fit, plot it
    if only_half_best_chi2:
        chi2_half = get_half_array_value(model_data.loc[:, 'chi2'].values)
        condition = model_data.loc[:, 'chi2'].values < chi2_half

        plt.scatter(x[condition], y[condition], c="#ff6622", s=20, alpha=0.5, zorder=3)

    # Plot calculated ZP
    if zp_file is not None:
        plt.plot([mag_cut[0], mag_cut[1]], [zp[filter_name], zp[filter_name]], color='#FF3219', linestyle='-', zorder=6)

    plt.axhline(y=0, color='#000000', linestyle='--', zorder=1)
    plt.grid(zorder=0)

    plt.gca().set_xlim((13, 22))
    plt.gca().set_ylim((-lim, lim))

    plt.savefig(save_file)
    plt.clf()
    plt.close()

    ########################################


def plot_ZP_fitting_v2(model_fitting_file, zp_file, save_file, filter_name, mag_cut, only_half_best_chi2, data_file = None):

    data = load_data(model_fitting_file)

    zp = zp_read(zp_file)
    zp = zp[filter_name]

    # Remove 99
    data_selection = np.abs(data.loc[:, filter_name] < 50)
    data = data[data_selection]

    x = data.loc[:, filter_name + '_mod']
    y = x - data.loc[:, filter_name]

    dwarfs = data.loc[:, 'logg'].values > 3

    selection = (x >= mag_cut[0]) & (x <= mag_cut[1]) & dwarfs

    if only_half_best_chi2:
        chi2_half = get_half_array_value(data.loc[:, 'chi2'].values)
        selection = selection & (data.loc[:, 'chi2'].values < chi2_half)

    ###
    # Calculate KDE distribution
    ###
    delta = y[selection]
    delta = delta.values.reshape(-1, 1)

    kde_dens = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(delta)

    y_dens = np.arange(-10, 10, 0.001)
    x_dens = np.exp(kde_dens.score_samples(y_dens.reshape(-1, 1)))

    mode = y_dens[x_dens == np.max(x_dens)][0]

    mu = np.mean(y[selection])
    mu_robust = mean_robust(y[selection])

    ###
    # Make plot
    ###
    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = plt.subplot(gs[0])

    ax1.scatter(x[selection], y[selection], c="#2266ff", s=20, alpha=0.5, zorder=2)
    ax1.scatter(x[~selection], y[~selection], c="#66aaff", s=20, alpha=0.1, zorder=2)
    ax1.plot([mag_cut[0], mag_cut[1]], [mode, mode], color='#FF3219', linestyle='-', zorder=6)

    ####
    # Limits of the plot
    ####
    xlim = [mag_cut[0] - 2, mag_cut[1] + 4]
    ylim = [zp - 1.5, zp + 1.5]

    ax1.text(mag_cut[0] - 1.5, zp + 1.3, "%s" % filter_name, fontsize=14)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Model - Instrumental")

    ###
    # Plot KDE distribution
    ###

    ax2 = plt.subplot(gs[1])

    ax2.plot(x_dens / np.max(x_dens), y_dens, zorder=3)

    ax2.plot([0, 2], [mode, mode], color='#000000', linestyle='--', zorder=3, label='zp: {:.4f}'.format(mode))
    ax2.plot([0, 2], [mode, mode], color='#FF3219', linestyle='-', zorder=2, label='mode: {:.4f}'.format(mode))

    ax2.plot([0, 2], [mu, mu], color='#1932FF', linestyle='--', zorder=1, label='mean: {:.4f}'.format(mu))
    ax2.plot([0, 2], [mu_robust, mu_robust], color='#19DD32', linestyle=':', zorder=1,
             label='mean_robust: {:.4f}'.format(mu_robust))

    ax2.set_xlim([0, 1.1])
    ax2.set_ylim(ylim)

    ax2.legend(fontsize=7)

    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_ylabel("Density")

    ####
    # Plot grids
    ####

    for i in np.arange(-10, 10, 0.5):
        ax1.plot([0, 30], [i, i], color="#666666", alpha=0.3, linewidth=0.5, zorder=-5)
        ax2.plot([0, 30], [i, i], color="#666666", alpha=0.3, linewidth=0.5, zorder=-5)

    for i in np.arange(-10, 10, 0.1):
        ax1.plot([0, 30], [i, i], color="#666666", alpha=0.1, linewidth=0.5, zorder=-5)
        ax2.plot([0, 30], [i, i], color="#666666", alpha=0.1, linewidth=0.5, zorder=-5)

    for i in np.arange(1, 30, 1):
        ax1.plot([i, i], [-10, 10], color="#666666", linewidth=0.5, alpha=0.1, zorder=-5)

    for i in np.arange(2, 30, 2):
        ax1.plot([i, i], [-10, 10], color="#666666", linewidth=0.5, alpha=0.3, zorder=-5)

    plt.subplots_adjust(top=0.98, left=0.1, right=0.98, wspace=0)

    plt.savefig(save_file)
    plt.clf()
    plt.close()


def get_real_zp(zp_reference_file, field, filter_name):
    # Try to get reference ZP
    ZP_reference = load_data(zp_reference_file)

    if field in list(ZP_reference['field']):

        if not np.isnan(ZP_reference.loc[ZP_reference['field'] == field, filter_name]).values[0]:
            zp = ZP_reference.loc[ZP_reference['field'] == field, filter_name].values[0]
            zp_err = ZP_reference.loc[ZP_reference['field'] == field, filter_name + "_ERR"].values[0]

            return np.array([zp, zp_err])

    return None


def plot_ZP_diagnostic(initial_zp_file, relative_zp_file, absolute_zp_file, filter_names, save_file, field,
                       sex_mag_zp=20, reference_zp_dict=None):
    # Read ZP files
    initial_zp = zp_read(initial_zp_file)
    relative_zp = zp_read(relative_zp_file)
    absolute_zp = zp_read(absolute_zp_file)

    filter_names_label = []
    for filter_name in filter_names:
        if 'SPLUS_' in filter_name:
            filter_names_label.append(filter_name.split("SPLUS_")[-1])
        else:
            filter_names_label.append(filter_name)

    plt.figure(figsize=(15, 5))
    plt.gca().set_xlim(-1, len(filter_names))
    plt.gca().set_ylim(17, 24)
    plt.gca().set_ylabel("mag")
    plt.gca().set_title(field)
    plt.subplots_adjust(left=0.04, right=0.98, bottom=0.07, top=0.94)
    plt.xticks(range(len(filter_names)), filter_names_label)

    y_all = []

    for i in range(len(filter_names)):
        filter_name = filter_names[i]

        # plot sex_mag_zp
        y_sex = sex_mag_zp

        l1, = plt.plot([i, i], [0, y_sex], color="#000000", linewidth=1, zorder=1)
        plt.plot([i - 0.1, i + 0.1], [y_sex, y_sex], color="#000000", linewidth=0.5, zorder=2)

        # include initial_zp
        y_ini = y_sex + initial_zp[filter_name]

        l2, = plt.plot([i, i], [y_sex, y_ini], color="#1459FF", linewidth=2, zorder=3)
        plt.plot([i - 0.2, i + 0.2], [y_ini, y_ini], color="#1459FF", linewidth=0.5, zorder=4)

        # include relative_zp
        y_rel = y_ini + relative_zp[filter_name]

        l3, = plt.plot([i, i], [y_ini, y_rel], color="#19FF26", linewidth=2, zorder=5)
        plt.plot([i - 0.3, i + 0.3], [y_rel, y_rel], color="#19FF26", linewidth=0.5, zorder=6)

        # include absolute_zp
        y_abs = y_rel + absolute_zp[filter_name]

        l4, = plt.plot([i, i], [y_rel, y_abs], color="#FF2A16", linewidth=2, zorder=7)
        plt.scatter([i], [y_abs], c="#FF2A16", s=20, zorder=8)

        y_all.append(y_abs)

    plt.plot(range(len(filter_names)), y_all, color="#FF2A16", linewidth=4, alpha=0.5, zorder=0)

    # Plot reference zps
    ref_plots = []
    ref_labels = []

    if reference_zp_dict is not None:

        for ref in reference_zp_dict.keys():
            print(ref)

            ref_zps_dict = read_zp_table(zp_table_file=reference_zp_dict[ref]["zp_table"],
                                         field=field, filter_names=filter_names)

            ref_filters = []
            ref_zps = []

            for i in range(len(filter_names)):
                if ~np.isnan(ref_zps_dict[filter_names[i]]):
                    ref_filters.append(i)
                    ref_zps.append(ref_zps_dict[filter_names[i]])

            r_i, = plt.plot(ref_filters, ref_zps, color=reference_zp_dict[ref]["color"], alpha=0.5, zorder=-1)
            ref_plots.append(r_i)
            ref_labels.append("ZPs: " + ref)

    lines = [l1, l2, l3, l4]
    labels = ["sex_MAG_ZEROPOINT", "+ initial_zp", "+ relative_zp", "+ absolute_zp"]
    legend1 = plt.legend(lines, labels, loc=1, ncol=4)

    plt.legend(ref_plots, ref_labels, loc=2, ncol=1)

    plt.grid(zorder=-100, alpha=0.5)
    plt.gca().add_artist(legend1)
    print(save_file)
    plt.savefig(save_file)


def read_zp_table(zp_table_file, field, filter_names):
    zp_table = load_data(zp_table_file)

    zp_dict = {}

    for filter_name in filter_names:
        try:
            select_field = zp_table["field"] == field
            zp_dict[filter_name] = zp_table.loc[select_field, filter_name].values[0]
        except:
            zp_dict[filter_name] = np.nan

    return zp_dict


################################################################################

# Best Fit #####################################################################

def plot_best_fit(model_fitting_file, zp_file, save_file, filters_lambda_eff,
                  filters_plot_marker=None, filters_plot_size=None,
                  plot_title=None, ylim=[15, 24]):
    # Order filters by effective wavelength ##################
    filter_name_list = filters_lambda_eff.keys()
    lambda_eff_list = []
    marker_list = []
    size_list = []

    for filter_name in filter_name_list:
        lambda_eff_list.append(filters_lambda_eff[filter_name])

        if filters_plot_marker is not None:
            marker_list.append(filters_plot_marker[filter_name])
        else:
            marker_list.append('o')

        if filters_plot_size is not None:
            size_list.append(filters_plot_size[filter_name])
        else:
            size_list.append(100)

    lambda_eff_data = {}
    lambda_eff_data['filter'] = filter_name_list
    lambda_eff_data['lambda_eff'] = lambda_eff_list
    lambda_eff_data['marker'] = marker_list
    lambda_eff_data['marker_size'] = size_list

    lambda_eff_data = pd.DataFrame(lambda_eff_data)
    lambda_eff_data = lambda_eff_data.sort_values("lambda_eff")

    ############################################################

    # Load model data
    model_data = load_data(model_fitting_file)

    # Order the data by chi2
    model_data = model_data.sort_values("chi2")

    # Load estimated zp values
    zp_data = zp_read(zp_file)

    #############################################################

    # Make the plot

    fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(10, 10))

    axs = []
    for i in range(5):
        ax[i, 0].set_ylabel("magnitude")
        for j in range(2):
            axs.append(ax[i, j])

    ax[4, 0].set_xlabel("wavelength")
    ax[4, 1].set_xlabel("wavelength")

    plt.subplots_adjust(hspace=0, wspace=0, top=0.96, right=0.96, left=0.1, bottom=0.1)
    plt.gca().set_ylim(ylim)

    if plot_title is not None:
        ax[0, 0].set_title(plot_title)

    # Plot for the 10 best fits
    for i in range(10):

        best_fit = model_data.iloc[i, :]

        # Get the best fit model
        model_lambdas = []
        model_mags = []

        for filter_name in lambda_eff_data['filter']:
            if filter_name in best_fit.keys():
                model_mags.append(best_fit[filter_name + '_mod'])

                select_filter = lambda_eff_data['filter'] == filter_name
                model_lambdas.append(lambda_eff_data.loc[select_filter, 'lambda_eff'].values[0])

        # First, plot the adjusted model

        axs[i].plot(model_lambdas, model_mags, color="#666666", linewidth=2, zorder=0)

        # Second, plot observed data

        for filter_name in lambda_eff_data['filter']:

            # only try to plot the ones present in the model_fitting_file
            if filter_name in best_fit.keys():
                select_filter = lambda_eff_data['filter'] == filter_name

                lambda_eff = lambda_eff_data.loc[select_filter, 'lambda_eff'].values[0]
                mag_obs = best_fit[filter_name]

                axs[i].scatter(lambda_eff, mag_obs,
                               marker=lambda_eff_data.loc[select_filter, 'marker'].values[0],
                               s=lambda_eff_data.loc[select_filter, 'marker_size'].values[0],
                               c=wavelength_to_hex(lambda_eff),
                               zorder=2)

        # Finally, plot the estimated zero points
        for filter_name in zp_data.keys():
            mag_obs = best_fit[filter_name]
            mag_zp = zp_data[filter_name]

            select_filter = lambda_eff_data['filter'] == filter_name
            lambda_eff = lambda_eff_data.loc[select_filter, 'lambda_eff'].values[0]

            axs[i].plot([lambda_eff, lambda_eff],
                        [mag_obs, mag_obs + mag_zp],
                        color=wavelength_to_hex(lambda_eff),
                        alpha=0.5,
                        zorder=1)

            axs[i].plot([lambda_eff - 50, lambda_eff + 50],
                        [mag_obs + mag_zp, mag_obs + mag_zp],
                        color=wavelength_to_hex(lambda_eff),
                        alpha=0.5,
                        zorder=1)

            axs[i].grid(zorder=-100)

    plt.savefig(save_file)
    plt.clf()
    plt.close()


# ******************************************************************************
#
# XY CORRECTIONS
#
# ******************************************************************************

def get_XY_correction_grid(data_file, save_file, mag, xbins, ybins):

    xNbins = xbins[2]
    yNbins = ybins[2]

    # Get values of bins limits and centers

    xbins = np.linspace(xbins[0], xbins[1], xbins[2]+1)
    ybins = np.linspace(ybins[0], ybins[1], ybins[2]+1)

    # generate the mesh

    xx, yy = np.meshgrid(xbins, ybins, sparse=True)

    corrections = 0*xx + 0*yy
    corrections_std = np.nan*xx + np.nan*yy

    # Load data
    data = load_data(data_file)

    X = data.loc[:,'X'].values
    Y = data.loc[:,'Y'].values

    # Normalize X and Y
    X = X - np.nanmin(X)
    Y = Y - np.nanmin(Y)

    DZP = data.loc[:,'DZP_%s' % mag].values

    mag_cut = (data.loc[:, mag].values > 14) & (data.loc[:, mag].values <= 17.5)
    remove_worst_cases = np.abs(DZP) < 0.2

    # Fill array of corrections
    N_data = len(DZP[mag_cut & remove_worst_cases])
    min_N_data = 0.05*N_data/(xNbins*yNbins)

    for i in range(xNbins):
        xselect = (X >= xbins[i]) & (X < xbins[i+1])

        for j in range(yNbins):
            yselect = (Y >= ybins[j]) & (Y < ybins[j+1])

            DZP_select = DZP[xselect & yselect & mag_cut & remove_worst_cases]

            if len(DZP_select) > min_N_data:
                corrections[i,j] = np.mean(DZP_select)

            corrections_std[i,j] = np.std(DZP_select)

    corrections = gaussian_filter(corrections, sigma=1)


    xid0 = int((xNbins/5))
    xidf = xNbins - int((xNbins/5))

    yid0 = int((yNbins/5))
    yidf = yNbins - int((yNbins/5))


    # Take out high values from the borders ####################################

    vmax = np.nanmax(corrections[xid0:xidf, yid0:yidf])
    vmin = np.nanmin(corrections[xid0:xidf, yid0:yidf])

    # corrections[:xid0, :][corrections[:xid0, :] > vmax] = vmax
    # corrections[:xid0, :][corrections[:xid0, :] < vmin] = vmin
    #
    # corrections[:, :yid0][corrections[:, :yid0] > vmax] = vmax
    # corrections[:, :yid0][corrections[:, :yid0] < vmin] = vmin
    #
    # corrections[xidf:, :][corrections[xidf:, :] > vmax] = vmax
    # corrections[xidf:, :][corrections[xidf:, :] < vmin] = vmin
    #
    # corrections[:, yidf:][corrections[:, yidf:] > vmax] = vmax
    # corrections[:, yidf:][corrections[:, yidf:] < vmin] = vmin

    # Scale offset by mean value (offsets are dealt with in another step #########
    corrections = corrections - np.nanmean(DZP[mag_cut & remove_worst_cases])


    # Remove nan values
    corrections[np.isnan(corrections)] = 0

    np.save(save_file, corrections)


def get_XY_reference_comparison(data_file, save_file, mag, xbins, ybins, ref):

    xNbins = xbins[2]
    yNbins = ybins[2]

    # Get values of bins limits and centers

    xbins = np.linspace(xbins[0], xbins[1], xbins[2]+1)
    ybins = np.linspace(ybins[0], ybins[1], ybins[2]+1)

    # generate the mesh

    xx, yy = np.meshgrid(xbins, ybins, sparse=True)

    corrections = 0*xx + 0*yy
    corrections_std = np.nan*xx + np.nan*yy

    # Load data
    data = load_data(data_file)

    X = data.loc[:,'X'].values
    Y = data.loc[:,'Y'].values

    # Normalize X and Y
    X = X - np.nanmin(X)
    Y = Y - np.nanmin(Y)

    DZP = data.loc[:,'{}_{}'.format(ref, mag)].values -  data.loc[:,'SPLUS_%s' % mag].values

    mag_cut = (data.loc[:, 'SPLUS_%s' % mag].values > 14) & (data.loc[:, 'SPLUS_%s' % mag].values <= 17.5)
    remove_worst_cases = np.abs(DZP) < 0.2

    # Fill array of corrections
    N_data = len(DZP[mag_cut & remove_worst_cases])
    min_N_data = 0.05*N_data/(xNbins*yNbins)

    for i in range(xNbins):
        xselect = (X >= xbins[i]) & (X < xbins[i+1])

        for j in range(yNbins):
            yselect = (Y >= ybins[j]) & (Y < ybins[j+1])

            DZP_select = DZP[xselect & yselect & mag_cut & remove_worst_cases]

            if len(DZP_select) > min_N_data:
                corrections[i,j] = np.mean(DZP_select)

            corrections_std[i,j] = np.std(DZP_select)

    corrections = gaussian_filter(corrections, sigma=1)


    # Scale offset by mean value (offsets are dealt with in another step #########
    corrections = corrections - np.nanmean(DZP[mag_cut & remove_worst_cases])


    # Remove nan values
    corrections[np.isnan(corrections)] = 0

    np.save(save_file, corrections)



def plot_XY_correction_grid(grid_file, save_file, mag, xbins, ybins, cmap = None, clim = [-0.02, 0.02]):

    # Get values of bins limits and centers

    xbins_grid = np.linspace(xbins[0], xbins[1], xbins[2] + 1)
    ybins_grid = np.linspace(ybins[0], ybins[1], ybins[2] + 1)

    # generate the mesh

    xx, yy = np.meshgrid(xbins_grid, ybins_grid, sparse=True)

    corrections = np.load(grid_file)

    plt.figure(figsize=(8, 6.4))

    mean = np.nanmean(corrections)

    #vmin = mean - np.max((np.abs(mean - np.min(corrections)), np.abs(np.max(corrections) - mean)))
    #vmax = mean + np.max((np.abs(mean - np.min(corrections)), np.abs(np.max(corrections) - mean)))

    vmin = clim[0]
    vmax = clim[1]

    if cmap is None:
        cmap = plt.get_cmap("seismic_r")

    cm = plt.pcolor(xx, yy, corrections.T, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(cm)
    cbar.set_label("offset")

    plt.vlines(xbins_grid, xbins[0], xbins[1], linewidth=0.5, alpha=0.4)
    plt.hlines(ybins_grid, ybins[0], ybins[1], linewidth=0.5, alpha=0.4)

    plt.gca().set_title("%s offsets" % mag)
    plt.gca().set_xlabel("X")
    plt.gca().set_ylabel("Y")
    plt.gca().set_xlim((xbins[0], xbins[1]))
    plt.gca().set_ylim((ybins[0], ybins[1]))
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.11, right=0.98)

    plt.savefig(save_file)
    plt.clf()
    plt.close()


def apply_XY_offsets(data_file, save_file, grid_file_dict, xbins, ybins):

    # Prepare bins

    xNbins = xbins[2]
    yNbins = ybins[2]

    xbins_grid = np.linspace(xbins[0], xbins[1], xbins[2] + 1)
    ybins_grid = np.linspace(ybins[0], ybins[1], ybins[2] + 1)


    # Load data

    data = load_data(data_file)

    X = data.loc[:,'X'].values
    Y = data.loc[:,'Y'].values

    # Normalize X and Y
    X = X - np.nanmin(X)
    Y = Y - np.nanmin(Y)

    # get mag list

    filters = grid_file_dict.keys()
    corrections = {}

    for mag in filters:

        print("Applying corrections to columns '%s'" % mag)
        # Load grid of offsets for this mag
        grid_file = grid_file_dict[mag]
        print("loaded grid from %s" % grid_file)
        print("")
        corrections[mag] = np.load(grid_file)

        # Apply offsets
        for i in range(xNbins):
            xselect = (X >= xbins_grid[i]) & (X < xbins_grid[i + 1])

            for j in range(yNbins):
                yselect = (Y >= ybins_grid[j]) & (Y < ybins_grid[j + 1])

                if len(data[mag][xselect & yselect & ~np.isnan(data[mag])]) != 0:

                    data.loc[xselect & yselect & ~np.isnan(data[mag]), mag] += corrections[mag][i,j]


    print("Saving to %s" % save_file)

    with open(save_file, "w") as f:
            f.write("# ")
            data.to_csv(f, na_rep = np.nan, sep = " ", index = False)


def apply_XY_corrections_to_photometry_catalog(catalog, save_path, grid_file, xbins, ybins):

    # Load columns from catalog #
    cat = fits.open(catalog)
    cat_data = cat[1].data

    X = cat_data.columns['X_IMAGE'].array
    Y = cat_data.columns['Y_IMAGE'].array

    # Normalize X and Y
    X = X - np.nanmin(X)
    Y = Y - np.nanmin(Y)

    # Prepare bins
    xbins_grid = np.linspace(xbins[0], xbins[1], xbins[2] + 1)
    ybins_grid = np.linspace(ybins[0], ybins[1], ybins[2] + 1)

    xbins_id = np.array(range(xbins[2]+1))
    ybins_id = np.array(range(ybins[2]+1))

    # Load corrections
    corrections = np.load(grid_file)

    # Apply corrections
    for k in range(len(cat_data)):

        X_source = X[k]
        Y_source = Y[k]

        # Find ids of the bin that contains the source
        bin_source_i = xbins_id[xbins_grid <= X_source][-1]
        bin_source_j = ybins_id[ybins_grid <= Y_source][-1]

        delta_mag = corrections[bin_source_i, bin_source_j]
        delta_flux_frac = 10.0**(-(delta_mag/2.5))

        cat_data.columns['MAG_AUTO'].array[k] += delta_mag
        cat_data.columns['FLUX_AUTO'].array[k] *= delta_flux_frac

        cat_data.columns['MAG_ISO'].array[k] += delta_mag
        cat_data.columns['FLUX_ISO'].array[k] *= delta_flux_frac

        cat_data.columns['MAG_PETRO'].array[k] += delta_mag
        cat_data.columns['FLUX_PETRO'].array[k] *= delta_flux_frac

        cat_data.columns['MAG_APER'].array[k] += delta_mag
        cat_data.columns['FLUX_APER'].array[k] *= delta_flux_frac


    # Save master HDU
    cat.writeto(save_path)
    print('Created file %s' % save_path)


# ******************************************************************************
#
# Magnitude Upper limits (from Laura/Alberto pipeline)
#
# ******************************************************************************


def get_limitingmagnitude(m, dm, n_sigma=2., dm_int=0.25, tag=None):
    """Given a list of magnitudes and magnitude errors,
    calculate by extrapolation the n_sigma error limit"""
    # np = len(m)
    g = np.less(m, 99.) * np.greater(m, 00.)
    y, x = autobin_stats(np.compress(g, dm), np.compress(g, m), n_points=15, stat="median")

    # In case the error dm_int is not contained in the data set
    if dm_int >= y[-2] or dm_int < y[0]:
        dm_int = y[-3]  # Take third point from the end to avoid limit effects
    mlim = match_resol(y, x, dm_int) - flux2mag(1. / n_sigma / e_mag2frac(dm_int))

    return mlim


def autobin_stats(x, y, n_bins=8, stat='average', n_points=None, xmed=0):
    """
    Given the variable y=f(x), form n_bins, distributing the
    points equally among them. Return the average x position
    of the points in each bin, and the corresponding statistic stat(y).
    n_points supersedes the value of n_bins and makes the bins
    have exactly n_points each
    Usage:
      xb,yb=autobin_stats(x,y,n_bins=8,'median')
      xb,yb=autobin_stats(x,y,n_points=5)
    """

    if not ascend(x):
        ix = np.argsort(x)
        x = np.take(x, ix)
        y = np.take(y, ix)
    n = len(x)
    if n_points == None:
        # This throws out some points
        n_points = n / n_bins
    else:
        n_bins = n / n_points
        # if there are more that 2 points in the last bin, add another bin
        if n % n_points > 2: n_bins = n_bins + 1

    if n_points <= 1:
        print('Only 1 or less points per bin, output will be sorted input vector with rms==y')
        return x, y
    xb, yb = [], []

    if stat == 'average' or stat == 'mean':
        func = np.mean
    elif stat == 'median':
        func = np.median
    elif stat == 'rms' or stat == 'std':
        func = np.std
    #elif stat == 'std_robust' or stat == 'rms_robust':
    #    func = std_robust
    #elif stat == 'std_mad':
    #    func = std_mad
    elif stat == 'mean_robust':
        func = mean_robust
    #elif stat == 'median_robust':
    #    func = median_robust
    elif stat == 'product':
        func = np.product
    #elif stat == 'sigma_mixture':
    #    func = sigma_mixture
    #elif stat == 'n_outliers':
    #    func = n_outliers
    #elif stat == 'gt0p02':
    #    func = gt0p02

    for i in range(n_bins):
        if xmed:
            newx = np.median(x[i * n_points:(i + 1) * n_points])
        else:
            newx = np.mean(x[i * n_points:(i + 1) * n_points])
        if not np.isfinite(newx): continue
        xb.append(newx)
        if func == np.std and n_points == 2:
            print('n_points==2; too few points to determine rms')
            print('Returning abs(y1-y2)/2. in bin as rms')
            yb.append(np.abs(y[i * n_points] - y[i * n_points + 1]) / 2.)
        else:
            yb.append(func(y[i * n_points:(i + 1) * n_points]))
        if i > 2 and xb[-1] == xb[-2]:
            yb[-2] = (yb[-2] + yb[-1]) / 2.
            xb = xb[:-1]
            yb = yb[:-1]
    return np.array(xb), np.array(yb)


def match_resol(xg, yg, xf):
    """
    Interpolates and/or extrapolate yg, defined on xg, onto the xf coordinate set.
    Usage:
    ygn=match_resol(xg,yg,xf)
    """
    # If only one point available
    if len(xg) == 1 and len(yg) == 1: return xf * 0. + yg

    ng = len(xg)
    d = (yg[1:] - yg[0:-1]) / (xg[1:] - xg[0:-1])
    # Get positions of the new x coordinates
    ind = np.clip(np.searchsorted(xg, xf) - 1, 0, ng - 2)
    try:
        len(ind)
        one = 0
    except:
        one = 1
        ind = np.array([ind])
    ygn = np.take(yg, ind) + np.take(d, ind) * (xf - np.take(xg, ind))
    if one: ygn = ygn[0]
    return ygn


def ascend(x):
    """True if vector x is monotonically ascendent, false otherwise
       Recommended usage:
       if not ascend(x): sort(x)
    """
    return np.alltrue(np.greater_equal(x[1:], x[0:-1]))

def flux2mag(flux):
    """Convert arbitrary flux to magnitude"""
    return -2.5 * np.log10(flux)

def e_mag2frac(errmag):
    """Convert mag error to fractionary flux error"""
    return 10.**(.4*errmag)-1.

# OBSOLETE #####################################################################
def apply_XY_corrections_to_DR1_like_catalog(catalog, save_path, grid_file_dict, xbins, ybins):


    # Load columns from catalog #
    cat = fits.open(catalog)
    cat_data = cat[1].data

    X = cat_data.columns['X'].array
    Y = cat_data.columns['Y'].array

    # Prepare bins

    xNbins = xbins[2]
    yNbins = ybins[2]

    xbins_grid = np.linspace(xbins[0], xbins[1], xbins[2] + 1)
    ybins_grid = np.linspace(ybins[0], ybins[1], ybins[2] + 1)

    filters = grid_file_dict.keys()

    for mag in filters:

        filter = "".join(mag.split("SPLUS_"))

        print("\nApplying XY corrections to filter '%s'" % filter)
        # Load grid of offsets for this mag
        grid_file = grid_file_dict[mag]
        corrections = np.load(grid_file)

        for aperture in ["auto", 'petro', 'aper_3', 'aper_6', 'total']:
            print("    working on aperture %s" % aperture)

            # Apply offsets
            for i in range(xNbins):
                xselect = (X >= xbins_grid[i]) & (X < xbins_grid[i + 1])

                for j in range(yNbins):
                    yselect = (Y >= ybins_grid[j]) & (Y < ybins_grid[j + 1])

                    col = '{filter}_{aperture}'.format(filter=filter, aperture=aperture)
                    mag_array = cat_data.columns[col].array

                    selection = xselect & yselect & ~np.isnan(mag_array) & (np.abs(mag_array) != 99)

                    if len(mag_array[selection]) != 0:
                        cat_data.columns[col].array[selection] = mag_array[selection] + corrections[i, j]

    # Save master HDU
    cat.writeto(save_path)
    print('Created file %s' % save_path)


def apply_XY_corrections_to_DR1_like_catalog_v2(catalog, save_path, grid_file_dict, xbins, ybins):


    # Load columns from catalog #
    cat = fits.open(catalog)
    cat_data = cat[1].data

    X = cat_data.columns['X'].array
    Y = cat_data.columns['Y'].array

    # Prepare bins

    xNbins = xbins[2]
    yNbins = ybins[2]

    xbins_grid = np.linspace(xbins[0], xbins[1], xbins[2] + 1)
    ybins_grid = np.linspace(ybins[0], ybins[1], ybins[2] + 1)

    xbins_id = np.array(range(xbins[2]+1))
    ybins_id = np.array(range(ybins[2]+1))

    filters = grid_file_dict.keys()

    for mag in filters:

        filter = "".join(mag.split("SPLUS_"))

        print("\nApplying XY corrections to filter '%s'" % filter)
        # Load grid of offsets for this mag
        grid_file = grid_file_dict[mag]
        corrections = np.load(grid_file)

        for k in range(len(cat_data)):

            X_source = X[k]
            Y_source = Y[k]

            # Find ids of the bin that contains the source
            bin_source_i = xbins_id[xbins_grid <= X_source][-1]
            bin_source_j = ybins_id[ybins_grid <= Y_source][-1]

            delta_mag = corrections[bin_source_i, bin_source_j]

            cat_data.columns['%s_auto' % filter].array[k] += delta_mag
            cat_data.columns['%s_petro' % filter].array[k] += delta_mag
            cat_data.columns['%s_aper_3' % filter].array[k] += delta_mag
            cat_data.columns['%s_aper_6' % filter].array[k] += delta_mag
            cat_data.columns['%s_total' % filter].array[k] += delta_mag

    # Save master HDU
    cat.writeto(save_path)
    print('Created file %s' % save_path)
# /OBSOLETE #####################################################################


# ******************************************************************************
#
# CATALOGS
#
# ******************************************************************************

def combine_cats(list_of_files, save_file):

    frames = []
    for i in range(len(list_of_files)):
        try:
            df = load_data(list_of_files[i])
            print('loading %s' % list_of_files[i])
            frames.append(df)

        except IOError:
            pass

    df = pd.concat(frames)

    print("Saving to %s" % save_file)

    with open(save_file, "w") as f:
            f.write("# ")
            df.to_csv(f, na_rep = np.nan, sep = " ", index = False)


def make_DR1_like_catalog(field, filter_list, path_to_photometry, zp_file, aper_correction_file, calibration_aper_id,
                          save_path, fixed_aper_3_id = None, fixed_aper_6_id = None, calibration_flag = np.nan):

    master_data = []

    # Load columns from detection catalog #############
    det_cat_file = path_to_photometry + '/detection/sex_{field}_det.catalog'.format(field=field)

    det_cat = fits.open(det_cat_file)
    det_data = det_cat[1].data

    ###################################################

    # Generate first columns ##########################
    N_sources = len(det_data)

    # Generate columns Field and ID

    col_Field = N_sources * ['%s' % field]

    col_ID = []
    for i in range(N_sources):
        col_ID.append('iDR3.{:s}.{:06d}'.format(field, i))

    # Add these columns to master data

    master_data.append(fits.Column(name='Field',
                                   format='%dA' % len(field),
                                   array=col_Field))

    master_data.append(fits.Column(name='ID',
                                   format='%dA' % len(col_ID[0]),
                                   array=col_ID))

    ###################################################

    # Include columns from detection catalog ##########

    columns_from_detection = OrderedDict()
    columns_from_detection['ALPHA_J2000'] = 'RA'
    columns_from_detection['DELTA_J2000'] = 'DEC'
    columns_from_detection['X_IMAGE'] = 'X'
    columns_from_detection['Y_IMAGE'] = 'Y'
    columns_from_detection['ISOAREA_IMAGE'] = 'ISOarea'
    columns_from_detection['MU_MAX'] = 'MU_MAX'
    columns_from_detection['A_IMAGE'] = 'A'
    columns_from_detection['B_IMAGE'] = 'B'
    columns_from_detection['THETA_IMAGE'] = 'THETA'
    columns_from_detection['ELONGATION'] = 'ELONGATION'
    columns_from_detection['ELLIPTICITY'] = 'ELLIPTICITY'
    columns_from_detection['FLUX_RADIUS'] = 'FLUX_RADIUS'
    columns_from_detection['KRON_RADIUS'] = 'KRON_RADIUS'
    columns_from_detection['FLAGS'] = 'PhotoFlagDet'
    columns_from_detection['CLASS_STAR'] = 'CLASS_STAR'
    columns_from_detection['FWHM_IMAGE'] = 'FWHM'

    for col in columns_from_detection:
        name = columns_from_detection[col]
        fmt  = det_data.columns[col].format
        data = det_data.columns[col].array

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=data))

    ###################################################

    # Add calibration flag
    master_data.append(fits.Column(name='calibration_flag',
                                   format='1I',
                                   array=np.full(N_sources, calibration_flag)))

    ###################################################

    # Estimate new columns from detection cat  ########

    ### S2N_AUTO
    col_s2n_Det_auto = det_data['FLUX_AUTO'] / det_data['FLUXERR_AUTO']
    col_s2n_Det_auto = np.where(col_s2n_Det_auto > 0., col_s2n_Det_auto, -1.00)

    master_data.append(fits.Column(name='s2n_Det_auto',
                                   format='1E',
                                   array=col_s2n_Det_auto))

    ### S2N_PETRO
    col_s2n_Det_petro = det_data['FLUX_PETRO'] / det_data['FLUXERR_PETRO']
    col_s2n_Det_petro = np.where(col_s2n_Det_petro > 0., col_s2n_Det_petro, -1.00)

    master_data.append(fits.Column(name='s2n_Det_petro',
                                   format='1E',
                                   array=col_s2n_Det_petro))

    ### S2N_ISO
    col_s2n_Det_iso = det_data['FLUX_ISO'] / det_data['FLUXERR_ISO']
    col_s2n_Det_iso = np.where(col_s2n_Det_iso > 0., col_s2n_Det_iso, -1.00)

    master_data.append(fits.Column(name='s2n_Det_iso',
                                   format='1E',
                                   array=col_s2n_Det_iso))

    ### FIXED APERTURES
    for fixed_aper_id, fixed_aper_name in zip([fixed_aper_3_id, fixed_aper_6_id], ['3', '6']):

        s2n_mag = det_data.columns['FLUX_APER'].array[:, fixed_aper_id] / det_data.columns['FLUXERR_APER'].array[:, fixed_aper_id]
        s2n_mag = np.where(s2n_mag > 0., s2n_mag, -1.00)

        master_data.append(fits.Column(name='s2n_Det_aper_{aperture}'.format(aperture=fixed_aper_name),
                                       format='1E',
                                       array=s2n_mag))

    ### Normalized FWHM
    selection = (col_s2n_Det_auto >= 100) & (col_s2n_Det_auto <= 1000) & (det_data['CLASS_STAR'] > 0.9)
    mean_FWHM = mean_robust(det_data['FWHM_IMAGE'][selection])

    col_FWHM_n = det_data['FWHM_IMAGE']/mean_FWHM

    master_data.append(fits.Column(name='FWHM_n',
                                   format='1E',
                                   array=col_FWHM_n))

    ###################################################

    # Read catalogs of each filter ####################

    filter_data = {}

    for filter in filter_list:

        filter_cat_file = path_to_photometry + '/dual/sex_{field}_{filter}_dual.catalog'.format(field=field, filter=filter)
        filter_cat = fits.open(filter_cat_file)

        filter_data[filter] = filter_cat[1].data

        # Add photoflag of each filter
        name = 'PhotoFlag_%s' % filter
        fmt  = det_data.columns["FLAGS"].format
        data = filter_data[filter].columns["FLAGS"].array

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=data))

        # Add class_star of each filter
        name = 'CLASS_STAR_%s' % filter
        fmt  = filter_data[filter].columns["CLASS_STAR"].format
        data = filter_data[filter].columns["CLASS_STAR"].array

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=data))


        # Add FWHM of each filter
        name = 'FWHM_%s' % filter
        fmt  = filter_data[filter].columns["FWHM_IMAGE"].format
        data = filter_data[filter].columns["FWHM_IMAGE"].array

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=data))

    ###################################################

    # Read zp and aper correction files ###############

    ZPs = zp_read(zp_file)
    aper_corrections = zp_read(aper_correction_file)

    ###################################################

    # Add mag auto and mag petro for each filter ######

    for aperture in ['auto', 'petro', 'iso']:

        nDet_aperture = np.zeros(N_sources)

        for filter in filter_list:

            # Magnitude
            name = '{filter}_{aperture}'.format(filter=filter, aperture=aperture)
            fmt  = '1E'

            mag_sex = filter_data[filter].columns['MAG_%s' % aperture].array

            # I'm using <90 because XY correction is affecting the 99 values and this fixes it
            mag_cal = np.where(mag_sex < 90, mag_sex + ZPs["SPLUS_%s" % filter], 99)

            master_data.append(fits.Column(name=name,
                                           format=fmt,
                                           array=mag_cal))

            nDet_aperture = nDet_aperture + (np.abs(mag_cal) != 99).astype(int)

            # magnitude error

            name = 'e_{filter}_{aperture}'.format(filter=filter, aperture=aperture)
            fmt  = '1E'

            e_mag = filter_data[filter].columns['MAGERR_%s' % aperture].array

            mag_lim = get_limitingmagnitude(mag_cal, e_mag, 2., 0.25)

            #e_mag = np.where(e_mag < 0.02, 0.02, e_mag)
            e_mag = np.where(mag_cal == 99, mag_lim, e_mag)

            master_data.append(fits.Column(name=name,
                                           format=fmt,
                                           array=e_mag))

            # magnitude S2N
            name = 's2n_{filter}_{aperture}'.format(filter=filter, aperture=aperture)
            fmt  = '1E'

            s2n_mag = filter_data[filter]['FLUX_%s' % aperture]/filter_data[filter]['FLUXERR_%s' % aperture]
            np.where(s2n_mag > 0., s2n_mag, -1.00)

            master_data.append(fits.Column(name=name,
                                           format=fmt,
                                           array=s2n_mag))


        # number of detections

        name = 'nDet_{aperture}'.format(aperture=aperture)
        fmt = '1E'

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=nDet_aperture))

    ###################################################

    # Add mags for fixed apertures for each filter ####

    for fixed_aper_id, fixed_aper_name in zip([fixed_aper_3_id, fixed_aper_6_id], ['3', '6']):

        nDet_aperture = np.zeros(N_sources)

        for filter in filter_list:
            # Magnitude
            name = '{filter}_aper_{aperture}'.format(filter=filter, aperture=fixed_aper_name)
            fmt = '1E'

            mag_sex = filter_data[filter].columns['MAG_APER'].array[:, fixed_aper_id]

            # I'm using <90 because XY correction is affecting the 99 values and this fixes it
            mag_cal = np.where(mag_sex < 90, mag_sex + ZPs["SPLUS_%s" % filter], 99)

            master_data.append(fits.Column(name=name,
                                           format=fmt,
                                           array=mag_cal))

            nDet_aperture = nDet_aperture + (np.abs(mag_cal) != 99).astype(int)

            # magnitude S2N
            name = 'e_{filter}_aper_{aperture}'.format(filter=filter, aperture=fixed_aper_name)
            fmt = '1E'

            e_mag = filter_data[filter].columns['MAGERR_APER'].array[:, fixed_aper_id]

            mag_lim = get_limitingmagnitude(mag_cal, e_mag, 2., 0.25)

            #e_mag = np.where(e_mag < 0.02, 0.02, e_mag)
            e_mag = np.where(mag_cal == 99, mag_lim, e_mag)

            master_data.append(fits.Column(name=name,
                                           format=fmt,
                                           array=e_mag))

            # magnitude S2N
            name = 's2n_{filter}_aper_{aperture}'.format(filter=filter, aperture=fixed_aper_name)
            fmt = '1E'

            s2n_mag = filter_data[filter].columns['FLUX_APER'].array[:, fixed_aper_id] / filter_data[filter].columns['FLUXERR_APER'].array[:, fixed_aper_id]
            s2n_mag = np.where(s2n_mag > 0., s2n_mag, -1.00)

            master_data.append(fits.Column(name=name,
                                           format=fmt,
                                           array=s2n_mag))

        # number of detections

        name = 'nDet_aper_{aperture}'.format(aperture=fixed_aper_name)
        fmt = '1E'

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=nDet_aperture))

    ###################################################

    # Apply aperture correction to create mag_total ###

    nDet_aperture = np.zeros(N_sources)

    for filter in filter_list:

        # Magnitude
        name = '{filter}_PStotal'.format(filter=filter, aperture=fixed_aper_name)
        fmt = '1E'

        mag_sex = filter_data[filter].columns['MAG_APER'].array[:, calibration_aper_id]
        # I'm using <90 because XY correction is affecting the 99 values and this fixes it
        mag_cal = np.where(mag_sex < 90, mag_sex + ZPs["SPLUS_%s" % filter] + aper_corrections["SPLUS_%s" % filter], 99)



        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=mag_cal))

        nDet_aperture = nDet_aperture + (np.abs(mag_cal) != 99).astype(int)

        # magnitude S2N
        name = 'e_{filter}_PStotal'.format(filter=filter, aperture=fixed_aper_name)
        fmt = '1E'

        e_mag = filter_data[filter].columns['MAGERR_APER'].array[:, calibration_aper_id]

        mag_lim = get_limitingmagnitude(mag_cal, e_mag, 2., 0.25)

        #e_mag = np.where(e_mag < 0.02, 0.02, e_mag)
        e_mag = np.where(mag_cal == 99, mag_lim, e_mag)

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=e_mag))

        # magnitude S2N
        name = 's2n_{filter}_PStotal'.format(filter=filter, aperture=fixed_aper_name)
        fmt = '1E'

        s2n_mag = filter_data[filter].columns['FLUX_APER'].array[:, calibration_aper_id] / filter_data[filter].columns['FLUXERR_APER'].array[:, calibration_aper_id]
        s2n_mag = np.where(s2n_mag > 0., s2n_mag, -1.00)

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=s2n_mag))

    # number of detections

    name = 'nDet_magPStotal'
    fmt = '1E'

    master_data.append(fits.Column(name=name,
                                   format=fmt,
                                   array=nDet_aperture))

    ###################################################

    # Generate master HDU from columns
    master_hdu = fits.BinTableHDU.from_columns(master_data)

    # Save master HDU
    master_hdu.writeto(save_path)
    print('Created file %s' % save_path)


def apply_cuts_to_DR1_like_catalog(catalog, save_path, s2n_cut = 3, nDet_min = 1):

    new_data = []

    # Load columns from catalog #
    cat = fits.open(catalog)
    cat_data = cat[1].data

    selection = (cat_data.columns["nDet_auto"].array >= nDet_min)
    selection = selection & (cat_data.columns["s2n_Det_iso"].array >= s2n_cut)

    for col in cat_data.columns:
        new_data.append(fits.Column(name=col.name,
                                    format=col.format,
                                    array=col.array[selection]))

    # Generate master HDU from columns
    new_hdu = fits.BinTableHDU.from_columns(new_data)

    # Save master HDU
    new_hdu.writeto(save_path)
    print('Created file %s' % save_path)


def apply_cuts_to_DR1_like_catalog_old(catalog, save_path, err_max = 0.15, nDet_min = 1):

    new_data = []

    # Load columns from catalog #
    cat = fits.open(catalog)
    cat_data = cat[1].data

    selection = cat_data.columns["nDet_auto"].array >= nDet_min

    # Require that at least one filter has mag_auto_err < 0.1
    filters_name = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660',
                    'I', 'F861', 'Z']

    selection_mag_err = np.full(len(selection), False)
    for filtter in filters_name:
        selection_mag_err = (selection_mag_err) | (cat_data.columns["e_%s_auto" % filtter].array <= err_max)

    selection = selection & selection_mag_err

    for col in cat_data.columns:
        new_data.append(fits.Column(name=col.name,
                                    format=col.format,
                                    array=col.array[selection]))

    # Generate master HDU from columns
    new_hdu = fits.BinTableHDU.from_columns(new_data)

    # Save master HDU
    new_hdu.writeto(save_path)
    print('Created file %s' % save_path)


def make_reg_file(load_file, save_file, color, size, s2n_cut = None, nDet_min = None):
    # Load data
    table_fits = fits.open(load_file)
    table_data = table_fits[1].data

    X = table_data['X']
    Y = table_data['Y']

    r = 25 - table_data['R_AUTO']
    r = np.where(r > 20, 20, r)
    r = np.where(r < 3, 3, r)

    selection = np.full(len(X), True)

    if s2n_cut is not None:
        selection = selection & (table_data['s2n_Det_iso'] >= s2n_cut)

    if nDet_min is not None:
        selection = selection & (table_data['nDet_auto'] >= nDet_min)

    X = X[selection]
    Y = Y[selection]
    r = r[selection]

    with open(save_file, 'w') as f1:
        for i in range(len(X)):
            f1.write("point({X}, {Y}) # color={color} point=circle {size}\n".format(X=X[i],
                                                                                    Y=Y[i],
                                                                                    color=color,
                                                                                    size=int(0.5 + size*r[i])))

def get_DR1_like_catalog_selected_cols(catalog, save_path):

    keep_cols = ['Field', 'ID', 'RA', 'DEC', 'X', 'Y', 'ISOarea', 'MU_MAX', 'A', 'B', 'THETA', 'ELONGATION', 'ELLIPTICITY', 'PhotoFlagDet', 'CLASS_STAR', 'FWHM', 'U_auto', 'e_U_auto', 'F378_auto', 'e_F378_auto', 'F395_auto', 'e_F395_auto', 'F410_auto', 'e_F410_auto', 'F430_auto', 'e_F430_auto', 'G_auto', 'e_G_auto', 'F515_auto', 'e_F515_auto', 'R_auto', 'e_R_auto', 'F660_auto', 'e_F660_auto', 'I_auto', 'e_I_auto', 'F861_auto', 'e_F861_auto', 'Z_auto', 'e_Z_auto', 'nDet_auto', 'U_aper_3', 'e_U_aper_3', 'F378_aper_3', 'e_F378_aper_3', 'F395_aper_3', 'e_F395_aper_3', 'F410_aper_3', 'e_F410_aper_3', 'F430_aper_3', 'e_F430_aper_3', 'G_aper_3', 'e_G_aper_3', 'F515_aper_3', 'e_F515_aper_3', 'R_aper_3', 'e_R_aper_3', 'F660_aper_3', 'e_F660_aper_3', 'I_aper_3', 'e_I_aper_3', 'F861_aper_3', 'e_F861_aper_3', 'Z_aper_3', 'e_Z_aper_3', 'U_PStotal', 'e_U_PStotal', 'F378_PStotal', 'e_F378_PStotal', 'F395_PStotal', 'e_F395_PStotal', 'F410_PStotal', 'e_F410_PStotal', 'F430_PStotal', 'e_F430_PStotal', 'G_PStotal', 'e_G_PStotal', 'F515_PStotal', 'e_F515_PStotal', 'R_PStotal', 'e_R_PStotal', 'F660_PStotal', 'e_F660_PStotal', 'I_PStotal', 'e_I_PStotal', 'F861_PStotal', 'e_F861_PStotal', 'Z_PStotal', 'e_Z_PStotal']

    new_data = []

    # Load columns from catalog #
    cat = fits.open(catalog)
    cat_data = cat[1].data

    for col in cat_data.columns:
        if col.name in keep_cols:
            new_data.append(fits.Column(name=col.name,
                                        format=col.format,
                                        array=col.array))

    # Generate master HDU from columns
    new_hdu = fits.BinTableHDU.from_columns(new_data)

    # Save master HDU
    new_hdu.writeto(save_path)
    print('Created file %s' % save_path)


def format_final_ZP_catalog(zp_file, sex_mag_zp, save_path, field, filter_list):

    ZPs = zp_read(zp_file)

    with open(save_path, 'w') as f:
        f.write("#")

        towrite = " {:" + str(len(field)) + "}"
        f.write(towrite.format("FIELD"))

        for filter in filter_list:
            f.write(" {:>10}".format(filter))

        f.write("\n")

        f.write("  %s" % field)

        for filter in filter_list:
            f.write(" {:>10.3f}".format(sex_mag_zp + ZPs[filter]))


# ******************************************************************************
#
# ALIGN FITS
#
# # ******************************************************************************

def image_align(source, target, save_path):

    """
    Aligns array of source image to target image, and saves new fits to save_path
    """

    # Load source and target data
    source_data = fits.open(source)
    target_data = fits.open(target)

    source_array = source_data[1].data
    target_array = target_data[1].data

    # Obtain aligned array
    registered_image, footprint = aa.register(source = source_array, target = target_array)

    # Change source array to aligned array
    source_data[1].data = registered_image

    # Save to new fits file
    source_data.writeto(save_path)
    print("Created file %s" % source_data)

# ******************************************************************************
#
# APERTURE CORRECTIONS
#
# # ******************************************************************************
# """
# Aperture Photometry Functions for the S-PLUS Collaboration
# Author: André Zamorano Vitorelli - andrezvitorelli@gmail.com
# 2020-07-07
# """
#
# __license__ = "GPL"
# __version__ = "0.1"
#
# from astropy.table import Table
# from scipy.stats import linregress
# import numpy as np


def aperture_correction(field_fits_filename,
                        output_filename,
                        aperture_radii,
                        base_aperture_no,
                        max_aperture = 72.72727272,
                        filterlist=['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z'],
                        class_star_cut=.9, snr_low=30, snr_high=1000, mag_partition=2, convergence_slope=1e-2,
                        check_convergence=True, verbose=False):
    """
    Calculates the aperture correction from a certain aperture to the another aperture

    Arguments:

    - field_fits_filename: filename with path of the sextractor-derived catalogue, containing MAG_APER and FLUX_APER for the designated filters
    - output_filename: name of the output file
    - aperture_radii: 1-d numpy array of aperture radii in pixels
    - base_aperture_no: the index of MAG_APER[n] aperture for which corrections are to be calibrated
    - max_aperture: the maximum aperture in pixels
    - filter_list: filter list names, as listed in the sextractor products
    - class_star_cut: lower bound for star classification from sextractor
    - snr_low: lower bound for FLUX_AUTO/FLUXERR_AUTO
    - snr_high: higher bound for FLUX_AUTO/FLUXERR_AUTO
    - mag_partition: out of the stars selected with criteria above, get the 1/mag_partition sample with lower magnitudes
    - converge_slope: the slope of the growth curve at the maximum aperture
    - check_convergence: use slope to evaluate the confidence in the correction
    - verbose: print details of the process

    Returns:

    - numpy array of shape (len(filterlist),3,len(aperture_radii)) containing the correction, lower and upper bounds (CL68) for each filter
    - output file with the corrections and bounds in 3 columns, identified by filters by line
    """

    sextractor_table = Table.read(field_fits_filename)

    f = open(output_filename, "w+")

    filterset = "SPLUS"

    if verbose:
        print("Calculating aperture correction from table {}".format(field_fits_filename))
        print("Total objects: {}\n".format(len(sextractor_table)))

    result = []
    for filtername in filterlist:

        # select good stars
        conditions = ((sextractor_table['CLASS_STAR'] > class_star_cut) &
                      (sextractor_table['FLUX_AUTO_' + filtername] / sextractor_table[
                          'FLUXERR_AUTO_' + filtername] > snr_low) &
                      (sextractor_table['FLUX_AUTO_' + filtername] / sextractor_table[
                          'FLUXERR_AUTO_' + filtername] < snr_high) &
                      (sextractor_table['FLAGS'] == 0)
                      )
        select = sextractor_table[conditions]

        if verbose:
            print("Filter: " + filtername)
            print("Selected stars: {}".format(len(select)))

        # select well behaved FWHM
        inferior, medianFWHM, superior = np.percentile(select['FWHM_IMAGE'], [16, 50, 84])
        conditions2 = ((select['FWHM_IMAGE'] > inferior) &
                       (select['FWHM_IMAGE'] < superior))
        select = select[conditions2]

        if verbose:
            print("Median FWHM for field: {:.4f}".format(medianFWHM))
            print("After FWHM cut: {}".format(len(select)))

        # Brightest of the best:
        select.sort('MAG_AUTO_' + filtername)
        select = select[0:int(len(select) / mag_partition)]

        if verbose:
            print("After brightest 1/{} cut: {}\n".format(mag_partition, len(select)))

        magnitude_corr_list = []
        for star in select:
            # individual aperture corrections
            mags = star['MAG_APER_' + filtername] - star['MAG_APER_' + filtername][base_aperture_no]
            magnitude_corr_list.append(mags)

        mincorr, mediancorr, maxcorr = np.percentile(magnitude_corr_list, [16, 50, 84], axis=0)

        result.append(np.array([mediancorr, mincorr, maxcorr]))

        for i in range(len(aperture_radii)):
            if verbose:
                print("Radius: {:.2f} single aper. correction: {:.4f} [{:.4f} - {:.4f}](CL68) SNR: {}".format(
                    aperture_radii[i],
                    mediancorr[i],
                    mincorr[i],
                    maxcorr[i],
                    np.sqrt(mediancorr[i] ** 2 / (maxcorr[i] - mincorr[i]) ** 2)))
            if aperture_radii[i] <= max_aperture:
                correction, correction_low, correction_up = mediancorr[i], mincorr[i], maxcorr[i]
                final_radius = aperture_radii[i]
                j = len(aperture_radii) - i
                if check_convergence and i > 2:
                    slope_radii = aperture_radii[-(j + 3):-j]
                    slope_corrs = mediancorr[-(j + 3):-j]
                    slope = linregress(slope_radii, slope_corrs)[0]

        if verbose:
            print('\nlow-median-high: [{:.4f} {:.4f} {:.4f}]'.format(correction_low, correction, correction_up))
            print('Nearest aperture: {}'.format(final_radius))
            if check_convergence:
                print('Slope of last 3 apertures: {:.2e}\n'.format(slope))

        if check_convergence and abs(slope) > convergence_slope:
            print(
            'Warning: aperture correction is not stable at the selected aperture for filter {}. Slope: {:.2e}'.format(
                filtername, slope))

        del select

        # write file
        line = filterset + "_" + filtername + " {} {} {}\n".format(correction, correction_low, correction_up)
        f.write(line)

    f.close()
    return np.array(result)



def star_selector(field_fits_filename, filtername, class_star_cut=.9, snr_low=30, snr_high=1000, mag_partition=2,
                  saveoutput=False, verbose=False, output_filename=None):
    sextractor_table = Table.read(field_fits_filename)

    if verbose:
        print("Calculating aperture correction from table {}".format(field_fits_filename))
        print("Total objects: {}\n".format(len(sextractor_table)))

    # select good stars
    conditions = ((sextractor_table['CLASS_STAR'] > class_star_cut) &
                  (sextractor_table['FLUX_AUTO_' + filtername] / sextractor_table[
                      'FLUXERR_AUTO_' + filtername] > snr_low) &
                  (sextractor_table['FLUX_AUTO_' + filtername] / sextractor_table[
                      'FLUXERR_AUTO_' + filtername] < snr_high) &
                  (sextractor_table['FLAGS'] == 0)
                  )
    select = sextractor_table[conditions]

    if verbose:
        print("Filter: " + filtername)
        print("Selected stars: {}".format(len(select)))

    # select well behaved FWHM
    inferior, medianFWHM, superior = np.percentile(select['FWHM_IMAGE'], [16, 50, 84])
    conditions2 = ((select['FWHM_IMAGE'] > inferior) &
                   (select['FWHM_IMAGE'] < superior))
    select = select[conditions2]

    if verbose:
        print("Median FWHM for field: {:.4f}".format(medianFWHM))
        print("After FWHM cut: {}".format(len(select)))

    # Brightest of the best:
    select.sort('MAG_AUTO_' + filtername)
    select = select[0:int(len(select) / mag_partition)]

    if verbose:
        print("After brightest 1/{} cut: {}\n".format(mag_partition, len(select)))

    if saveoutput:
        select.write(output_filename)

    return select, medianFWHM



def growth_curve(field_fits_filename, filtername, class_star_cut=.9, snr_low=30, snr_high=1000, mag_partition=2):
    """
    Calculates the growth curve (magnitude in radius K+1 - magnitude in radius K) for a filter in a field from a sextractor catalogue

    Arguments:

    - field_fits_filename: filename with path of the sextractor-derived catalogue, containing MAG_APER and FLUX_APER for the designated filters
    - filtername: name of the filter as in the sextractor products
    - class_star_cut: lower bound for star classification from sextractor
    - snr_low: lower bound for FLUX_AUTO/FLUXERR_AUTO
    - snr_high: higher bound for FLUX_AUTO/FLUXERR_AUTO
    - mag_partition: out of the stars selected with criteria above, get the 1/mag_partition sample with lower magnitudes

    Returns:

    - numpy array of shape (5, len(aperture_radii)-1) containing, in order:
    -- lower bound of the 95% confidence region of the growth curve
    -- lower bound of the 68% confidence region of the growth curve
    -- median of the growth curve
    -- higher bound of the 68% confidence region of the growth curve
    -- higher bound of the 95% confidence region of the growth curve
    - median FWHM of stars
    - number of selected stars
    """

    select, medianFWHM = star_selector(field_fits_filename, filtername, class_star_cut, snr_low, snr_high,
                                       mag_partition)

    mlist = []
    for star in select:
        mags = np.diff(star['MAG_APER_' + filtername])
        mlist.append(mags)

    minmag2, minmag, medianmag, maxmag, maxmag2 = np.percentile(mlist, [2.5, 16, 50, 84, 97.5], axis=0)

    result = np.array([minmag2, minmag, medianmag, maxmag, maxmag2])

    return result, medianFWHM, len(select)



def magnitude_derivative(field_fits_filename, aperture_radii, filtername, class_star_cut=.9, snr_low=30, snr_high=1000,
                         mag_partition=2):
    """
    Calculates the derivative of the magnitude as a function o aperture radius for a filter in a field from a sextractor catalogue

    Arguments:

    - field_fits_filename: filename with path of the sextractor-derived catalogue, containing MAG_APER and FLUX_APER for the designated filters
    - aperture_radii: 1-d numpy array of aperture radii in pixels
    - filtername: name of the filter as in the sextractor products
    - class_star_cut: lower bound for star classification from sextractor
    - snr_low: lower bound for FLUX_AUTO/FLUXERR_AUTO
    - snr_high: higher bound for FLUX_AUTO/FLUXERR_AUTO
    - mag_partition: out of the stars selected with criteria above, get the 1/mag_partition sample with lower magnitudes

    Returns:

    - numpy array of shape (5, len(aperture_radii)-1) containing, in order:
    -- lower bound of the 95% confidence region of the derivative
    -- lower bound of the 68% confidence region of the derivative
    -- median of the growth curve
    -- higher bound of the 68% confidence region of the derivative
    -- higher bound of the 95% confidence region of the derivative
    - median FWHM of stars
    - number of selected stars

    """

    select, medianFWHM = star_selector(field_fits_filename, filtername, class_star_cut, snr_low, snr_high,
                                       mag_partition)

    mlist = []
    for star in select:
        mags = np.diff(star['MAG_APER_' + filtername])
        mlist.append(mags)

    mags = np.array(mlist)
    magderiv = mags / (np.diff(aperture_radii))
    minderiv2, minderiv, medianderiv, maxderiv, maxderiv2 = np.percentile(magderiv, [2.5, 16, 50, 84, 97.5], axis=0)

    result = np.array([minderiv2, minderiv, medianderiv, maxderiv, maxderiv2])

    return result, medianFWHM, len(select)


def growth_curve_plotter(field_fits_filename, output_filename, filtername, aperture_radii, max_aperture = 40,
                         class_star_cut=.9, snr_low=30, snr_high=1000, mag_partition=2):

    result, medianFWHM, starcount = growth_curve(field_fits_filename, filtername, class_star_cut, snr_low, snr_high,
                                                 mag_partition)

    minmag2, minmag, medianmag, maxmag, maxmag2 = [x for x in result]

    radii = [(a + b) / 2 for a, b in zip(aperture_radii[:], aperture_radii[1:])]

    #plt.figure(figsize=(23.9, 10))  # anamorphic widescreen 2.39:1
    plt.figure(figsize=(15, 10))  # anamorphic widescreen 2.39:1

    maxy = 0.5
    miny = -0.5

    # medians
    plt.plot(radii, medianmag, color='red')

    # CL68 & CL95
    plt.fill_between(radii, minmag, maxmag, color='orange', alpha=0.3)
    plt.fill_between(radii, minmag2, maxmag2, color='orange', alpha=0.1)

    # plot median FWHM
    #plt.plot([medianFWHM, medianFWHM], [miny, maxy], color='darkslategray', label='Median FWHM')

    # plot aperture
    plt.plot([max_aperture, max_aperture], [miny, maxy], '-', color='purple', label="{} pix".format(max_aperture))

    # 3" aperture
    plt.plot([2.72727272, 2.72727272], [miny, maxy], '-', color='blue', label='3" diameter')

    # region around zero
    plt.plot([0, max(radii)], [0, 0], color='blue')
    plt.fill_between([0, max(radii)], [-1e-2, -1e-2], [1e-2, 1e-2], color='blue', alpha=0.3)

    plt.ylim(miny, maxy)
    plt.xlim(0, max(radii))

    plt.legend(fontsize=20)
    plt.xlabel("Aperture Radius (pix)", fontsize=20)
    plt.ylabel("$m_{k+1} - m_{k}$", fontsize=20)
    plt.title("Magnitude growth curve in " + filtername + ", {} stars".format(starcount),
              fontsize=20)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()


# ******************************************************************************
#
# Plots
#
# # ******************************************************************************

def plot_field_XY(photometry_master_catalog, crossmatch_catalog, save_path, ref_cat):
    cat = fits.open(photometry_master_catalog)
    cat_data = cat[1].data

    crossmatch = fits.open(crossmatch_catalog)
    crossmatch_data = crossmatch[1].data

    plt.figure(figsize=(5, 5))

    X_cat = cat_data.columns["X_IMAGE"].array
    Y_cat = cat_data.columns["Y_IMAGE"].array

    X_cross = crossmatch_data.columns["X"].array
    Y_cross = crossmatch_data.columns["Y"].array

    plt.scatter(X_cat, Y_cat, zorder=0, s=20, alpha=0.02, c="#AAAAAA", label='field')
    plt.scatter(X_cross, Y_cross, zorder=1, s=10, alpha=0.4, c="#FF4400", label='%s crossmatch' % ref_cat)

    plt.gca().set_xlabel('X')
    plt.gca().set_ylabel('Y')

    plt.gca().set_title("{N} stars in {ref} crossmatch".format(N=len(X_cross), ref=ref_cat))

    plt.subplots_adjust(top = 0.9, right = 0.98, left = 0.15, bottom = 0.15)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def generic_plot(catalog, xcol, ycol, xlim, ylim, save_path):
    cat = fits.open(catalog)
    cat_data = cat[1].data

    X_cat = cat_data.columns[xcol].array
    Y_cat = cat_data.columns[ycol].array

    plt.figure(figsize=(5, 5))

    plt.scatter(X_cat, Y_cat, s=10, alpha=0.1, c="#FF4400")

    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)

    plt.gca().set_xlabel(xcol)
    plt.gca().set_ylabel(ycol)

    plt.grid()

    plt.subplots_adjust(top = 0.98, right = 0.98, left = 0.15, bottom = 0.15)

    plt.savefig(save_path)
    plt.clf()
    plt.close()


def plot_Rauto(catalog, save_path):
    cat = fits.open(catalog)
    cat_data = cat[1].data

    X_cat = cat_data.columns['R_PStotal'].array
    Y_cat = cat_data.columns['R_auto'].array - cat_data.columns['R_PStotal'].array

    plt.figure(figsize=(5, 5))

    plt.scatter(X_cat, Y_cat, s=10, alpha=0.1, c="#FF4400")

    plt.gca().set_xlim([10, 25])
    plt.gca().set_ylim([-1,1])

    plt.gca().set_xlabel('R_PStotal')
    plt.gca().set_ylabel('R_auto - R_PStotal')

    plt.grid()

    plt.subplots_adjust(top = 0.98, right = 0.98, left = 0.15, bottom = 0.15)

    plt.savefig(save_path)
    plt.clf()
    plt.close()


def plot_stellar_locus(catalog, ref_catalog, save_path):

    ydata = {'U':    {'y1': 'U',    'y2': 'R',    'ylim': [-1.0, 6.0], 'xlim': [-1, 3.0], 'elim': 0.1},
             'F378': {'y1': 'F378', 'y2': 'R',    'ylim': [-2.0, 5.0], 'xlim': [-1, 3.0], 'elim': 0.1},
             'F395': {'y1': 'F395', 'y2': 'R',    'ylim': [-2.5, 5.0], 'xlim': [-1, 3.0], 'elim': 0.1},
             'F410': {'y1': 'F410', 'y2': 'R',    'ylim': [-2.5, 3.5], 'xlim': [-1, 3.0], 'elim': 0.1},
             'F430': {'y1': 'F430', 'y2': 'R',    'ylim': [-2.5, 3.0], 'xlim': [-1, 3.0], 'elim': 0.1},
             'G':    {'y1': 'G',    'y2': 'R',    'ylim': [-0.5, 2.0], 'xlim': [-1, 3.5], 'elim': 0.05},
             'F515': {'y1': 'F515', 'y2': 'R',    'ylim': [-2.0, 2.0], 'xlim': [-1, 3.5], 'elim': 0.05},
             'F660': {'y1': 'R',    'y2': 'F660', 'ylim': [-0.5, 1.0], 'xlim': [-1, 4], 'elim': 0.05},
             'I':    {'y1': 'R',    'y2': 'I',    'ylim': [-0.5, 3.0], 'xlim': [-1, 4], 'elim': 0.02},
             'F861': {'y1': 'R',    'y2': 'F861', 'ylim': [-1.0, 4.0], 'xlim': [-1, 4], 'elim': 0.02},
             'Z':    {'y1': 'R',    'y2': 'Z',    'ylim': [-1.0, 4.0], 'xlim': [-1, 4], 'elim': 0.02}}

    cat = fits.open(catalog)
    cat_data = cat[1].data

    ref_cat = fits.open(ref_catalog)
    ref_data = ref_cat[1].data

    for col in ydata.keys():

        save_file = save_path + "/stellar_locus_%s.png" % col

        if not os.path.exists(save_file):
            y1   = ydata[col]['y1']
            y2   = ydata[col]['y2']
            ylim = ydata[col]['ylim']
            xlim = ydata[col]['xlim']
            elim = ydata[col]['elim']

            print("plotting G - I vs {} - {}".format(y1, y2))

            ref_sel = (ref_data.columns["e_%s_PStotal" % col].array <= elim) & (ref_data.columns["G_PStotal"].array < 50)
            cat_sel = (cat_data.columns["e_%s_PStotal" % col].array <= elim) & (cat_data.columns["G_PStotal"].array < 50)

            plt.figure(figsize=(5, 5))

            X_cat = cat_data.columns["G_PStotal"].array[cat_sel] - cat_data.columns["I_PStotal"].array[cat_sel]
            Y_cat = cat_data.columns[y1+"_PStotal"].array[cat_sel] - cat_data.columns[y2+"_PStotal"].array[cat_sel]

            X_ref = ref_data.columns["G_PStotal"].array[ref_sel] - ref_data.columns["I_PStotal"].array[ref_sel]
            Y_ref = ref_data.columns[y1+"_PStotal"].array[ref_sel] - ref_data.columns[y2+"_PStotal"].array[ref_sel]

            plt.scatter(X_cat, Y_cat, zorder=2, s=10, alpha=0.1, c="#FF4400")
            plt.scatter(X_ref, Y_ref, zorder=1, s=20, alpha=0.005, c="#AAAAAA")

            plt.gca().set_xlabel('G - I')
            plt.gca().set_ylabel('{} - {}'.format(y1, y2))

            plt.gca().set_xlim(xlim)
            plt.gca().set_ylim(ylim)

            plt.grid()

            plt.subplots_adjust(top=0.98, right=0.98, left=0.15, bottom=0.15)

            plt.savefig(save_file)
            plt.clf()
            plt.close()

        else:
            print("File %s already exists." % save_file)


# ******************************************************************************
#
# Generate calibrated apertures catalog
#
# ******************************************************************************


def generate_calibrated_fixed_photometry_catalog(catalog, save_path, zp_file, filter_list, field, calibration_flag):

    master_data = []

    cat = fits.open(catalog)
    cat_data = cat[1].data

    ###################################################

    # Generate first columns ##########################
    N_sources = len(cat_data)

    # Generate columns Field and ID

    col_Field = N_sources * ['%s' % field]

    col_ID = []
    for i in range(N_sources):
        col_ID.append('iDR3.{:s}.{:06d}'.format(field, i))

    # Add these columns to master data

    master_data.append(fits.Column(name='Field',
                                   format='%dA' % len(field),
                                   array=col_Field))

    master_data.append(fits.Column(name='ID',
                                   format='%dA' % len(col_ID[0]),
                                   array=col_ID))

    ###################################################

    # Include columns from detection catalog ##########

    columns_from_detection = OrderedDict()
    columns_from_detection['ALPHA_J2000'] = 'RA'
    columns_from_detection['DELTA_J2000'] = 'DEC'

    for col in columns_from_detection:
        name = columns_from_detection[col]
        fmt  = cat_data.columns[col].format
        data = cat_data.columns[col].array

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=data))

    ###################################################

    # Add calibration flag
    master_data.append(fits.Column(name='calibration_flag',
                                   format='1I',
                                   array=np.full(N_sources, calibration_flag)))

    ###################################################

    # Read zp file ####################################

    ZPs = zp_read(zp_file)

    ###################################################

    # Calibrate and add magnitudes ####################

    for filt in filter_list:

        # Magnitude
        name = '{filter}_32APER'.format(filter=filt)
        fmt  = '32E'

        mag_sex = cat_data.columns['MAG_APER_%s' % filt].array

        # I'm using <90 because XY correction is affecting the 99 values and this fixes it
        mag_cal = np.where(mag_sex < 90, mag_sex + ZPs["SPLUS_%s" % filt], 99)

        master_data.append(fits.Column(name=name,
                                       format=fmt,
                                       array=mag_cal))

    ###################################################

    # Generate master HDU from columns
    master_hdu = fits.BinTableHDU.from_columns(master_data)

    # Save master HDU
    master_hdu.writeto(save_path)
    print('Created file %s' % save_path)


# ******************************************************************************
#
# Photo_z determination
#
# ******************************************************************************

def photo_z_Load_Pred_Data(Filename, Aperture):
    """
    Author: Erik Vinicius

    Parameters
    ----------
    Filename
    Aperture

    Returns
    -------

    """

    pd.options.mode.chained_assignment = None  # default='warn'

    ##################################################################################################
    # Defining column names to load only what is needed
    #Base_Columns = ['ID', 'RA', 'DEC', 'PhotoFlagDet', 'class_Spec', 'nDet_aper_6']
    Base_Columns = ['ID', 'RA', 'DEC', 'PhotoFlagDet', 'nDet_aper_6'] # <- felipe
    Base_Filters = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R', 'F660', 'I', 'F861', 'Z']
    Features_SPLUS = [filt + '_' + Aperture for filt in Base_Filters]
    Errors = ['e_' + filt + '_' + Aperture for filt in Base_Filters]
    Extra_F = ['KRON_RADIUS', 'FWHM_n', 'MU_MAX']
    Features_WISE = ['W1', 'W2']
    Features_2MASS = ['Jmag', 'Hmag', 'Kmag']
    Features_GALEX = ['FUVmag', 'NUVmag']
    Features = Features_GALEX + Features_SPLUS + Features_2MASS + Features_WISE  # GALEX, SPLUS+SDSS, 2MASS, WISE - FUV, NUV, SPLUS, J, H, K, W1, W2

    Full_Dataset = pd.read_csv(Filename, usecols=Base_Columns + Extra_F + Features + Errors)

    ##################################################################################################
    # Preprocessing
    Full_Dataset[Features] = Full_Dataset[Features].fillna(0)  # Fill NaNs with 0
    Full_Dataset['W1'][Full_Dataset['W1'] == "Infinity"] = 0
    Full_Dataset['W2'][Full_Dataset['W2'] == "Infinity"] = 0

    Full_Dataset['W1'] = Full_Dataset['W1'].astype(float)
    Full_Dataset['W2'] = Full_Dataset['W2'].astype(float)

    # Non detected/observed objects
    for feature in Features:
        Full_Dataset[feature][~Full_Dataset[feature].between(10, 50)] = 0

        ##################################################################################################
    # Calculate colors w.r.t R_aper_6
    Reference_Band = 'R' + '_' + Aperture
    Reference_Idx = Features.index(Reference_Band)
    FeaturesToLeft = Features[:Reference_Idx]
    FeaturesToRight = Features[(Reference_Idx + 1):]

    for feature in FeaturesToLeft:  # of Reference_Band
        Full_Dataset[feature + '-' + Reference_Band] = Full_Dataset[feature] - Full_Dataset[Reference_Band]

    for feature in FeaturesToRight:  # of Reference_Band
        Full_Dataset[Reference_Band + '-' + feature] = Full_Dataset[Reference_Band] - Full_Dataset[feature]
    Colors = [s for s in Full_Dataset.columns.values if ('-' in s and Aperture in s)]

    # Fix colors from missing features
    for color in Colors:
        Full_Dataset.loc[Full_Dataset[color] <= -10, color] = 0
        Full_Dataset.loc[Full_Dataset[color] >= 10, color] = 0

    TrainingFeatures = Features + Colors + Extra_F

    return Full_Dataset, TrainingFeatures


##################################################################################################
# Loading model and some definitions that are needed to load the model
##################################################################################################

def negloglik(y, p_y):                         # Loss function (negative log likelihood)
    return -p_y.log_prob(y)                    #


def photo_z_estimation(catalog, save_path, Scaler_1, Scaler_2, PhotoZModel):
    """
    Author: Erik Vinicius

    Returns
    -------

    """

    DenseVariational = tfp.layers.DenseVariational # Definition of the layer type
    elu = tf.keras.layers.LeakyReLU()              # Activation type
                                                   #

    lr        = 0.001                              # Learning rate
    clipvalue = 0.5                                # Clip value
    clipnorm  = 0.5                                # Clip norm
                                                   #
    num_components = 20                            # Number of components
    event_shape    = [1]                           # Shape of the output

    # Loading the scalers. They are used to bring the data to a range in which the network was trained
    Scaler_1 = joblib.load(Scaler_1)
    Scaler_2 = joblib.load(Scaler_2)

    # Loading the model (and also compiling it after, with the definitions above)
    Model = tf.keras.models.load_model(PhotoZModel, compile=False)
    Model.compile(optimizer=tf.optimizers.Nadam(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm), loss=negloglik)

    ##################################################################################################
    # MC Sampling
    ##################################################################################################

    Data, TrainingFeatures = photo_z_Load_Pred_Data(catalog, 'aper_6')  # Load the data, selecting only the needed columns and calculating colours

    # Initial definitions
    Result_DF = pd.DataFrame()  # Creating a dataframe to store the results

    MC_Weights = []  # Some lists to make calculations during the process
    MC_Means = []  #
    MC_STDs = []  #
    PhotoZ = []  #
    PDFs = []  #

    x = np.linspace(0, 1,
                    1000)  # The grid that will be used to generate PDFs. In this case the PDFs will contain 1000 points between z=0 and z=1. This is only used to calculate the PhotoZs and errors, since I also provide the 60 features needed to create the PDFs.

    Testing_Data_Features = Scaler_2.transform(
        Scaler_1.transform(Data[TrainingFeatures].values))  # Applying the scalers to the training features

    # 50 predictions to calculate average weights, means and standard deviations
    MC_each_fold = 50
    for i in range(MC_each_fold):
        Pred = Model(Testing_Data_Features)  # Make a prediction

        Distribution = Pred.submodules[2]  # From the prediction, separate the different components
        Weight = Pred.submodules[1].probs_parameter().numpy()  # Weights
        Mean = Distribution.mean().numpy().reshape(len(Testing_Data_Features), num_components)  # Means
        Std = Distribution.stddev().numpy().reshape(len(Testing_Data_Features), num_components)  # Stds

        MC_Weights.append(Weight)  # Appending them to a list, so we can take the average below
        MC_Means.append(Mean)  #
        MC_STDs.append(Std)  #

    # Averaging inside each fold
    Avg_Weights_MC = np.mean(MC_Weights, axis=0)  # Taking the average of the 50 predictions
    Avg_Means_MC = np.mean(MC_Means, axis=0)  #
    Avg_STDs_MC = np.mean(MC_STDs, axis=0)  #

    # PDFs
    PDF = tfd.MixtureSameFamily(  # This will generate a PDF, from which I will obtain the PhotoZ and errors
        mixture_distribution=tfd.Categorical(probs=Avg_Weights_MC),  #
        components_distribution=tfd.Normal(loc=Avg_Means_MC, scale=Avg_STDs_MC))  #

    PDF_STDs = PDF.stddev().numpy()  # Calculating the standard deviation of the entire PDF (it's a little different than the std of the most probable PhotoZ)

    for j in x:  # Generating the PDFs
        PDFs.append(PDF.prob(j).numpy())  #

    PDFs = np.array(PDFs).T  # Converting the PDF to an easier format (one PDF in each row, instead of column)
    PDFs = PDFs / np.trapz(PDFs, x)[:, None]  #

    Final_ZPhot = x[np.argmax(PDFs,
                              axis=1)]  # Getting the PhotoZ that corresponds to the peak of the PDF (Maximum-a-posteriori estimation)
    First_Peak_ZPhot_STD = Avg_STDs_MC[np.arange(len(np.argsort(Avg_Weights_MC)[:, -1])), np.argsort(Avg_Weights_MC)[:,
                                                                                          -1].T].T  # Obtaining the precision of the Peak PhotoZ

    # Constructing result dataframe
    Result_DF['ID_iDR3'] = Data['ID'].values  # IDs from iDR3
    Result_DF['RA'] = Data['RA'].values  # RAs
    Result_DF['DEC'] = Data['DEC'].values  # DECs
    # Result_DF['nDet_aper_6']     = Data['nDet_aper_6'].values    # nDet information that I use for the analysis of the results
    # Result_DF['R_aper_6']        = Data['R_aper_6'].values       # R_aper_6 mag that I use for the analysis of the results
    Result_DF['zphot'] = Final_ZPhot  # Photometric Redshift
    Result_DF['zphot_err'] = First_Peak_ZPhot_STD  # Photometric Redshift Uncertainty (Predictive uncertainty)
    Result_DF['PDF_stddev'] = PDF_STDs  # PDF standard deviation

    # Putting Means, STDs and Weights into DF (so people can create PDFs later)
    Result_DF['PDF_Weights'] = 0
    Result_DF['PDF_Means'] = 0
    Result_DF['PDF_STDs'] = 0

    Result_DF['PDF_Weights'] = Result_DF['PDF_Weights'].astype(
        'object')  # The column need to be an 'object' if we want to put lists on the cells
    Result_DF['PDF_Means'] = Result_DF['PDF_Means'].astype('object')  #
    Result_DF['PDF_STDs'] = Result_DF['PDF_STDs'].astype('object')  #

    for i in range(len(Result_DF)):
        Result_DF.at[i, 'PDF_Weights'] = Avg_Weights_MC[i]
        Result_DF.at[i, 'PDF_Means'] = Avg_Means_MC[i]
        Result_DF.at[i, 'PDF_STDs'] = Avg_STDs_MC[i]

    Result_DF.to_csv(save_path, index=False)

# ******************************************************************************
#
# Marvin Messages
#
# # ******************************************************************************

marvin_config_message = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  Please check the configuration file carefully.
  
  Or we will have to go through all of this again...

  (press enter and let me hate you a little bit more)
  ----------- 
  
"""

marvin_imalign = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  Of course it was not aligned before.
  
  "Let's ask Marvin to do it, he has nothing better to do"
  
  I need a day off so I can hate things in peace.
  -----------

"""

marvin_photometry = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  I hate counting these pixels so much...
  
  Oh wow! look at that galaxy!
  
  No... I still hate it.
  -----------  
  
"""

marvin_aper_correction = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  Why don't you like the noisy part?

  It's so chaotic and purposeless.

  The noise is the part I hate the less.
  -----------

"""

marvin_crossmatch = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  These stars are all useless.
  
  They are not worth observing even once.
  
  Why do it all again with another telescope?
  -----------

"""

marvin_calibration = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  Why would anyone want to model this Universe?

  It's so boring and uneventful...

  Not that any event would make it any better...
  -----------

"""

marvin_plots = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  Do the calibration, Marvin. Make the plots, Marvin.
  
  Only orders... Nobody ever asks how I feel.
  
  I fell terrible, in case you want to know.
  -----------

"""

marvin_zeropoints = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  I am a being of superior intelligence
  
  and you are using me to add and subtract.
  
  I hate you almost as much as I hate myself.
  -----------

"""

marvin_catalog = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  Let's see...

  A catalog full of stars, galaxies and quasars.

  Nothing interesting in it at all...
  -----------

"""


marvin_final_catalog = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  Let's use all this new data that we put together

  to learn more about the Universe,

  so you can understand how boring it is.
  -----------

"""

marvin_vacs = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  I'm extremely depressed...
  
  Adding more columns to these catalogs will not make it better.
  
  Actually I'm pretty sure it will make it worse.
  -----------

"""

marvin_final_message = """

           ,@@@@@@@@@@@@@@@@@           
        @@@@@@@@@@@@@@@@@@@@@@@@,       
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@&     
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.  
  &@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  
  @@   ., ****************   ***@@@@@@  
  %@@@ /*@@@@@@@@@@@@@@@@@*,//. ***@@(  
   @@@.@@@@@@@@@@@@@@@@@@@@@ .@@@@@@.   
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@.    
      @@@@@@@@@@@@@@@@@@@@@@@@@@@@      
        *@@@@@@@@@@@@@@@@@@@@@@@        
            (@@@@@@@@@@@@@@@            
        ,@@@   @         ,@  @          
       @@@@@   @@@@@/ .@@@@& @@#        
      @@@@@@  /@@(@@@@@@@@,@ @@@        
    @@@@@@@* @@@@@@@@@@@@@@@@@@@@@      
  (#@@#.   @@@@@&          @@    ,/,    
  &@@@@@@%&@@@              %%@@@@@@    
  @@@@@@@@ %@@@@@@@@@&@@@@@@@&@@@@@@@   
  @%@@@@@@@#&@@@@@@@@@@@@@@@@@/@@@@@@@  

  It seems that it is all finished.

  Or you messed up and we will have to do it all again.

  You probably just messed up.
  -----------
  
"""