# -*- coding: utf-8 -*-

# *********************************************************************
#                   S-PLUS CALIBRATION PIPELINE
#                 Script to test different models
#
#   January, 2021
#
#   by Felipe Almeida-Fernandes*
#
#   *email: felipefer42@gmail.com
#
# *********************************************************************


"""

-------------------------------------------------------------------------------
   INPUTS:
------------------------------------------------------------------------------

> needs crossmatched files to run

> configuration needs to be set in the beginning of this file

------------------------------------------------------------------------------
   FUNCTIONS:
------------------------------------------------------------------------------

create_output_paths()
fit_models_to_reference()
get_initial_zps()
create_initial_corrected_catalog()
fit_models_to_splus()
get_relative_zps()

------------------------------------------------------------------------------
   COMMENTS:
------------------------------------------------------------------------------

This script is only meant to be used to test the calibration pipeline for
different models, using the STRIPE82 iDR3/DR2 data.

------------------------------------------------------------------------------
   USAGE:
------------------------------------------------------------------------------

After setting all the configuration parameters:

$python fit_zp_for_different_models.py *field_name*

----------------
"""

import os
import sys
import utils

################################################################################
# Configuration starts here

path_to_models_file = ""

path_to_crossmatched_catalogs = ""

path_to_save_output = ""

reference_catalog = "SDSS"

reference_mag_cols = ['SDSS_U', 'SDSS_G', 'SDSS_R', 'SDSS_I', 'SDSS_Z']

splus_mag_cols = ['SPLUS_F378', 'SPLUS_F395', 'SPLUS_U',    'SPLUS_F410', 
                  'SPLUS_F430', 'SPLUS_G',    'SPLUS_F515', 'SPLUS_R', 
                  'SPLUS_F660', 'SPLUS_I',    'SPLUS_F861', 'SPLUS_Z']

zp_fitting_mag_cut = [14, 19] # Pode deixar como esta



################################################################################
# Working on configuration parameters - these shouldn't be eddited

field = sys.argv[1]

path_to_save_output_field = path_to_save_output  + "/%s" % field
calibration_path_initial  = path_to_save_output_field + '/initial_fitting'
calibration_path_relative = path_to_save_output_field + '/relative_fitting'

################################################################################
# Script starts here

# ***************************************************
#    Create paths
# ***************************************************

def create_output_paths():
    print('')
    print('*********** Generating output paths ***************')
    print('')
    
    utils.makeroot(path_to_save_output_field)
    
    utils.makeroot(calibration_path_initial)
    
    utils.makeroot(calibration_path_relative)


create_output_paths()


# ******************************************************************************
#    Initial calibration
# ******************************************************************************

# ***************************************************
#    Fit models to reference catalogs
# ***************************************************

def fit_models_to_reference():
    
    print('')
    print('*********** Fitting models to reference ***************')
    print('')
    
    
    models_file = path_to_models_file
    data_file   = path_to_crossmatched_catalogs + "/{field}_crossmatch_splus_{ref}.catalog".format(field = field, ref = reference_catalog)
    save_file   = calibration_path_initial + "/reference_fit_models.cat"
    
    ref_mag_cols  = reference_mag_cols
    pred_mag_cols = splus_mag_cols
    
    if not os.path.exists(save_file):
        utils.get_model_mags_v2(models_file   = models_file,
                                data_file     = data_file,
                                save_file     = save_file,
                                ref_mag_cols  = ref_mag_cols,
                                pred_mag_cols = pred_mag_cols)
    
    else:
        print("Models already fitted to reference.")


fit_models_to_reference()



# ***************************************************
#    Get initial zps
# ***************************************************

def get_initial_zps():

    print('')
    print('*********** Getting inital ZPs ***************')
    print('')

    data_file = calibration_path_initial + "/reference_fit_models.cat"
    save_file = calibration_path_initial + "/%s_initial_zps.cat" % field
    mag_cols  = splus_mag_cols

    if not os.path.exists(save_file):
        utils.obtain_ZPs(data_file = data_file,
                         save_file = save_file,
                         mag_cols  = mag_cols,
                         mag_cut   = zp_fitting_mag_cut)
    else:
        print("initial ZPs file already created.")


get_initial_zps()


# ***************************************************
#    Create initial corrected catalog
# ***************************************************

def create_initial_corrected_catalog():
    
    print('')
    print('******** Creating Initial corrected catalog **********')
    print('')
    
    data_file  = path_to_crossmatched_catalogs + "/{field}_crossmatch_splus_{ref}.catalog".format(field = field, ref = reference_catalog)
    
    save_file  = calibration_path_initial + "/initial_corrected_catalog.cat"
    
    zp_file    = calibration_path_initial + "/%s_initial_zps.cat" % field
    
    model_file = calibration_path_initial + "/reference_fit_models.cat"
    
    if not os.path.exists(save_file):
        utils.apply_ZPs(data_file = data_file,
                        save_file = save_file,
                        zp_file   = zp_file,
                        model_file=model_file,
                        mag_cols=splus_mag_cols)
    
    else:
        print("Initial corrected catalog already created")


create_initial_corrected_catalog()


# ***************************************************
#    Create initial corrected catalog
# ***************************************************

def create_initial_corrected_catalog():
    
    print('')
    print('******** Creating Initial corrected catalog **********')
    print('')
    
    data_file  = path_to_crossmatched_catalogs + "/{field}_crossmatch_splus_{ref}.catalog".format(field = field, ref = reference_catalog)
    
    save_file  = calibration_path_initial + "/initial_corrected_catalog.cat"
    
    zp_file    = calibration_path_initial + "/%s_initial_zps.cat" % field
    
    model_file = calibration_path_initial + "/reference_fit_models.cat"
    
    if not os.path.exists(save_file):
        utils.apply_ZPs(data_file = data_file,
                        save_file = save_file,
                        zp_file   = zp_file,
                        model_file=model_file,
                        mag_cols=splus_mag_cols)
    
    else:
        print("Initial corrected catalog already created")


create_initial_corrected_catalog()


# ******************************************************************************
#    Relative calibration
# ******************************************************************************


# ***************************************************
#    Fit models to splus corrected magnitudes
# ***************************************************

def fit_models_to_splus():
    
    print('')
    print('*********** Fitting models to S-PLUS ***************')
    print('')
    
    
    models_file = path_to_models_file
    data_file   = calibration_path_initial  + "/initial_corrected_catalog.cat"
    save_file   = calibration_path_relative + "/splus_fit_models.cat"
    
    if not os.path.exists(save_file):
        utils.get_model_mags_v2(models_file   = models_file,
                                data_file     = data_file,
                                save_file     = save_file,
                                ref_mag_cols  = splus_mag_cols)
    
    else:
        print("Models already fitted to splus.")


fit_models_to_splus()


# ***************************************************
#    Get relative zps
# ***************************************************

def get_relative_zps():
    
    print('')
    print('*********** Getting relative ZPs ***************')
    print('')
    
    data_file = calibration_path_relative + "/splus_fit_models.cat"
    save_file = calibration_path_relative + "/%s_relative_zps.cat" % field
    mag_cols  = splus_mag_cols
    
    if not os.path.exists(save_file):
        utils.obtain_ZPs(data_file = data_file,
                         save_file = save_file,
                         mag_cols  = mag_cols,
                         mag_cut   = zp_fitting_mag_cut)
    else:
        print("reference ZPs file already created.")


get_relative_zps()






