#====================================================================================================================#

# For general purposes
import numpy as np
import os, sys

# To handle .fits files and create tables
from astropy.io import fits
from astropy.table import Table, vstack

# To interpolate the filters
from scipy.interpolate import interp1d

# To add the extinction in the flux
from dust_laws import CCM

# To calculate the prior probability of the models
from prior import PriorCalculator

# To make the interpolation
import paintbox as pb

# Just for type hinting
from typing import Callable

#====================================================================================================================#


class SpectralModel:

    """
    This class is used as a base for generating models (from spectral libraries), and it is meant to be used as a
    parent class*, given that each spectral library has its own way of loading the wavelength and flux values from
    the files (and also obtaining the parameters from the name of the file).
    The E(B-V) values that you want the model to use need to be passed in the constructor (otherwise it will just use 0).

    * THIS CLASS WONT WORK WITHOUT A LOAD FUNCTION TO GET THE FLUXES AND WAVELENGHTS OUT OF THE MODEL FILES

    Methods:
    ------------
    self.fill(path_to_filters: str, *args: list[str]):
        Fills the data table with the stellar parameters and magnitudes of each model on the given filters.
    
    self.add_prior(reference: tuple, bins: dict, regbins: bool, **kwargs):
        Adds a prior column, to the the data table, calculated based on the given reference catalog.
    
    self.interpolate(params: list[tuple], restrinct: function):
        Iterpolates the original fluxes to get a thinner grid of models.

    self.save(path: str, fmt: str):
        Saves the data table into a file of the given format.
    
    self.remove(col: str, value):
        Removes every line in the catalog that has value in the column col.

    self.copy(model: SpectralModel):
        Loads the data that another SpectralModel object has already loaded and interpolated.

    self.resampler(x_old, y_old, x_new):
        Used for interpolating filters.
    
    self.synmag(wl, flux, filter_curve):
        Convolutes the transmission curve of the filter with the SED to get the AB magnitude (on that filter).
    """

    def __init__(self, ebvs: list = [0], order: list[str] = ["Teff", "logg", "FeH", "aFe"]):
        self.ebvs           = np.array(ebvs)
        self.save_prior     = False
        self.order          = order
        self.cols           = ["model_id", "EB_V", "Teff", "logg", "FeH", "aFe"]


    def fill(self, path_to_filters: str, *args: list[str]) -> None:

        """
        Fills the self.synthetic_data table with the magnitudes of every model on the filters given.

        Parameters:
        ------------
        path_to_filters: str
            Path to the folder containing the filter transmitance curves.
        *args: list[str]
            Any quantity of arrays with the names of the filters.
        """
        
        # Initializes the filter dictionary
        self.filters = {}

        # Gets the filter curves
        for arg in args:
            for filt in arg:
                self.filters[filt] = np.genfromtxt(path_to_filters + filt + '.dat').transpose()
                self.cols.append(filt)
        
        print()
        # Creates an empty array for the data table
        self.synthetic_data = np.empty((len(self.templates) * len(self.ebvs), 6 + len(self.filters)))

        for e, extinction in enumerate(self.ebvs):
            for i, template in enumerate(self.templates):

                # Gets the name of the model
                file = self.files[i]

                # Message to the user
                message = f"Processing \u001b[33;1m{file}\u001b[0m with E(B-V) = \u001b[38;5;204m{extinction}\u001b[0m"
                sys.stdout.write(f'\u001b[32;1m>>>\u001b[0m {message} \r')
                sys.stdout.flush()

                # Gets the wavelenghts and fluxes from the model dictionary
                wl, flux = self.models_dict[file]['lin_wl'], self.models_dict[file]['flux']

                # Calculates the flux with extinction
                av = 3.1 * extinction
                a_lambda = av * CCM(wl)
                flux_with_dust = flux * 10 ** (-0.4 * a_lambda)

                # Gets the stellar parameters of the model
                Teff, logg, FeH, aFe = template

                # Writes data of this model into the table
                index_data                          = i + e * len(self.templates)
                self.synthetic_data[index_data, 0]  = index_data
                self.synthetic_data[index_data, 1]  = extinction
                self.synthetic_data[index_data, 2]  = Teff
                self.synthetic_data[index_data, 3]  = logg
                self.synthetic_data[index_data, 4]  = FeH
                self.synthetic_data[index_data, 5]  = aFe
                self.synthetic_data[index_data, 6:] = np.array([self.synmag(wl           = wl,
                                                                            flux         = flux_with_dust,
                                                                            filter_curve = self.filters[filt])
                                                                            for filt in self.filters])


    def add_prior(self, reference: tuple, bins: dict, regbins: bool = False, **kwargs) -> None:
        
        """
        Adds a column with the values of the prior probabilities of each model
        (Used AFTER the self.fill(...) function).

        Parameters:
        ------------
        reference: tuple
            Ordered* tuple with the path to the reference, the column names of
            the parameters on the reference and the format of the file.
        bins: dict
            Ordered* dictionary with the parameter names and a list with the bins
            of that parameter to be used.
            It is important to add the last parameter with an empty list (even it
            not having the need to have bins, we need the name of the column to 
            separate and order it before calling the calculator with the templates).
        regbins: bool (optinal)
            If the bins are regular (evenly spaced), in which case the "bins"
            parameter should contain tuples of the type (N, r0, r1) instead of the
            actual bins.
        **kwargs:
            bw: str ('scott' / 'silverman') or float
                bandwidth method/factor that's going to be used on the KDEs.
            normalize_by: str
                Normalization method (for now, 'max' or 'sum').
        
        * by "ordered" we mean from the most independent parameter to the most 
          nested (if the calculation is meant to be P(q0)P(q1|q0)...P(qn|q(n-1)))
          the order should be q0 -> qn.
        """

        self.cols.append("prior")
        print()

        # (For the self.save)
        self.save_prior = True

        # Decompose the bins into two parts:
        pos         = []
        real_bins   = []
        for p in bins:
        
            # The positions (in order) of the template columns
            pos.append(self.order.index(p))
        
            # The actual bins (or tuples to the regular_bins constructor)
            real_bins.append(bins[p])

        # If the user chooses to use regular bins
        if regbins:
            
            # We initialize the PriorCalculator without passing in the bins
            self.p = PriorCalculator(**kwargs)

            # And construct them with the regular_bins method
            self.p.regular_bins(real_bins[:-1])
        
        # If not
        else:

            # Simply pass the bins to the constructor
            self.p = PriorCalculator(real_bins[:-1], **kwargs)
        
        # Reads the reference and creates the KDEs with it
        self.p.interpret(*reference, lastBound = real_bins[-1][-2:])

        # Creates a new list of tuples but with only the parameters needed and in the right order
        tmps = list(zip(*self.templates))
        params_in_order = [tmps[i] for i in pos]
        params_in_order = list(zip(*params_in_order))

        # Generates a prior column to the templates passed in (is the same for every ebv)
        prior_column = self.p.prior_column(params_in_order) * len(self.ebvs)

        # Adds the prior column to the data table
        self.synthetic_data = np.insert(self.synthetic_data, len(self.synthetic_data[0]), [prior_column], axis=1)


    def interpolate(self, params: list[tuple], restrict: Callable[[tuple], bool] = None) -> None:

        '''
        Interpolates the original model templates* to generate a thinner grid.

        Parameters:
        ------------
        params: list[tupple]
            Array of quadruples (Teff, logg, [Fe/H], [a/Fe]) to (try*) adding to the grid.
        restrict: function (optional)
            Function that receives a quadruplet of parameters and return a boolean (should use or not).

        * For now, this implementation just works for coelho14 (because of the wavelenght array).
        ** Some of the quadrupples are going to be ignored if the interpolation for them doesn't work
          (if they are out of the convex hull) .
        '''
        
        # Creates empty arrays for the important quantities
        table   = [] # Stellar parameters
        waves   = [] # Wavelenghts
        fluxes  = [] # Fluxes

        # For every file of the original models
        for i, file in enumerate(self.files):
            
            # Takes the (already saved) wavelenght and flux arrays
            wl, flux = self.models_dict[file]['lin_wl'], self.models_dict[file]['flux']
            
            # Saves the stellar parameters in a Table instance
            a = Table(np.array(self.templates[i]), names=["T", "g", "Z", "alpha"])
            
            # Append everything to the arrays
            waves.append(wl)
            fluxes.append(flux)
            table.append(a)

        # With those quantities, uses paintbox to create an interpolator
        star = pb.ParametricModel(waves[0], vstack(table), np.array(fluxes))

        # Filters the parameters with the restriction passed in
        filtered_pars = list(filter(restrict, params)) if (restrict != None) else params

        print()

        # For every requested stellar parameter quadruple
        for theta in filtered_pars:

            # Uses the interpolator to calculate the flux array
            flux = star(theta)

            # If the quadruple is not already on the templates (and the interpolation works)
            if (not theta in self.templates) and (0 not in flux):
            #if (not theta in self.templates):

                # Gives a name to the new model 
                file = "interpolated_T{}_g{}_z{}_a{}".format(*theta)

                # Message to the user
                message = f"Adding \u001b[33;1m{file}\u001b[0m to the templates"
                sys.stdout.write(f'\u001b[32;1m>>>\u001b[0m {message}\r')
                sys.stdout.flush()

                # Appends, both the quadruple to the templates and the file name to self.files
                self.templates.append(theta)
                self.files.append(file)

                # Adds the model information to the models_dict (calculating the flux with the interpolator)
                self.models_dict[file]           = {}
                self.models_dict[file]['lin_wl'] = np.array(waves[0])
                self.models_dict[file]['flux']   = flux


    def save(self, path: str, fmt: str, form: list[int] = [0, 3, 5]) -> None:

        """
        Saves the data table into a file of the given format.

        Parameters:
        ------------
        path: str
            Path to save model catalog
        fmt: str
            Format that the catalog should be saved as (e.g., 'ascii').
        form: list[int] (optional)
            Number of digits of the index, the stellar parameters (+ magnitudes)
            and priors, in this order.
        """

        # Message to the user
        message = f"Saving model catalog at \u001b[33;1m{path}\u001b[0m (\u001b[38;5;204m{fmt}\u001b[0m format)"
        print(f'\n\u001b[32;1m>>>\u001b[0m {message}\n')


        # Saves as a .cat file
        if (fmt == "ascii"):

            # Creates the file
            with open(path, 'w') as f:
                
                # Format of the index / magnitudes
                ffmmt = [f'%.{form[0]}f'] + [f'%.{form[1]}f' for _ in range(len(self.synthetic_data[0]) - 1)]

                # Writes the stellar parameters names into the header
                f.write('# model_id EB_V Teff logg FeH aFe')
                
                # Writes the filter names into the header
                for filter in self.filters:
                    f.write(' {}'.format(filter))
                
                # If self.add_prior(...) was called, adds 'prior' to the header
                if self.save_prior:
                    f.write(' prior')
                    ffmmt[-1] = f'%.{form[2]}f'
                
                # Puts the data into the file
                f.write('\n')
                np.savetxt(f, self.synthetic_data, fmt=ffmmt, delimiter = ' ')


    def copy(self, model):

        """
        Loads the data that another SpectralModel object has already loaded and interpolated.

        Parameters:
        ------------
        model: SpectralModel instance
            Model from which whit function will copy the loaded data from.
        """

        self.templates      = [theta[:] for theta in model.templates]
        self.models_dict    = model.models_dict.copy()
        self.files          = model.files[:]


    def remove(self, col: str, value, error = 0):

        C = self.cols.index(col)

        if error != 0:
            del_indexes = [i for i in range(self.synthetic_data.shape[0])
                            if abs(self.synthetic_data[i, C] - value) < error]    
        else:
            del_indexes = [i for i in range(self.synthetic_data.shape[0])
                            if self.synthetic_data[i, C] == value]

        self.synthetic_data = np.delete(self.synthetic_data, del_indexes, 0)


    def resampler(self, x_old, y_old, x_new):

        " This function is called by synmag to interpolate filters"
        
        interp = interp1d(x_old, y_old, bounds_error=False, fill_value=(0., 0.))
        y_new  = interp(x_new)
        
        return y_new


    def synmag(self, wl, flux, filter_curve):

        """
        Convolutes the filter trasmitance curve with the SED to give the AB magnitude on that filter.

        Parameters
        ------------
        wl: array-like
            Wavelengths.
        flux: array-like
            Fluxes.
        filter_curve: str or array-like
            Filter curve, can either be a 2xn array where position 0 a vector of wavelengths and position 1 is a vector
            of trasmitances or a string containing the path to a filter file.

        Returns
        ------------
        m_ab: array-like
            Synthetic AB magnitude.
        """

        # Reading filter if needed:
        if type(filter_curve) is str:
            wl_filter, transmitance = np.genfromtxt(filter_curve).transpose()
        else:
            wl_filter, transmitance = filter_curve[0], filter_curve[1]

        # Resampling filter and spectrum to 1 Angstrom intervals:
        wl_new = np.arange(np.round(wl_filter[0]) - 5, np.round(wl_filter[-1]) + 5)

        transmitance = self.resampler(wl_filter, transmitance, wl_new)
        flux = self.resampler(wl, flux, wl_new)

        # Calculating the magnitude:
        m_ab = -2.5 * np.log10(np.trapz(flux * transmitance * wl_new, dx=1) / np.trapz(transmitance / wl_new, dx=1)) - 2.41

        return m_ab


#====================================================================================================================#


class Coelho_2014(SpectralModel):

    """
    This class is a child class of Model, it uses all the original methods from it, but it has a Load function to
    load the data from the model files.

    The models to be used with this class are described on:
        Coelho P. R. T., 2014, MNRAS, 440, 1027

    Methods:
    ------------
    self.load(path: str):
        Loads the wavelenghts, fluxes and stellar parameters from the models folder passed as argument.
    """

    def load(self, path: str) -> None:

        # Looks into the directory passed for the templates
        file_list = os.listdir(path)

        # Reads a reference model to get wavelengths
        model   = fits.open(path + file_list[0])
        log_wl  = np.array([model[0].header['CRVAL1'] + i * model[0].header['CDELT1']
                            for i in range(model[0].header['NAXIS1'])])
        lin_wl  = 10**log_wl

        # Initializes all the containers
        self.models_dict = { }
        self.templates   = [ ]
        self.files       = [ ]

        print()
        for i, file in enumerate(file_list):

            # Reads the model
            model = fits.open(path + file)

            # Removes the extension from the name of the file
            file  = file.replace('.fits', '')

            # Message to the user
            message = f'Reading model \u001b[38;5;204m{i + 1}\u001b[0m of \u001b[38;5;204m{len(file_list)}\u001b[0m'
            sys.stdout.write(f'\u001b[32;1m>>>\u001b[0m {message} : \u001b[33;1m{file}\u001b[0m\r')
            sys.stdout.flush()

            # Fills the model dictionary
            self.files.append(file)
            self.models_dict[file]            = {}
            self.models_dict[file]['lin_wl']  = lin_wl
            self.models_dict[file]['flux']    = model[0].data

            # Gets the parameters from the file name
            params = file.split("_")
            
            Teff = float(params[0][1:])
            logg = float(params[1][1:])
            
            FeH_sign = params[2][0]
            FeH_sign = +1 if FeH_sign == 'p' else -1
            FeH = FeH_sign * float(params[2][1] + '.' + params[2][2])
            
            aFe_sign = params[2][3]
            aFe_sign = +1 if aFe_sign == 'p' else -1
            aFe = aFe_sign * float(params[2][4] + '.' + params[2][5])

            # Appends, to the template array, the stellar parameters of the model
            self.templates.append([Teff, logg, FeH, aFe])


#====================================================================================================================#


class Castelli_Kurucz_2003(SpectralModel):

    """
    This class is a child class of Model, it uses all the original methods from it, but it has a Load function to
    load the data from the model files.

    The models to be used with this class are described on:
        Castelli F., Kurucz R. L., 2003, in Piskunov N., Weiss W. W., Gray
            D. F., eds, Vol. 210, Modelling of Stellar Atmospheres. p. A20
            (arXiv:astro-ph/0405087)

    Methods:
    ------------
    self.load(path: str):
        Loads the wavelenghts, fluxes and stellar parameters from the models folder passed as argument.
    """

    def load(self, path: str) -> None:

        # logg values to be used in the models
        kurucz_g = ['g00', 'g05', 'g10', 'g15',
                    'g20', 'g25', 'g30', 'g35',
                    'g40', 'g45', 'g50']

        # Looks into the directory passed for the templates
        file_list = os.listdir(path)

        # Initializes all the containers
        self.models_dict = { }
        self.templates   = [ ]
        self.files       = [ ]

        print()        
        for i, file in enumerate(file_list):

            # Reads the model
            model = fits.open(path + file)

            # Removes the extension from the name of the file
            file  = file.replace('.fits', '')

            for g in kurucz_g:

                # Message to the user
                a = i * len(kurucz_g) + 1
                b = len(file_list) * len(kurucz_g)
                message = f'Reading model \u001b[38;5;204m{a}\u001b[0m of \u001b[38;5;204m{b}\u001b[0m'
                sys.stdout.write(f'\u001b[32;1m>>>\u001b[0m {message} : \u001b[33;1m{file}_{g}\u001b[0m\r')
                sys.stdout.flush()

                if len(set(model[1].data[g])) > 1: # models without data are set to 0 for all wavelengths
                
                    # Fills the model dictionary
                    self.files.append(file+"_"+g)
                    self.models_dict[file+"_"+g] = {}
                    self.models_dict[file+"_"+g]['lin_wl'] = model[1].data['WAVELENGTH']
                    self.models_dict[file+"_"+g]['flux']   = model[1].data[g]

                    # Gets the parameters from the file name
                    Teff = int(file.split("_")[1].split('.')[0])
                    logg = int(g[-2]) + float(g[-1])/10
                    
                    FeH_sign = - 1 if (file[2] == 'm') else + 1
                    FeH  = FeH_sign * (int(file.split("_")[0][3]) + float(file.split("_")[0][4])/10)  

                    aFe = 0.4           

                    # Appends, to the template array, the stellar parameters of the model
                    self.templates.append([Teff, logg, FeH, aFe])


#====================================================================================================================#


class NGSL(SpectralModel):

    """
    This class is a child class of Model, it uses all the original methods from it, but it has a Load function to
    load the data from the model files.

    The models to be used with this class are described on:
        Gregg M. D., et al., 2006, in Koekemoer A. M., Goudfrooĳ P., Dressel L. L.,
            eds, The 2005 HST Calibration Workshop: Hubble After the Transition
            to Two-Gyro Mode. p. 209
        Heap S. R., Lindler D. J., 2007, in Vallenari A., Tantalo R., Portinari L.,
            Moretti A., eds, Astronomical Society of the Pacific Conference Series
            Vol. 374, From Stars to Galaxies: Building the Pieces to Build Up the
            Universe. p. 409

    Methods:
    ------------
    self.load(path: str):
        Loads the wavelenghts, fluxes and stellar parameters from the models folder passed as argument.
    """

    def load(self, path: str) -> None:

        # Looks into the directory passed for the templates
        file_list = os.listdir(path)

        # Initializes all the containers
        self.models_dict = { }
        self.templates   = [ ]
        self.files       = [ ]
        
        print()     
        for i, file in enumerate(file_list):

            # Reads the model
            model = fits.open(path+file)

            # Removes the extension from the name of the file
            file  = file.replace('.fits', '')

            # Message to the user
            message = f'Reading model \u001b[38;5;204m{i + 1}\u001b[0m of \u001b[38;5;204m{len(file_list)}\u001b[0m'
            sys.stdout.write(f'\u001b[32;1m>>>\u001b[0m {message} : \u001b[33;1m{file}\u001b[0m\r')
            sys.stdout.flush()

            # Fills the model dictionary
            self.files.append(file)
            self.models_dict[file] = {}
            self.models_dict[file]['lin_wl'] = model[1].data['WAVELENGTH']
            self.models_dict[file]['log_wl'] = np.log10(model[1].data['WAVELENGTH'])
            self.models_dict[file]['flux']   = model[1].data['FLUX']

            # Set the stellar parameters to 99 (undefined)
            Teff = 99
            logg = 99
        
            FeH  = 99
            aFe  = 99

            # Appends, to the template array, the stellar parameters of the model
            self.templates.append([Teff, logg, FeH, aFe])


#====================================================================================================================#