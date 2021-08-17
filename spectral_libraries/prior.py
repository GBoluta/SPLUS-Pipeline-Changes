#====================================================================================================================#

# For general purposes
import numpy as np

# To message the user
import sys

# To deal with the data as a dataframe
import pandas as pd

# To calculate tr priors
from scipy.stats import gaussian_kde as kde

# Just for type hinting
from typing import Callable

#====================================================================================================================#


class PriorCalculator:

	"""
    This class is used as a calculator of prior probabilities of models using KDEs, it is implemented calculating
	nested probabilities for que parameters passed in, for example, if tuples of the parameters (p0, p1, p2), are
	passed inthe priors returned are simply P(q0) * P(q1 | q0) * P(q2 | q1, q0), where each of this probabilities
	is calculated with a KDE inside bins of previous parameters, in this example, P(q0) is just the value of q0 in
	the total KDE of p0, but P(q1 | q0) is the value of q1 in the KDE calculated inside a bin p0, the same for
	P(q2 | q1, q0), it's calculated inside a bin of p0 and p1. (The normalization method is up to the user to choose)

    In the contructor, the calculator receives the bins that you want to use on the calculation and a bandwidth_factor
	(bw) to be used on the KDEs. It's optional to the user to instead of passing in the bins, constructing them via
	the method self.regular_bins(...) which, as the name sugests, creates N bins of equal size in the given	range for
	every parameter.
	
	Methods:
    ------------
    
    """
	
	def __init__(self, bins: list[tuple] = [], bw = 'scott', normalize_by: str = 'max'):
		
		self.bw = bw
		self.bins = bins
		self.normalize_by = normalize_by


	def regular_bins(self, vectors: list[tuple]) -> None:

		'''
		This is meant to be called when the choice of bins is such that the 
		limits are evenly spaced. (It constructs the bins, therefore, only use
		when the bins are not passed in the constructor).

		Parameters:
    	------------
		vectors: list
			A list of ordered tuples of the kind (N, r0, r1) where N is the
			number of bins, r0 is the lower limit and r1 is the upper limit.
			(If there is m parameters, vectors should contain m - 1 tuples).
		'''

		# For every parameter (except for the last)
		for N, r0, r1 in vectors:
			
			# Use the boundaries and N to define the separation
			step = (r1 - r0) / N

			# Constructs the bins for that parameter
			bins = [(r0 + i * step, r0 + (i + 1) * step)
					 for i in range(N)]
			
			# Adds it to the bin list
			self.bins.append(bins)


	def interpret(self, ref: str, cols: tuple[str], fmt: str, **kwargs) -> None:

		'''
		This method takes the file passed as a reference, reads it
		into a dataframe and than fills a tensor of KDE vectors for
		each bin.

		Parameters:
    	------------
		ref: str
			Path to the reference that's going to be used as a base
			for the prior calculation (as by using the density as a
			probability).
		cols: list[str]
			Name of the columns with the parameters (in order) that
			are going to be used.
		fmt: str
			Format of the file passed in the ref argument.
		'''
		
		# If it is a .cat file (ascii)
		if (fmt == 'ascii'):

			# Opens the file into a dataframe
			self.ref = pd.read_csv(ref,
								   delim_whitespace = True,
								   escapechar       = '#',
							       skipinitialspace = True)
			self.ref.columns = self.ref.columns.str.replace(' ', '')

		# Populates a tensor with all the KDEs
		self.fill_bins(cols, **kwargs)


	def indexes(self, theta: tuple) -> tuple:

		'''
		This method is used to find the indexes of the bins that a
		vector (theta) fits inside.

		Parameters:
    	------------
		theta: tuple
			Parameter vector (p0, ..., pn).

		Returns:
		------------
		indexes: tuple
			Indexes of the bin that contains theta.
		'''

		# Initializes the index list with -1s 
		indexes = [ -1 for _ in self.bins]

		# For each parameter (except for the last)
		for p in range(len(self.bins)):

			# Goes through the bins (if the bin is not found it keeps the -1)
			for i, b in enumerate(self.bins[p]):

				b0, b1 = b

				# Until it finds the bin that contains the parameter
				if (theta[p] >= b0) and (theta[p] < b1):

					# it adds the index to the list
					indexes[p] = i

					# and goes to the next parameter
					break
				
		return tuple(indexes)


	def fill_bins(self, cols: list[str], lastBound: list[float] = [-10000, 10000]) -> None:
		
		'''
		This method is used to fill the self.prior_bins tensor with KDE vectors that
		are going to be accessed when calculating the priors.

		(This is a generalization of the process of calculating KDEs inside bins of 
		previous parameters.)

		How it works:
		------------

			The cells (c[0], ..., c[n-1]) have two properties, a controlling index
			(STATE) and a list (STORAGE), they are used to construct the tensor that
			contains the KDE vectors.

			A cell is said to be FULL when it's STATE reaches the length of the bin set
			of the corresponding parameter.

			The local reference array is defined in such a way that the element i is
			a restricted version of the element in i - 1 (in practice, every new element
			is the previous but reduced to the data contained on the current bin).

			We first shall initialize cells with STATE = 0 and empty STORAGE (= []),
			one for each parameter that has bins.

			> Start a KDE vector with the kde for p0 (which is independent of bins) 

			> For every other parameter, starting from p1, with the last local reference,
			take it's intersection with the current bin of the previous parameter. Add this
			new reference to the local references and use it to calculate the KDE for
			the current parameter. Add the KDE to the vector.

			> Add the complete KDE vector to the STORAGE of the last cell.

			> For every cell, biginning from the last, if the cell is FULL, set it's
			STATE back to 0 and put it's STORAGE inside the STORAGE of the cell to it's left.

			> If the cell is not FUll yet, up it's STATE by 1;

			When cell[0] is FULL the program is over, the tensor is it's STORAGE.

		Parameters:
    	------------
		cols: list[str]
			Name of the columns with the parameters (in order) that
			are going to be used.
		'''

		# Sorts the bins (just in case)
		for b in self.bins:
			b.sort()

		# Initializes the cells with STATE = 0 adn empty STORAGE
		cells = [[0, [ ]] for _ in self.bins]

		# Do while there's still bins of p0 to cover
		while cells[0][0] < len(self.bins[0]):

			# Message to the user
			message = f"Populating bins with KDEs... indexes = \u001b[38;5;219m{[c[0] for c in cells]}\u001b[0m"
			sys.stdout.write(f'\u001b[32;1m>>>\u001b[0m {message} \r')
			sys.stdout.flush()

			# Initializes the local_refs array with the global reference in it
			local_refs  = [self.ref]

			# Calculates the KDE for p0
			K0 = kde(local_refs[-1][cols[0]], bw_method = self.bw)
			
			# Initializes an empty KDE vector
			KDEs = [K0]

			# For every parameter (that has bins) (except for p0)
			for i in range(1, len(cols)):

				# Takes the instersection between the last local reference and the current bin
				lower = local_refs[-1][cols[i - 1]] >= self.bins[i - 1][cells[i - 1][0]][0]
				upper = local_refs[-1][cols[i - 1]] <  self.bins[i - 1][cells[i - 1][0]][1]

				# And adds it to local references
				local_refs.append(local_refs[-1].loc[lower & upper])
				
				# Crates the KDE inside all the previous bins.
				bounds = lastBound if (i + 1 == len(cols)) else [self.bins[i][0][0], self.bins[i][-1][-1]]
				Ki = self.normKDE(kde(np.array(local_refs[-1][cols[i]]), bw_method = self.bw), bounds)
				
				# Appends the KDE to the vector
				KDEs.append(Ki)
		
			# Appends the vector to the STORAGE of the last cell
			cells[-1][1].append(KDEs)

			# Updates the cells
			J = -1
			while J + len(self.bins) >= 0:

				# If the current cell is FULL
				if (cells[J][0] + 1 == len(self.bins[J])):

					# If the cell is c[0] than it's all finished
					if (J + len(self.bins) == 0):
						cells[J][0] += 1
						break
					
					# If not, puts a copy of current STORAGE inside the STORAGE to the left
					cells[J - 1][1].append(cells[J][1][:])
					
					# Resets the current cell
					cells[J] = [0, [ ]]

					# Update J
					J -= 1
				
				# If the current cell is not yet Full
				else:

					# Just update it's STATE
					cells[J][0] += 1
					break

				
		# When finished, the tensor is just the STORAGE of cell[0]
		self.prior_bins = np.array(cells[0][1], dtype=object)


	def calculate(self, theta: tuple) -> float:

		'''
		For the given vector of parameters (p0, p1, ..., pn), calculates the prior
		probability (still not normalized) of that vector via the equation

			P(p) = P(p0) P(p1 | p0) P(p2 | p0, p1) ... P(pn | p0, ..., p(n-1)).

		In practical terms it means that it finds the bins that contain the vector
		and multiply the values obtained from the KDEs inside the bins.

		Parameters:
    	------------
		theta: tuple
			Vector with the parameters that we want to calculate the prior to.

		Returns:
		------------
		prior: float
			The not still normalized calculated prior.
		'''
		
		# Gets the indexes of the bins that contain theta
		indexes = self.indexes(theta)

		# If one of the indexes is -1, by our convention it means theta is out of the boudaries
		if (-1 in indexes):

			# Therefore should be assigned a value of 0 to prior
			return 0

		# Access the tensor at that indexes to get the corresponding KDE vector
		vec = self.prior_bins[indexes]

		# Initializes the prior at 1
		prior = 1

		# For every KDE in the vector
		for i, KDE in enumerate(vec):

			# Calculates the value of the parameter on the KDE and multiplies the prior by it
			prior *= KDE(theta[i])[0]

		return prior
	

	def prior_column(self, parameters: list[tuple]) -> list:

		'''
		For a given list of parameter vectors, calculates the prior for each 
		one and returns it into a list.

		Parameters:
    	------------
		parameters: list[tuple]
			List of parameter tuples for which this function calculates
			the prior to.
		
		Returns:
		------------
		normalized_column: list
			An ordered list representing a column with the normalize priors
			of the parameter tuples passed as input.
		'''

		# Creates the column with the prior For every parameter vector
		column = [self.calculate(theta) for theta in parameters]

		# Creates a normalized version of the column
		normalized_column = self.normalized(column, by = self.normalize_by)

		return normalized_column


	def normalized(self, column: list[float], by: str) -> list:
		
		'''
		This method normalizes the prior values in a given column using
		the method specified bu the user (by sum, by max, ...).

		Parameters:
    	------------
		column: list[float]
			Column containing the prior values to be normalized.
		by: str
			Method to be used in the normalization.

		Returns:
		------------
		normalized_column: list
			The same column as the input but with normalized values.
		'''

		# If the method is a normalization by maximum
		if (by == "max"):

			# The normalization factor is just the biggest value
			factor = max(column)

		# If the method is a normalization by sum
		elif (by == "sum"):

			# We should use the sum of the priors in the column as a factor
			factor = sum(column)
		
		else:
			return column

		# Returns the list with all the values divided by the calculated factor
		normalized_column = list(map(lambda x: x / factor, column))
		return normalized_column
	

	def normKDE(self, KDE: Callable[[float], float], bounds: list[float]) -> Callable[[float], float]:

		'''
		Given a gaussian_kde returns a normalized version of it.

		Parameters:
    	------------
		KDE: Callable[[float], float]
			Kernel Density Estimator to be normalized.
		bounds: list[gloat]
			KDE boundaries (to calculate the integral).

		Returns:
		------------
		nKDE: Callable[[float], float]:
			Normalized KDE.
		'''

		area = KDE.integrate_box(*bounds)
		nKDE = (lambda x: KDE(x) / area)
		return nKDE