#numpy, getting started. Matrix and vectors
import numpy
#open csv, set all values as strings (U75) and skip header
nfl = numpy.genfromtxt(f, delimiter=",", dtype="U75", skip_header=1) #array object numpy
row_four=nfl[3,:] #array object numpy
some_columns_row_four=row_four[2:4] #array object numpy
#size of matrix or array
print (nfl.shape)
print (row_four.shape)
