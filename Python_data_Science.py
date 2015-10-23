#numpy, getting started. Matrix and vectors
import numpy
#open csv, set all values as strings (U75) and skip header
nfl = numpy.genfromtxt(f, delimiter=",", dtype="U75", skip_header=1) #array object numpy
row_four=nfl[3,:] #array object numpy
some_columns_row_four=row_four[2:4] #array object numpy
#size of matrix or array
print (nfl.shape)
print (row_four.shape)
#convert columns 
column_four.astype(float)
#sum values
total_sum_column_four=column_four.sum()

#get values iqual to .....
#Example1
beer = world_alcohol[:,3] == "Beer" # get boolean for that column
print world_alcohol[:,3][beer] #mask boolean to the column and get only the values with masked as TRUE
#Example2
types = world_alcohol[:,3][0:10]
beer_boolean = types == "Beer"
print(types[beer_boolean])
#Example3
beer = world_alcohol[:,3] == "Beer"
print world_alcohol[beer,:]
#Example4
yemen_1987_bool=(world_alcohol[:,0]=="1987") & (world_alcohol[:,2]=="Yemen")
yemen_1987=world_alcohol[yemen_1987_bool,:]
