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


###################### PANDAS
pandas_dat=pandas.read_csv("file.csv")
pandas_dataframe.iloc[0,0]
pandas_dataframe.iloc[:,0]
pandas_dataframe.iloc[0,:] 
# I can also get columns by name (we can get columns 2 at a time, etc ...)
fiber = food_data_frame['Fiber_TD_(g)']
fiber_and_sugar = food_data_frame[['Fiber_TD_(g)', 'Sugar_Tot_(g)']]
# we can do math with columns/vectors (same length) or apply an operation to the whole column
grams_of_protein_per_calorie = food_info["Protein_(g)"] / food_info["Energ_Kcal"
protein_kilograms = food_info["Protein_(g)"] / 1000
# we can sort the entire data.frame
descending_fat = food_info.sort(["Lipid_Tot_(g)"], ascending=[False])
print(descending_fat.iloc[0,:])
#normalizing
normalized_vitamin_c=food_info["Vit_C_(mg)"]/food_info["Vit_C_(mg)"].max()
#function sum()
row_total = food_info[column_list].sum(axis=1)
column_total = food_info[column_list].sum(axis=0)
#add a column
data_frame["new column"]=data_frame["other_column"]+2
#is null
age_null = pd.isnull(titanic_survival["age"])
age=titanic_survival["age"][age_null==False]
correct_mean_age=age.sum()/len(age)
correct_mean_age = titanic_survival["age"].mean() # this one compuntes mean (without using null values)
#pivot table
passenger_survival = titanic_survival.pivot_table(index="pclass", values="survived", aggfunc=np.mean)
