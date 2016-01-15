#random
import random
random.randint(0,10)
#to repeat sequence
random.seed(10)
print([random.randint(0,10) for _ in range(5)])
random.seed(10)
# Same sequence as above.
print([random.randint(0,10) for _ in range(5)])
# SAMPLE DATA!!!!
shopping = [300, 200, 100, 600, 20]
shopping_sample = random.sample(shopping, 3)
# to get test set:
def select_random_sample(count):
    random_indices = random.sample(range(0, income.shape[0]), count)
    return income.iloc[random_indices]

# Linspace is as numpy function to produced evenly spaced numbers over a specified interval.
# Create an array with 50 values between -6 and 6 as t
t = np.linspace(-6,6,50, dtype=float)

#Categorical data: 
#to do categorical
dfc = pd.DataFrame({'a':['a','b','c','a','d','c']})
dfcc = pd.Categorical(dfc['a'])
dfcc.describe()
#to change to numbers (sometimes is better for machiche learing alg in sklearn)
dfc = pd.DataFrame({'a':['a','b','c','a','d','c']})
dfcc = pd.Categorical.from_array(dfc['a'])
dfc['a'] = dfcc.codes

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
port_stats = titanic_survival.pivot_table(index="embarked", values=["age","survived","fare"],aggfunc=np.mean)
yemen_1987=world_alcohol[yemen_1987_bool,:]


###################### PANDAS
#types of columns: loat, integer, boolean, and object types. The object type can contain string data.
print(sp500.info())
dataframe["column_name"] = dataframe["column_name"].astype(float) #to convert columns (in order to use them in ML)
# Dummies
embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
titanic_df = titanic_df.join(embark_dummies_titanic)
titanic_df.drop(['Embarked'], axis=1,inplace=True)

train[["value"]] #dataframe
train_v2=train_v1.copy()
train["value"] #series

pandas_dat=pandas.read_csv("file.csv")
auto = pandas.read_table(auto_file, delim_whitespace=True, names=names) # to read .txt delimited by unknown number of spaces. Adding column names (names list)
pandas_dat.index #gives me the rows back (not values, the names)
pandas_dat
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
#rename columns
data.rename(columns={'gdp':'log(gdp)'}, inplace=True)#accepts dictionary to rename {old:new}
df.columns = ['a', 'b'] # this will rename all columns
#normalizing
normalized_vitamin_c=food_info["Vit_C_(mg)"]/food_info["Vit_C_(mg)"].max()
#function sum()
row_total = food_info[column_list].sum(axis=1)
column_total = food_info[column_list].sum(axis=0)
#is null
all_ages.isnull().sum() # total nulls
age_null = pd.isnull(titanic_survival["age"])
age=titanic_survival["age"][age_null==False]
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True) # substitute nan for a value
new_titanic_survival = titanic_survival.dropna(axis=1) #drop NA/null values. (axis=1 drops columns. otherwise I drop rows)

correct_mean_age=age.sum()/len(age)
correct_mean_age = titanic_survival["age"].mean() # this one compuntes mean (without using null values)
#pivot table
passenger_survival = titanic_survival.pivot_table(index="pclass", values="survived", aggfunc=np.mean)
#more complex pivot table
port_stats = titanic_survival.pivot_table(index="embarked", values=["age","survived","fare"],aggfunc=np.mean)
#add a column
data_frame["new column"]=data_frame["other_column"]+2
#delete a column
del auto["car_name"]
#drop columns
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
#drop NA/null values. (axis=1 drops columns. otherwise I drop rows)
new_titanic_survival = titanic_survival.dropna(axis=1) #inplace=True so I don't have to assign it again
#drop rows with one of these columns with null
titanic_reindexed = titanic_survival.dropna(subset=["age","boat"])
titanic_reindexed = titanic_reindexed.reset_index(drop=True) # indexes are maintained. This will reset them.
# In[18]: how many NULLs per column
total_na = all_ages.isnull().sum()
#iter rows:
for row in matches_all.iterrows():
    #index, Series) pairs.
    break
#apply. Applies function to colomn or row (axis=1)
def age_status (row):
    if row["age"]<18:
        return "minor"
    elif row["age"]>=18:
        return "adult"
    else:
        return "unknown"
age_labels = titanic_survival.apply(age_status,axis=1)
# to apply a funcition and give it values a I have to use lambda:
df.apply(lambda row: predict(tree, row), axis=1), axis=1) # in this case precict nees to attributes (tree and row)
#row will always be passes automatically but to pass tree I can only use do it with lambda.
#how many values of each
all_ages['Major_category'].value_counts()
#get unique values of a column
all_ages['Major_category'].value_counts().index #or
list(pd.unique(recent_grads["Major_category"].values.ravel()))
# are any values in a list in a column? 
df["column1"].isin([1,2,3,4])
#find index of first min
lowest_income_county = income["county"][income["median_income"].idxmin()]
#Booleans with data_frames: http://stackoverflow.com/questions/21415661/logic-operator-for-boolean-indexing-in-pandas
TP = sum(((pred == 1) & (credit["paid"] == 1)))

####### different ways todo same things :)
recent_grads.pivot_table(index=['Major_category'], values=['Median'],aggfunc=np.mean)
recent_grads.groupby(['Major_category'])['Median'].mean()  #sql like
#######
recent_grads["Major"][recent_grads["Median"]>60000]
recent_grads.loc[recent_grads.Median > 60000, "Major"]
#select stuff
liga[liga["AwayTeam"].isin(["Real Madrid","Barcelona"])]
liga[(liga["AwayTeam"]=="Barcelona") & (liga["HomeTeam"]=="Real Madrid")]
