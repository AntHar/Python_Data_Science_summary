import matplotlib.pyplot as plt
#create plot
plt.scatter(month, temperature)
plt.show()
# line chart. Remember to sort x values first!
forest_fires = forest_fires.sort(["temp"])
plt.plot(forest_fires["temp"], forest_fires["area"])
plt.xlabel("Wind speed when fire started")
plt.ylabel("Area consumed by fire")
plt.title("Wind speed vs fire area")
plt.show()
#bar char. this is good but I usually use data_frame.plot(kind='bar')
plt.bar(y_index, area_by_y)
plt.show()
#--------------this one better:
mask = recent_grads.pivot_table(index="Major_category",values=["Median","P25th","P75th"],aggfunc=np.mean).sort("Median")
mask.plot(kind='bar')
#change styles
print(plt.style.available)
plt.style.use('ggplot')
#histograms. Pandas function built on top of matplotlib (we need to import it)
#check all options in http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html#pandas-dataframe-hist
columns = ['Median','Sample_size']
recent_grads.hist(column=columns,layout=(2,1))
plt.axvline(recent_grads["Median"].mean(),color="r") # draws a vertical line where the mean is
plt.axvline(np.median(recent_grads["Median"]),color="b") # draws a vertical line where the median is
#boxplot for subsets:
recent_grads[["Major_category","Sample_size"]].boxplot(by="Major_category")
plt.xticks(rotation=90)
#multiple plots in one chart to check por correlation.
plt.scatter(recent_grads["ShareWomen"],recent_grads["Median"],color="red")
plt.scatter(recent_grads["Unemployment_rate"],recent_grads["Median"],color="blue")


#seaborn! http://stanford.edu/~mwaskom/software/seaborn/api.html#api-ref
import seaborn as sns
#historgram
sns.distplot(births['prglngth'], kde=False, rug=True, bins=10, hist=False)
sns.axlabel('Pregnancy Length, weeks', 'Frequency')
#scatterplots
sns.jointplot(x="Median",y="ShareWomen",data=recent_grads)
sns.jointplot(x="Median",y="ShareWomen",data=recent_grads, kind="hex") #hexbin plot. Good for large datasets
# for categorical data: sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)
#pairplots!
sns.pairplot(births[["agepreg","prglngth","birthord"]])
#visualize liniar relationships
sns.regplot(x="ShareWomen",y="Median",data=recent_grads)
sns.regplot(x="ShareWomen",y="Median",hue="sex",data=recent_grads, ) # I can add a hue to fit 2 lines of subgroups and see if additive or interaction model, for markers and colors markers=["o", "x"], palette="Set1"
#if I add col= is like adding another hue, I'll get separete plots. row= will add yet another one!!!! i could have hue="smokers", col="sex" and row="season"
sns.regplot(x="ShareWomen",y="Median",data=recent_grads, lowess=True) #Shows the curve data
sns.residplot(x="ShareWomen",y="Median",data=recent_grads) # check residuals
###### CATEGORICAL
#1.observations within each level of the categorical variable:  stripplot(), boxplot(), and violinplot(), 
#boxplot for subsets: I could use hue for yet another categorical value!
sns.boxplot(x='Major_category',y='Median',data=recent_grads)
plt.xticks(rotation=90)
#I can pass a data.frame to see boxplot of all values :
sns.boxplot(data=recent_grads)
#distribution of observations:
sns.stripplot(x='Major_category',y='Median',data=recent_grads)
plt.xticks(rotation=90)
#2.apply a statistical estimation to show a measure of central tendency and confidence interval:  barplot(), countplot(), and pointplot()
#barplot. This will show the man by default. Again, I could use hue to add yet another categorical value
sns.barplot(x='Major_category',y='Median',data=recent_grads.sort("Median"))
plt.xticks(rotation=90)
#countplot: number of observations for each each categorical value
sns.countplot(x='Major_category',data=recent_grads,palette='Greens_d')
plt.xticks(rotation=90)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)



