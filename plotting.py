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
#boxplot for subsets:
recent_grads[["Major_category","Sample_size"]].boxplot(by="Major_category")
plt.xticks(rotation=90)
#multiple plots in one chart to check por correlation.
plt.scatter(recent_grads["ShareWomen"],recent_grads["Median"],color="red")
plt.scatter(recent_grads["Unemployment_rate"],recent_grads["Median"],color="blue")


#seaborn!
import seaborn as sns
#historgram
sns.distplot(births['prglngth'], kde=False)
sns.axlabel('Pregnancy Length, weeks', 'Frequency')
#boxplot for subsets:
sns.boxplot(x=births["birthord"], y=births["agepreg"])

