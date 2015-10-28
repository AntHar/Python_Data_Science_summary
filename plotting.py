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
#bar char
plt.bar(y_index, area_by_y)
plt.show()
#change styles
print(plt.style.available)
plt.style.use('ggplot')
#histograms. Pandas function built on top of matplotlib (we need to import it)
#check all options in http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html#pandas-dataframe-hist
columns = ['Median','Sample_size']
recent_grads.hist(column=columns,layout=(2,1))
