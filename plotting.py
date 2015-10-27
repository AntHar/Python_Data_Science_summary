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
