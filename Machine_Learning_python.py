#mean, sd
mean = nba_stats["pf"].mean()
std_dev = nba_stats["pf"].std()
#skew, kurtosis and modality (characteristics of histograms)
from scipy.stats import skew
from scipy.stats import kurtosis
positive_skew=skew(array)
#normal distribution
from scipy.stats import norm
points = np.arange(-1, 1, 0.01) #vector for -1 to 1 in increments of 0.01
probabilities = norm.pdf(points, 0, .3) #normal distribution (with probability) of points with mean=0 and sd=0.3
plt.plot(points, probabilities)
#correlation
from scipy.stats.stats import pearsonr
r, p_value = pearsonr(nba_stats["fga"], nba_stats["pts"])


