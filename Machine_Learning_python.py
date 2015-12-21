#training, test set
# Randomly shuffle our data for the training and test set
admissions = admissions.loc[np.random.permutation(admissions.index)]
# train with 700 and test with the following 300, split dataset 
num_train = 700
data_train = admissions[:num_train]
data_test = admissions[num_train:]

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

#LINEAR REGRESSION ###########################
from skilearn.linear.model import LinearRegression
lm = LinearRegresion() # creates linear regresion object
#object methods
lm.fit(x,y) #dataframes
lm.coefs_ 
lm.intercept_
lm.predict([[600., 3.0]])
lm.score() #R2
# or I can do it like this which gives me more info!! OLS -- Ordinary Least Squares Fit
linear = sm.OLS(y, X)
# fit model
linearfit = linear.fit()
linearfit.summary()

#LOGISTIC REGRESSION #########################
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(data_train[['gpa', 'gre']], data_train['admit']) #fit the model
print (logistic_model.coef_)
#predict_proba will return a matrix where the first column is the probability of the event not happening 
#and the second column is the probability of the event happening
fitted_vals = logistic_model.predict_proba(data_train[['gpa', 'gre']])[:,1]
# PREDICTION
predicted_test = logistic_model.predict(data_test[['gpa','gre']])
# TEST RESULTS
accuracy_test = (predicted_test==data_test['admit']).mean()






