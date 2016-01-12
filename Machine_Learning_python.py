#training, test set
# Randomly shuffle our data for the training and test set
admissions = admissions.loc[np.random.permutation(admissions.index)]
# Select 70% of the dataset to be training data
num_train = int(sp500.shape[0] * .7)
data_train = admissions.loc[:num_train,:]
data_test = admissions.loc[highest_train_row:,:]

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
from sklearn.linear.model import LinearRegression
lm = LinearRegresion() # creates linear regresion object
#object methods
lm.fit(x,y) #dataframe x, y can be both series or df
lm.coefs_ 
lm.intercept_
lm.predict([[600., 3.0]]) #data frame 
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
#accuracit
accuracy_test = (predicted_test==data_test['admit']).mean()
logreg.score(x,y) # same as accuracity test
#roc_auc (tpr, fpr)
from sklearn.metrics import roc_auc_score,roc_curve
test_probs = logistic_model.predict_proba(data_test[['gpa', 'gre']])[:,1]
testing_auc = roc_auc_score(obs, test_probs) #obs=observed predictions, probs= probs obtained by model
roc_train = roc_curve(data_train["admit"], train_probs) #tpr, fpr, threshold
plt.plot(roc_test[0], roc_test[1])
#pc_auc (precission, recall(tpr))
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(obs, probs)

#MULTICLASSIFICATION (one-versus-all technique (LOG REGRESSION))
# find the unique origins
unique_origins = modified_auto["origin"].unique()
unique_origins.sort()
# dictionary to store models
models = {}
for origin in unique_origins:
    # initialize model to dictionary
    models[origin] = LogisticRegression()
    # select columns for predictors and predictands
    X_train = train[features]
    y_train = train["origin"] == origin
    # fit model with training data
    models[origin].fit(X_train, y_train)
# Dataframe to collect testing probabilities
testing_probs = pandas.DataFrame(columns=unique_origins)
for origin in unique_origins:
    testing_probs[origin]=models[origin].predict_proba(test[features])[:,1]
print (testing_probs)
#check the results (get the one with highest prob)
predicted_origins = testing_probs.idxmax(axis=1)
#checking resutls: CONFUSION MATRIX
for pred in unique_origins:
    predicted = predicted_origins == pred
    print (predicted)
    for obs in unique_origins:
        observed = origins_observed == obs
        #print (observed)
        #print (len(predicted))
        #print (len(observed))
        result = (predicted & observed)
        confusion.loc[pred, obs] = sum(result)

#Validation: Precision, recall, fscore:
from sklearn.metrics import precision_score, recall_score, f1_score
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html #check average options
pr_micro = precision_score(test["origin"], predicted_origins, average='micro')
pr_weighted = precision_score(test["origin"], predicted_origins, average='weighted')
rc_weighted = recall_score(test["origin"], predicted_origins, average='weighted')
f_weighted = f1_score(test["origin"], predicted_origins, average='weighted')

#### K-MEANS CLUSTERING
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5) #5 is the number of clusters we want
kmeans.fit(point_guards[['ppg', 'atr']]) #features use for k-means (this case is 2 dimensinal)
point_guards['cluster'] = kmeans.labels_ #kmeans.labels will give me a series of each cluster for each entry
# we can then visualize the clusters, eg:
def visualize_clusters(df, num_clusters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for n in range(num_clusters):
        clustered_df = df[df['cluster'] == n]
        plt.scatter(clustered_df['ppg'], clustered_df['atr'], c=colors[n-1])
        plt.xlabel('Points Per Game', fontsize=13)
        plt.ylabel('Assist Turnover Ratio', fontsize=13)
visualize_clusters(point_guards, 5)





