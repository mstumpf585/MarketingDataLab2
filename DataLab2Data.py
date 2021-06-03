import pandas as pd 
import statsmodels.api as sm

# Question 1 ######################################
# Get Population dist of Geneder 
# read in data with pandas 
data = pd.read_csv("AFC.csv")

# First regression 

X = data[['age', 'gender', 'educat', 'income', 'status']]
Y = data['visits']

sm.add_constant(X)

model = sm.OLS(Y,X, missing='drop').fit()
predictions = model.predict(X)

print_model = model.summary()
print (print_model)

# Question 2 ########################################

data2 = pd.read_csv("AFC.csv")

X = data[['status', 'weight', 'classes', 'station', 'pool']]
Y = data['recom']

sm.add_constant(X)

model = sm.OLS(Y,X, missing='drop').fit()
predictions = model.predict(X)

print_model = model.summary()
print (print_model)



