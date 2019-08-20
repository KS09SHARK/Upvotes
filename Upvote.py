# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Dataset
train = pd.read_csv('Train_Set.csv')
test = pd.read_csv('Test_Set.csv')

train['Tag'].value_counts()
train.drop(['ID','Username'],axis = 1, inplace = True)
test.drop(['ID','Username'],axis = 1, inplace = True)

train.isnull().sum()

df = [train,test]
df = pd.concat(df,axis = 0)

train.corr()

X = train['Views'].values.reshape(-1,1)
Y = train['Upvotes']
X = pd.DataFrame(X)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(X,Y,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_cv_pred = regressor.predict(X_cv)

rmse = sqrt(mean_squared_error(Y_cv_pred,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

regressor_OLS.summary()

df_2 = [X,train['Reputation']]
df_2 = pd.concat(df_2,axis = 1)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(df_2,Y,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor_2 = LinearRegression()
regressor_2.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_pred_cv_2 = regressor_2.predict(X_cv)

rmse_2 = sqrt(mean_squared_error(Y_pred_cv_2,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

df_3 = [df_2,train['Answers']]
df_3 = pd.concat(df_3,axis = 1)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(df_3,Y,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor_3 = LinearRegression()
regressor_3.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_pred_cv_3 = regressor_3.predict(X_cv)

rmse_3 = sqrt(mean_squared_error(Y_pred_cv_3,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

regressor_OLS.summary()

df_4 = [train,test]
df_4 = pd.concat(df_4,axis = 0)

df_4 = pd.get_dummies(df_4)
df_4.drop(['Tag_a'],axis = 1, inplace = True)

x_train = df_4[:330045]
x_test = df_4[330045:]

x_test.drop(['Upvotes'],axis = 1, inplace = True)
y_train = x_train['Upvotes']
x_train.drop(['Upvotes'],axis = 1, inplace = True)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(x_train,y_train,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor_4 = LinearRegression()
regressor_4.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_pred_cv_4 = regressor_4.predict(X_cv)

rmse_4 = sqrt(mean_squared_error(Y_pred_cv_4,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

regressor_OLS.summary()

y_pred = regressor_4.predict(x_test)

y_pred = pd.DataFrame(y_pred)

y_pred.to_csv('Sample.csv')


from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

labelencoder_X = LabelEncoder()
train['Tag'] = labelencoder_X.fit_transform(train['Tag'])
test['Tag'] = labelencoder_X.transform(test['Tag'])
x_train = train.drop(['Upvotes'],axis = 1)
poly_reg = PolynomialFeatures(degree = 4,interaction_only=False, include_bias=True)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(x_train, y_train)
lin_reg_1 = linear_model.LassoLars(alpha=0.021,max_iter=150)
lin_reg_1.fit(X_poly, y_train)

y_pred = lin_reg_1.predict(poly_reg.fit_transform(test))


y_pred = pd.DataFrame(y_pred)

y_pred.to_csv('Sample.csv')



Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@KS09SHARK 
Learn Git and GitHub without any code!
Using the Hello World guide, you’ll start a branch, write comments, and open a pull request.

 
1
8 4 sand47/Enigma-codeFest-Machine-Learning
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Security  Insights
Enigma-codeFest-Machine-Learning/main.py
@sand47 sand47 Add files via upload
12197b2 on Sep 2, 2018
72 lines (49 sloc)  2.14 KB
'''    
import pandas as pd
import numpy as np
from  sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv('Train_Set.csv')
train = train.drop(train[train.Views > 3000000].index)

     
labelencoder_X = LabelEncoder()
train['Tag'] = labelencoder_X.fit_transform(train['Tag'])
train.drop(['ID','Username'], axis=1,inplace =True)
target = train['Upvotes']

from sklearn.preprocessing import Binarizer
bn = Binarizer(threshold=7)
pd_watched = bn.transform([train['Answers']])[0]
train['pd_watched'] = pd_watched


feature_names = [x for x in train.columns if x not in ['Upvotes']]

x_train, x_val, y_train, y_val = train_test_split(train[feature_names], target,test_size = 0.22,random_state =205)
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val = sc_X.transform(x_val)

poly_reg = PolynomialFeatures(degree = 4,interaction_only=False, include_bias=True)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(x_train, y_train)
lin_reg_1 = linear_model.LassoLars(alpha=0.021,max_iter=150)
lin_reg_1.fit(X_poly, y_train)

# predicitng 
pred_val = lin_reg_1.predict(poly_reg.fit_transform(x_val))

print(r2_score(y_val, pred_val))

# ---------------------------------------------------------------------------------------

# testing

test = pd.read_csv('Test_Set.csv')
ids = test['ID']
test.drop(['ID','Username'], axis=1,inplace =True)


labelencoder_X = LabelEncoder()
test['Tag'] = labelencoder_X.fit_transform(test['Tag'])

from sklearn.preprocessing import Binarizer
bn = Binarizer(threshold=7)
pd_watched = bn.transform([test['Answers']])[0]
test['pd_watched'] = pd_watched

   
test = sc_X.fit_transform(test)

pred_test = lin_reg_1.predict(poly_reg.fit_transform(test))
pred_test=abs(pred_test)


submission = pd.DataFrame({'ID': ids,
                           'Upvotes':pred_test
                           })

submission.to_csv("final_sub477.csv",index=False)

