#https://datatofish.com/statsmodels-linear-regression/

#multiple linear regression in python using both sklearn and statsmodels
#1. review example, 2. check for linearity, 3. performing multiple linear regresion in python, 4. adding tkinter graphical user interface to gather input from users and display prediction results

#in this example we will use multiple linear regression to predict the stock index price (dep variable) of a fictitious economy by using 2 ind/dep variables (int rates and unemployment rate)


#Step 1: capture the above data set in Python using Pandas DataFrame (for larger datasets, you may consider to import your data):

import pandas as pd

Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }

df = pd.DataFrame(Stock_Market, columns = ["Year", "Month", "Interest_Rate", "Unemployment_Rate", "Stock_Index_Price"])

#print(df)

#step 2: before executing a linear regression model, check linear relationship exists between dep and ind variables
#in this case check rel exists betweeen the 1. Stock_Index_Price (dep variable) and Interest_Rate (ind variable), 2. Stock_Index_Price (dep variable) and Unemployment_rate (ind variable)
#Use scatter plots (utlizing the matplotlib library)
#plot the relationship between the Stock_Index_Price and Interest_Rate

import pandas as pd
import matplotlib.pyplot as plt

plt.scatter(df["Interest_Rate"], df["Stock_Index_Price"], color = "green")
plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
plt.xlabel("Interest Rate", fontsize=14)
plt.ylabel("Stock Index Price", fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df["Unemployment_Rate"], df["Stock_Index_Price"], color = "green")
plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)
plt.xlabel("Unemployment Rate", fontsize=14)
plt.ylabel("Stock Index Price", fontsize=14)
plt.grid(True)
plt.show()

#Step3: linear regression

import statsmodels.api as sm
X = df[["Interest_Rate", "Unemployment_Rate"]]
Y = df["Stock_Index_Price"]

X = sm.add_constant(X) #adding a constant
#Why do we add constant in linear regression?
#The constant term prevents this overall bias by forcing the residual mean to equal zero. Imagine that you can move the regression line up or down to the point where the residual mean equals zero.

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)

#interpreting the results
#1. Adjusted R-squared - reflects the fit of the model, higher value generally indicates a better fit, assuming certain conditions are met
#2. Const Coefficient - Y-int. It means that if both the Interest_Rate and Uemployment_Rate coefficients are zero, then the expected output (i.e. the Y) would be equal to the const coef
#3. Interest_Rate Coefficient - represents the change in the output Y due to a change of one unit in the IR (everything else held constant)
#4. Unemployment_Rate coefficient - represents the change in the output Y due to a change of one unit in the unemployment rate (everything else held constant)
#5. std err - reflects the level of accuracy of the coefficients, the lower it is, the higher the level of accuracy
#6. P>|t| - is your p-value, a p-value of less than 0.05 is considered to be stat significant
#7. Confidence interval - represents the range in which our coefficients are likely to fall (with a likelyhood of 95%)

#Making Predictions based on the Regression results

#eq'n for multiple linear regression is Y = C + M1*X1 + M2*X2 + ....
#Stock_Index_Price = (const coef) + (Interest_Rate coef)*X1 + (Unemployment_Rate coef)*X2
#once we plug in coefficients:
#Stock_Index_Price = 1798.4040 + 345.5401*X1 + -250.1466*X2

#Example, let's suppose you want to predict the stock index price, where you just collected the following values for the first month of 2018:
#interest rate = 2.75 (i.e. X1 = 2.75), Unenployment Rate = 5.3 (i.e. X2 = 5.3)
#when you plug in, you get:
# Stock_Index_Price = (1798.4040) + (345.5401)*X1 + (-250.1466)*X2
# Stock_Index_Price = (1798.4040) + (345.5401)*2.75 +(-250.1466)*(5.3) = 1422.86
#The predicted/estimated value for the Stock_Index_Price in Jan 2018 is 1422.86
#Predicted value can be compared to actual value, 1435-1422.86 = 12.14