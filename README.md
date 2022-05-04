# suryakamal
# simple linear regression model in machine learning using pandas,numpy,matplotpy
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the dataset
data=pd.read_csv('Salary_Data.csv')
x=data['YearsExperience']
y=data['Salary']


# print 1st few rows
print(data.head())



# calculating_regression
def linear_regression(x,y):
  N=len(x)
  x_mean=x.mean()
  y_mean=y.mean()
  b1_num=((x-x_mean)*(y-y_mean)).sum()
  b1_den=((x-x_mean)**2).sum()
  b1=b1_num/b1_den
  b0=y_mean-(b1*x_mean)
  reg_line='y={}+{}B'.format(b0,round(b1,3))
  return(b0,b1,reg_line)



# calculating how well the regression line fits
def corr_coef(x,y):
  N=len(x)
  num=(N*(x*y).sum())-(x.sum()*y.sum())
  den=np.sqrt((N*(x**2).sum()-x.sum()**2)*(N*(y**2).sum()-y.sum()**2))
  R=num/den
  return R



# Applying these functions to our data,we can printout the results
b0,b1,reg_line=linear_regression(x,y)
print('Regression line:',reg_line)
R=corr_coef(x,y)
print('correlation coef:',R)
print('"goodness of fit":',R**2)




# plotting the regression line
plt.figure(figsize=(12,5))
plt.scatter(x,y,s=300,lineWidths=1,edgecolor='black')
text='"mean:{}years ymean:${}R:{}R^2:{}y={}+{}x"'.format(round(x.mean(),2),round(y.mean(),2),round(R,4),round(R**2,4),round(b0,3),round(b1,3))
plt.text(x=1,y=100000,s=text,fontsize=12,bbox={'facecolor':'grey','alpha':0.2,'pad':10})
plt.title('how experience affects salary')
plt.xlabel('year of experience',fontsize=15)
plt.ylabel('salary',fontsize=15)
plt.plot(x,b0+b1*x,c='r',lineWidth=5,alpha=0.5,solid_capstyle='round')
plt.scatter(x=x.mean(),y=y.mean(),marker='*',s=10**2.5,c='r')
