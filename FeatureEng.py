import numpy as np
import pandas as pd
import random as rd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

sales=pd.read_csv("D:\SFU\DataMining\Project\sales_train_v2.csv")
item_cat=pd.read_csv("D:\SFU\DataMining\Project\item_categories.csv")
item=pd.read_csv("D:\SFU\DataMining\Project\items.csv")
sub=pd.read_csv("D:\SFU\DataMining\Project\sample_submission.csv")
shops=pd.read_csv("D:\SFU\DataMining\Project\shops.csv")
#test=pd.read_csv("D:\SFU\DataMining\Project\test.csv")

sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
print(sales.info())

monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
monthly_sales.head(20)

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts);
#plt.show()

import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
fig = res.plot()
fig.show()


from pandas import Series as Series
import math
def growth(dataset, interval=12):
    grow=list()
    for i in range(interval, len(dataset)):
        value = math.log(dataset[i]) - math.log(dataset[i - interval])
        grow.append(value)
    return Series(grow)

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))

new_ts=growth(ts)
plt.plot(new_ts)
plt.plot()
#plt.show()

def test_stationarity(timeseries):
    
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


def difference(dataset, interval = 1):
    diff = list()
    for i in range (interval, len(dataset)):
        value2=new_ts[i]-new_ts[i-1]
        diff.append(value2)
    return (diff)

new2_ts=difference(new_ts)

plt.plot(new2_ts)
#plt.show()

test_stationarity(new2_ts)

