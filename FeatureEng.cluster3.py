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


monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date_block_num","item_price","item_cnt_day"].agg({"date_block_num":"mean", "item_price":"mean","item_cnt_day":"sum"})
df=pd.DataFrame(monthly_sales)
df=np.array(df)


#clustering by Clique    
from pyclustering.cluster.clique import clique, clique_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
intervals = 10
threshold = 0
clique_instance = clique(df, intervals, threshold)
clique_instance.process()
clusters = clique_instance.get_clusters()
noise = clique_instance.get_noise()
cells = clique_instance.get_cells()
print("Amount of clusters:", len(clusters))

#defining a variable (cluster) including cluster number for each shop-item id
cl=pd.DataFrame(clusters)
cluster=[]
#for k in range(4):
#       for j in range(1609120):
#              if cl[j][k] in range(1609124):
#                     cluster.append(k)
#              else: cluster.append('nan')             

for k in range(4):
       for j in range(1609120):
              if cl[j][k] >= 0:
                     cluster.append(k)
              else: cluster.append('nan')  
