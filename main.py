import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from matplotlib.pylab import rcParams
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


rcParams['figure.figsize'] = 15,6
warnings.filterwarnings('ignore')

os.chdir("C:\stock market")
print(os.getcwd())
df = pd.read_csv("prices.csv", header=0)
# print(df)
# print(df.shape)
# print(df.columns)
# print(df.symbol.value_counts())
# print(df.symbol.unique())
# print(df.symbol.unique().shape)
# print(df.symbol.unique()[0:20])

comp_info = pd.read_csv('securities.csv')
# print(comp_info["Ticker symbol"].nunique())
comp_info.info()

comp_plot = comp_info.loc[(comp_info["Security"] == 'Yahoo Inc.') | (comp_info["Security"] == 'Xerox Corp.') | (comp_info["Security"] == 'Adobe Systems Inc')
              | (comp_info["Security"] == 'Microsoft Corp.') | (comp_info["Security"] == 'Adobe Systems Inc') 
              | (comp_info["Security"] == 'Facebook') | (comp_info["Security"] == 'Goldman Sachs Group') , ["Ticker symbol"] ]["Ticker symbol"] 

for i in comp_plot:
    print (i)

filepath="stock_weights1.hdf5"
 
def plotter(code):
    global closing_stock ,opening_stock
    f, axs = plt.subplots(2,2,figsize=(15,8))
    plt.subplot(212)
    company = df[df['symbol']==code]
    company = company.open.values.astype('float32')
    company = company.reshape(-1, 1) 
    opening_stock = company
    plt.grid(True)
    plt.xlabel('Time') 
    plt.ylabel(code + " open stock prices") 
    plt.title('prices Vs Time') 
    plt.plot(company , 'g') 
    plt.subplot(211)
    company_close = df[df['symbol']==code]
    company_close = company_close.close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    plt.xlabel('Time') 
    plt.ylabel(code + " close stock prices")
    plt.title('prices Vs Time')
    plt.grid(True)
    plt.plot(company_close , 'b') 
    plt.show()

for i in comp_plot:
    plotter(i)

