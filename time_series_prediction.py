from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import pandas as pd
from sktime.utils.plotting import plot_series
import sys
from sklearn.metrics import mean_absolute_error, r2_score
from sktime.forecasting.arima import AutoARIMA
from fbprophet import Prophet
from sktime.datasets import load_airline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import Image
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import warnings
import datetime
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

current_year = datetime.date.today().year

data = pd.read_excel('anonimized_duzenlenmis_mb51.xlsx',parse_dates=True) #Material consumptions in the past 
data = data[data['Giriş tarihi']<f'{current_year}'] #it removes the current year data from the dataset
data['yil'] = data['Giriş tarihi'].dt.year  #it creates a columns that has year info.


def predict(mat,period=365):
    """This function calculates the sum of next period material consumptions with Facebook Prophet Model.
    it uses the past consumption data of the given material. But the consumption datas are too few to make
    a good model. So this function firstly groups the data by year and sum up the consumption values, then
    set these values to the end of the year and then it fills the days of the whole year with total year consumption 
    value that divided by 365(days).
    Finally it gives the manupulated data to Probhet model , then it gets the prediction of the next period time data 
    from the fitted model. 

    Args:
        mat (string): Material number that is asked to predict 
        period (int, optional): It indicates in days future period to be predicted.  Defaults to 365 days.

    Returns:
        float: it returns the sum of the yhat values in given period from prediction dataframe. yhat_lower_sum,yhat_sum,yhat_upper_sum respectively
    """

    temp_df = data[data['kod']==mat].reset_index(drop='index')  #it filters the given material number from the dataset
    #temp_df['Giriş tarihi'] = temp_df['Giriş tarihi'].dt.to_period('M').dt.to_timestamp()
    #temp_df['yil'] = temp_df['Giriş tarihi'].dt.year

    for year in range(data['yil'].min()-1,data['yil'].max()+1): #In case of missing year info in the bfilling process, it adds all neccesary years info with zero quantity
        temp_df=temp_df.append({'kod':mat,'Temel ölçü birimi':'XX','yil':year,'Miktar':0,'Giriş tarihi':pd.to_datetime(year, format='%Y')},ignore_index=True)

    temp_df = temp_df.groupby(['yil'])['Miktar'].sum().reset_index() #it adds up all consumption values by year

    temp_df['Giriş tarihi']=pd.to_datetime(temp_df['yil'], format='%Y')+pd.offsets.YearEnd() #it add date columns to the dataframe and set the date value as the end date of the year 

    temp_df= temp_df.drop(['yil'],axis=1) #it removes the year column


    temp_df_new = temp_df.set_index('Giriş tarihi').resample('D').bfill().assign(Miktar = lambda df : df['Miktar']/period) #it divides the total year quantity to period(365) and finds the value per day, then fills the all days of the year with this value

    temp_df_new=temp_df_new.reset_index().rename({'Giriş tarihi':'ds','Miktar':'y'},axis=1) #it renames the columns in order to make the Prophet model input.

    temp_df_new['y'] = temp_df_new['y'].apply(lambda x: abs(x)) #Our consumption values are all negative. To avoid the misunderstanding  , converts these values to positive.


    model = Prophet() #it creates a prophet model with default setting
    model.fit(temp_df_new)
    data_future = model.make_future_dataframe(periods=period,freq='D') #it creates future dataset. next period days.
    data_pred= model.predict(data_future)
    model.plot(data_pred)  # it plots the past and future data prediction


    #There may be negative values in the prediction. So it filters all positive values then sum up the positive values.
    yhat_lower_sum=data_pred.tail(period)['yhat_lower'][data_pred.tail(period)['yhat_lower']>0].sum()

    yhat_sum =data_pred.tail(period)['yhat'][data_pred.tail(period)['yhat']>0].sum()

    yhat_upper_sum =data_pred.tail(period)['yhat_upper'][data_pred.tail(period)['yhat_upper']>0].sum()
    
    return yhat_lower_sum,yhat_sum,yhat_upper_sum




predict_results={'Malzeme':[],'Miktar_lower':[],'Miktar':[],'Miktar_upper':[],'Miktar_up_lower_mean':[]} #this is a dictionary that we collect the results by material no,yhat_lower,yhat,yhat_upper and the mean of the lower and upper values respectively

for mat in data['kod'].unique():
    predict_results['Malzeme'].append(mat)
    l,m,u = predict(mat=mat)
    predict_results['Miktar_lower'].append(l)
    predict_results['Miktar'].append(m)
    predict_results['Miktar_upper'].append(u)
    predict_results['Miktar_up_lower_mean'].append(np.mean([l,u]))

result_df = pd.DataFrame(predict_results)

result_df[result_df.select_dtypes(exclude='object').columns]= result_df.select_dtypes(exclude='object').round(2) #it rounds the all numerical values in 2 decimal



result_df['Temel ölçü birimi'] = result_df['Malzeme'].apply(lambda mat:data[data['kod']==mat]['Temel ölçü birimi'].max()) #it retrieves the unit of measurement value from the dataset


result_df.to_excel('results.xlsx',index=False) 

