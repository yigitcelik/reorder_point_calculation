import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pred_result= pd.read_excel('results.xlsx')  #results of prophet models
mat_data = pd.read_excel('leadtime_and_shelflife.xlsx') #leadtimes and shelf life from material master data

def sScalculation(df,period=360):
    """This function calculates the reorder point and maximum stock level of a 
    material given in a dataframe row. The reorder point is the amount of inventory
    that triggers an order to replenish stock. The maximum stock level is the highest 
    amount of inventory that can be held in the warehouse at any given time. 
    This code takes into account factors such as demand, lead time and shelf life 
    to determine the optimal reorder point and maximum stock level for the material.

    Args:
        df (series): it involves next period predicted demand data
        period (int, optional): It indicates Demand period .  Defaults to 360 days.

    Returns:
        Tuple : This function returns Reorder Point(s),Maximum Stock Level(S), Leadtime(days), Shelf-Life(days) respectively
    """
    

    leadtime = mat_data[mat_data['Malzeme']==df['Malzeme']]['leadtime'].iloc[0] #it retrieves the leadtime info from mat_data

    s = (df['Miktar_up_lower_mean']/period) * leadtime  #the reorder point is the demand per period time * leadtime
    
    shelflife = mat_data[mat_data['Malzeme']==df['Malzeme']]['shelf_life(d)'].iloc[0] #it retrieves the self life info from mat_data 

    if np.isnan(shelflife): #Some of the materials may not have a shelf-life; if so, their shelf-life would be 9999.
        shelflife = 9999

    maxq = (df['Miktar_up_lower_mean']/period) * int(shelflife) #it calculates the maximum quantity in terms of the demand and shelf life info
    
    aS = s*2 # It is assumed that the S value will be 2 times the s value 
    
    if maxq > s:  
        S = min(aS,maxq) #2 times of s is the maximum quantity 
    else:
        S = maxq #if maxq is less than s, then S must be maxq
        s = maxq/2 #if S is maxq, then s could be half of the maxq (it s an assumption)

    return s,S,leadtime,shelflife


pred_result[['s(Reorder Point)','s(Maximum Stock)','Leadtime(day)','Shelf Life(day)']]= pred_result.apply(sScalculation,axis=1,result_type="expand").round(2)

pred_result.to_excel('final_results.xlsx',index=False)
