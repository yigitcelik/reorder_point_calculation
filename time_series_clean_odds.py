import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_excel('dataset.xlsx')

v3_data = pd.read_excel ("v3mb51.XLSX")
v3_data = v3_data.sort_values (by=["Malzeme", "Belge tarihi"],ascending=False)


def fix(mat):
    """
    It removes the positive values from the dataset in a way that described below.
    This function tries to find the positive value, and it checks the past values before this positive value
    and tries to find a value that is negative and its absolute value is greater of equal to this positive value.
    When it finds the proper value then it adds up the positive value to the negative value that it is founded. And 
    it sets the positive value to zero. It corrects the all values of the material in this way and return the fixed values of the
    material

    Args:
        mat (_type_): The Material no that is asked to be removed the positive consumption values 

    Returns:
        Series: The corrected consumption values of the material that is queried
    """
    t_df = v3_data[v3_data["Malzeme"]==mat].reset_index() #it filters the material no that is asked

    for i,qty in zip(t_df.index,t_df["Miktar"]): #it iterates the index and quantity values
        if qty>0: #when it found the positive value
            try:
                for j in range(i,t_df.shape[0]): #it iterates the past values before itself 
                    if (t_df.loc[j+1,"Miktar"]+qty)<=0:  #when it finds a big enough negative value that has absolute value bigger than the positive value 

                        t_df.loc[j+1, "Miktar"]+=qty # it updates the big negative value that is founded
                        t_df.loc[i, "Miktar"]=0 # it updates the self value to zero
                        break #it exits the loop, because we got rid of the positive value
            except: 
                pass # in case of getting error about indice values etc.
    return t_df["Miktar"]
            


for mat in v3_data["Malzeme"].unique():
    v3_data.loc[v3_data["Malzeme"]==mat, "Miktar"] =fix(mat).values

v3_data = v3_data[v3_data["Miktar"]<0] #if there are still positive values , we can remove them safely

v3_data.to_excel("anonimized_duzenlenmis_mb51.xlsx", Index=False)

