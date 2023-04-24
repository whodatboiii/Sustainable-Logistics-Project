import ortools
import pandas as pd
import openpyxl
import warnings

warnings.simplefilter(action='ignore', category=UserWarning) #stop the warnings >:(

def project_data():
    """Import the data"""
    #Manually deleted columns we didn't need in Excel
    database = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name = "Netz Bern 01.01.22 - 31.12.22",
        names = ['origin', 'dest', 'count']
    ) 

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name = "locations",
        names = ['id', 'name', 'lat', 'lon']
    )
    print(database.head(),"\n",locations.head())
    #So pretty

    #Will have to change data below
    charge_cost = 5
    warehouse_loc = pd.DataFrame({'id': ['9999'], 'name': ['Warehouse'], 'lat': ['46.9466'], 'lon': ['7.4443']})
    locations = pd.concat([locations, warehouse_loc], ignore_index=True)

    print(locations.tail()) #yay this works

    return database, locations

if __name__ == "__main__":
    project_data()