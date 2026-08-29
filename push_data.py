
import os
import sys
import json
import pymongo

from dotenv import load_dotenv
load_dotenv()
mongo_db_url=os.getenv("Mongo_DB_URL")
print(mongo_db_url)


import certifi
##this library provides tested collection of SSL/TSL root certificates.It helps python to verify that it is connecting to a secured https website.
import pandas as pd
import numpy as np
from skyguard.logger.logger import logger
from skyguard.exception.exception import CustomException

class NetworkDataExtract:
    def __init__(self):
        pass

    def csv_to_json_convertor(self, filepath, station_name):

        try:
           data = pd.read_csv(filepath)
           data.reset_index(drop=True, inplace=True)
           data["station"] = station_name
           records = list(json.loads(data.T.to_json()).values())
           return records
        except Exception as e:
           raise CustomException(e, sys)
    def insert_data_mongoDB(self,records,database,collection):
        try:
            self.records=records
            self.database=database
            self.collection=collection
            
            
            ca = certifi.where()

            self.mongo_client = pymongo.MongoClient(
             mongo_db_url,
             tlsCAFile=ca
)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    DATABASE = "SkyGuardDB"
    Collection = "Sensor_readings"
    RAW_DIR = "Sensor_DATA"

    networkobj = NetworkDataExtract()
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".csv"):
            station_name = filename.replace(".csv", "")
            filepath = os.path.join(RAW_DIR, filename)
            records = networkobj.csv_to_json_convertor(filepath, station_name)
            no_of_records = networkobj.insert_data_mongoDB(records, DATABASE, Collection)
            print(f"{station_name}: inserted {no_of_records} records")