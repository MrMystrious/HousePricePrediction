
import pandas as pd
import numpy as np
import logging
from abc import ABC,abstractmethod

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")
class OutlierDetection(ABC):
    @abstractmethod
    def detect_outliers(self,df):
        pass

class ZScoreDetection(OutlierDetection):
    def __init__(self,threshold=3):
        self.threshold = threshold
    
    def detect_outliers(self, df):
        logging.info("z score outlier detection")

        z_score = np.abs((df - df.mean())/df.std())

        outliers = z_score > self.threshold
        logging.info("Done z score outlier detection")

        return outliers

class IQRDetection(OutlierDetection):
    def detect_outliers(self, df):
        logging.info("IOR score outlier detection")

        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        
        outliers = (df < (q1 - 1.5*iqr)) | (df > (q3 + 1.5*iqr))

        logging.info("Done IOR score outlier detection")

        return outliers

class OutlierDetector:
    def __init__(self,strategy):
        self.strategy = strategy
    
    def set_strategy(self,strategy):
        logging.info("setting the outlier strategy")
        self.strategy = strategy
    
    def detect_outliers(self,df):
        logging.info(f"Detecing Outlier")
        return self.strategy.detect_outliers(df)

    def handle_outliers(self,df,method = "remove"):
        outliers= self.detect_outliers(df)
        if method == "remove":
            logging.info("removing ouliers")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("capping ouliers")
            df_cleaned= df.clip(lower=df.quantile(0.01),upper=(0.99),axis=1)
        else:
            logging.info(f"No method named {method}can be found... returning the dataframe")
            return df    
        logging.info("Oulier cleaned")
        return df_cleaned

    def detect(self,df,features=[],method=None):
        df_outlier = df.copy()
        if((method != None)&(len(features) != 0)):
            df_outlier[features] = self.handle_outliers(df[features],method=method)
        elif((method != None)&(len(features)==0)):
            df_outlier = self.handle_outliers(df,method=method)
        elif((method == None)&(len(features)!=0)):
            df_outlier = self.detect_outliers(df[features])
        else:
            df_outlier = self.detect_outliers(df)
        
        return df_outlier

            

