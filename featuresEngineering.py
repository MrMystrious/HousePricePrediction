import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pandas as pd
import numpy as np
from abc import ABC,abstractmethod
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def applr_transformation(self,df):
        pass

class LogTransformationStrategy(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features= features
    
    def applr_transformation(self,df):

        logging.info(f"Applying log transformation for {self.features}")

        df_transform = df.copy()

        for feature in self.features:
            df_transform[feature] = np.log1p(df[feature])
        
        logging.info("transformation Complete")
        return df_transform

class StrandradScaling(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features = features
        self.scaler= StandardScaler()
    
    def applr_transformation(self, df):

        logging.info(f"Applying strandrad scaler {self.features}")

        df_transform = df.copy()

        df_transform[self.features]= self.scaler.fit_transform(df[self.features])
        logging.info("Done Strandrad scaling..")
        return df_transform

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self,feature,feature_range=[0,1]):
        self.features = feature
        self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def applr_transformation(self, df):
        
        logging.info(f"Applying min max scaler {self.features}")
        df_transform = df.copy()
        df_transform[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Done min max scaling..")

        return df_transform

class OnehotEncoding(FeatureEngineeringStrategy):
    def __init__(self,feature):
        self.features = feature
        self.encoder = OneHotEncoder(drop="first")
    
    def applr_transformation(self, df):
        logging.info(f"Applying One hot encoding {self.features}")
        df_tranform = df.copy()

        encodedDf = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )
        df_tranform = df_tranform.drop(columns=self.features)
        df_tranform = pd.concat([df_tranform,encodedDf],axis=1)
        logging.info("Done one hot scaling..")
        return df_tranform


