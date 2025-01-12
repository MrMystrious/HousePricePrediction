import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import logging
from abc import ABC,abstractmethod
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplitStrategy(ABC):
    @abstractmethod
    def split_data(self,df):
        pass

class SimpleTestTrainStrategy(DataSplitStrategy):
    def __init__(self,test_size=0.2,random_state=32):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self,df,target):
        logging.info("Performing simple test train split")
        x = df.drop(target,axis=1)
        y = df[target]

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=self.test_size,random_state=self.random_state)

        logging.info("test train split completed")

        return x_train,x_test,y_train,y_test

class DataSpliter:
    def __init__(self,strategy):
        self.strategy =strategy
    
    def set_strategy(self,strategy):
        self.strategy = strategy
    
    def split(self,df,target):
        logging.info("Splitting....")
        x_train,x_test,y_train,y_test = self.strategy.split_data(df,target)
        return  x_train,x_test,y_train,y_test