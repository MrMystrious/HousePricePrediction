import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import pandas as pd

from abc import ABC,abstractmethod
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def buid_train_model(self,x_train,y_train):
        pass

class LinearRegressionStrategy(ModelBuildingStrategy):
    def buid_train_model(self, x_train, y_train):
        logging.info("starting linear reggression model pipeline")

        pipeline = Pipeline(
            [
                ("scaler",StandardScaler()),
                ("model",LinearRegression())
            ]
        )

        logging.info("Training Linear Reggression model")

        pipeline.fit(x_train,y_train)

        logging.info("Model training complete")

        return pipeline

class ModelBuilder:
    def __init__(self,strategy):
        self.strategy = strategy
    
    def set_strategy(self,strategy):
        self.strategy = strategy
        logging.info("Switing model strategy")
    
    def buildModel(self,x_train,y_train):
        logging.info("Building the model...")
        return self.strategy.buid_train_model(x_train,y_train)
