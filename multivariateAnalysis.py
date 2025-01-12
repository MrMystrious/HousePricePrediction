import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC,abstractmethod

class MultivariateAnalysis(ABC):
    def analyse(self,df):
        
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
    
    @abstractmethod
    def generate_correlation_heatmap(self,df):
        pass
    @abstractmethod
    def generate_pairplot(self,df):
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysis):
    def generate_correlation_heatmap(self, df):
        plt.figure(figsize=(12,6))
        sns.heatmap(df.corr(),fmt=".2f",cmap='coolwarm',annot=True)
        plt.title("Correlation heatmap")
        plt.show()
    
    def generate_pairplot(self, df):
        plt.figure(figsize=(12,6))
        sns.pairplot(df)
        plt.suptitle("Pair plot of the selected features",y=1.02)
        plt.show()