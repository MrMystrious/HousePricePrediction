

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC,abstractmethod

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df,feature):
        pass

class NumericalUnivariateAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df, feature):
        plt.figure(figsize=(12,8))
        sns.histplot(df[feature],kde=True,bins=70)
        plt.xlabel(feature)
        plt.ylabel("feature count")
        plt.show()

class CategoricalUivariateAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df, feature):
        plt.figure(figsize=(16,8))
        sns.countplot(x=feature,data=df,palette='muted')
        plt.xlabel(feature)
        plt.xticks(rotation = 90)
        plt.ylabel("feature count")
        plt.show()