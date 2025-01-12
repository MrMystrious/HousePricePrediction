import pandas as pd
from abc import ABC,abstractmethod

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self,df):
        pass

class DataTypeInspectionStrategy(DataInspectionStrategy):
    def inspect(self,df):
        print("Datatypes and Non-Null counts : ")
        print(df.info())

class SummaryInspectionStrategy(DataInspectionStrategy):
    def inspect(self,df):

        print("Summary (Numerical Features) : ")
        print(df.describe())
        print("\n\n\nSummary (Categorical Features) : ")
        print( df.select_dtypes(include=['object']).apply(pd.Series.describe))

class DataInspector:
    def __init__(self,strategy):
        self._strategy = strategy

    def setStrategy(self,strategy):
        self._strategy = strategy

    def ExecuteInspection(self,df):

        self._strategy.inspect(df)