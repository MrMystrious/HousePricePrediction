import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from abc import ABC,abstractmethod

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s")

class MissingValuesAnalysisTemplate(ABC): 
    @abstractmethod
    def idenetifyMissingValues(self,df):
        pass
    @abstractmethod
    def visualizeMissingValues(self,df):
        pass

class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def __init__(self,method,value=None):
        self.method = method
        self.value = value
        
    
    def idenetifyMissingValues(self, df):
        try:
            print("Missing values in the DataFrame : ")
            missingVal = pd.DataFrame({
                'Missing Values' : df.isnull().sum(),
                'Data Tyoe ': df.dtypes
            })
            print(missingVal[missingVal['Missing Values'] != 0])
        except Exception as e:
            print("Error Occured : ",e)
           
    
    def visualizeMissingValues(self, df):
        print("Visualising the null values : ")
        plt.figure(figsize=(20,8))
        sns.heatmap(df.isnull(),cmap='viridis')
        plt.xticks(rotation=90, fontsize=8)
        plt.title('Missing Values...')
        plt.show()
        df.isnull().sum()[df.isnull().sum() > 0].plot(kind='bar', color='skyblue')
    
    def Handle(self, df):

        numericalCols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
        for col in numericalCols:
            if df[col].isna().sum() > 0:  
                if self.method == "mn":
                    df[col] = df[col].fillna(df[col].mean())
                elif self.method == "md":
                    df[col] = df[col].fillna(df[col].median())
                elif self.method == "me":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    print("no method selected")
            else:
                print("skipping...",col)

       
        categoricalCols = df.select_dtypes(include=["object"]).columns
        for col in categoricalCols:
            if df[col].isna().sum() > 0:  
                top = df[col].mode()[0]
                df[col] = df[col].fillna(top)

        return df
    
    def Drop(self,df,lis,axis=1):
        df_droped = df.drop(lis,axis=axis) 
        return df_droped
        

class MissingValuesHandlingFactory():
    def __init__(self,method):
        self.strategy = SimpleMissingValuesAnalysis(method)
        
    def analyze(self,df):
        self.strategy.visualizeMissingValues(df)
        self.strategy.idenetifyMissingValues(df)
    def handle(self,df):
        return self.strategy.Handle(df) 
    def dropCol(self,df,lis,axis):
        return self.strategy.Drop(df = df,lis = lis,axis=axis)
    def setMethod(self,method):
        self.strategy.method = method
          