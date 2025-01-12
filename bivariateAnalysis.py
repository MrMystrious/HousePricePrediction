

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway,chisquare,chi2_contingency
from abc import ABC,abstractmethod

class BivariateAnalysisStrategy(ABC):
    
    @abstractmethod
    def analyze(self,df,feature1,feature2):
        pass
    
   

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df, feature1, feature2):
        plt.figure(figsize=(12,6))
        sns.scatterplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xticks(rotation = 90)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    
    def Anova(self,df,feature1,feature2,Return = False):
        grouped_data = df.loc[:, [feature1, feature2]].groupby(feature1)[feature2].apply(list)
        valid_groups = [group for group in grouped_data if len(set(group)) > 1]
        
        if len(valid_groups) < 2:
            print(f"Insufficient valid groups for ANOVA between {feature1} and {feature2}")
            return None if Return else None

        fRatio, pValue = f_oneway(*valid_groups)

        if not Return:
            print("\tfRatio : ", fRatio)
            print("\tpValue : ", pValue)
            return None
        # Always return the values
        return [fRatio, pValue]

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df, feature1, feature2):
        plt.figure(figsize=(12,6))
        sns.boxplot(data = df,x = feature1,y =feature2)
        plt.title(f"{feature1} vs {feature2}")
        plt.xticks(rotation = 90)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    
    def Anova(self,df,feature1,feature2,Return = False):
        features = df.loc[:, [feature1, feature2]].groupby(feature1)[feature2].apply(list)
        fRatio, pValue = f_oneway(*features)

        if not Return:
            print("\tfRatio : ", fRatio)
            print("\tpValue : ", pValue)
            return None
        # Always return the values
        return (fRatio, pValue)
    
    
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df, feature1, feature2):
        encodeVal = pd.get_dummies(df.loc[:,[feature1,feature2]])
        plt.figure(figsize = (14,6))
        sns.heatmap(encodeVal.corr(),annot=True,cmap ='Blues',fmt=".2f")
        plt.title(f"{feature1} vs {feature2}")
        plt.xticks(rotation = 90)
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    
    def Anova(self,df,feature1,feature2,Return = False):
        chi2_stat, p_value, dof, expected = chi2_contingency(pd.crosstab(df[feature1], df[feature2]))
        if(not Return):
            print(f"Chi-Square Statistic: {chi2_stat}")
            print(f"P-Value: {p_value}")
            print(f"Degrees of Freedom: {dof}")
            print("Expected Frequencies:")
            print(expected) 
            return None

        return (chi2_stat,p_value,dof,expected)

class BivariateStrategy:
    def __init__(self,strategy):
        self.strategy =strategy

    def set_strategy(self,strategy):
        self.strategy = strategy

    def execute_strategy(self,df,feature1,feature2):
        self.strategy.analyze(feature1 = feature1,feature2=feature2,df = df)
    
    def Anova(self,df,feature1,feature2,Return = False):
        values = self.strategy.Anova(df = df,feature1= feature1,feature2=feature2,Return=Return)
        return values