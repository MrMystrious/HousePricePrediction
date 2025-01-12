import zipfile
import os
import pandas as pd
from abc import ABC,abstractmethod

class DataIngester(ABC):
    @abstractmethod
    def ingest(self):
        pass


class ZipDataIngester(DataIngester):

    def __init__(self,filename):
        self.filename = filename

    def ingest(self):

        ext = os.path.splitext(self.filename)[1]
        if(ext != '.zip'):
            raise ValueError("The given file is not zip file...")
        
        try:
            with zipfile.ZipFile(self.filename,'r') as zipRef:
                zipRef.extractall("./ExtractedData")
            
            extracted_files = os.listdir("./ExtractedData")
            csvFiles = [file for file in extracted_files if file.endswith(".csv")]

            if(len(csvFiles) < 1):
                print("No CSV files found...")
            elif(len(csvFiles) > 1):
                print("Number of CSV files found ... specify the csv file")

            csvPath = os.path.join("./ExtractedData",csvFiles[0])

            df = pd.read_csv(csvPath)

            return df
        
        except FileNotFoundError as e:
            print(e)

class DataIngesterFactory:
    @staticmethod
    def getData(filepath):   
        if(filepath.endswith(".zip") ==True):
            return ZipDataIngester(filepath)
        else:
            raise ValueError("No Ingester is for this file extension")
        
if __name__ =="__main__":
    
    print(os.getcwd())
    filePath = "C:/Users/Admin/Documents/practice/ml/mlops/dataSet/HouseData.zip"

    dataIngest = DataIngesterFactory.getData(filePath)

    df = dataIngest.ingest()
    print(df.head())


