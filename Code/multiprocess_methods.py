from multiprocessing import Pool
from tkinter import W
from tqdm import tqdm
import numpy as np
import json
import csv

selectedKeys = []

def worker(line):
    row = json.loads(line)
    return [row.get(key) for key in selectedKeys]

def splitDataset(dataPath:str, totalSize:int, selectedKeys:list, outputPath:str, processNum:int = 16, chunkSize:int = 1024, printParameters:bool = False):
    """
    multiprocess split function

    Args:
        dataPath (str): original dataset path
        totalSize (int): total size of original dataset
        selectedKeys (list): a list containing selected keys
        outputPath (str): output dataset path
        processNum (int, optional): number of sub-processes. Defaults to 16.
        chunkSize (int, optional): the size of each chunk splitted from the iterable. Defaults to 1024.
        printParameters (bool, optional): show the parameters or not. Defaults to False.
    """

    if printParameters:
        print(f"Dataset File: {dataPath}")
        print(f"Total Size: {totalSize}")
        print(f"Selected Keys: {selectedKeys}")
        print(f"Output File: {outputPath}")
        print(f"Number of processes: {processNum}")
        print(f"Size of each chunk: {chunkSize}")
    
    pool = Pool(processNum)
    with open(dataPath, 'r', encoding="utf-8") as file:
        with open(outputPath, 'w', newline='') as outputFile:
            csvWriter = csv.writer(outputFile, selectedKeys)
            csvWriter.writerow(selectedKeys) # write header
            with tqdm(total=totalSize, desc="Processing", unit_scale=True) as pbar:
                for row in pool.imap(worker, file, chunkSize):
                    csvWriter.writerow(row)
                    pbar.update()
    pool.close()
    pool.join()

if __name__ == '__main__':
    dataPath = "Data/All_Amazon_Review_5.json"
    totalSize = 157260921
    outputPath = "Data/test.csv"
    processNum = 16 # number of processes (customize this variable based on different CPUs)
    chunkSize = 1024 # the size of each chunk splitted from the iterable
    try:
        main(dataPath, totalSize, selectedKeys, outputPath, processNum=16, chunkSize=1024)
    except Exception as e:
        print(e)
        exit()