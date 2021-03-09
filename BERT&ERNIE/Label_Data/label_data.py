import pandas as pd
import numpy as np
import json
import csv
import codecs

test_data_file="test_data.csv"

unsame_data_file="unsame.csv"


def get_documents_from_csv(test_data_file,unsame_data_file):
    id_content={}
    ids=[]
    ljw=[]
    mf=[]
    contents=[]
    csvFile = open(test_data_file, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        id_content[item[0]]=item[1]

    csvFile.close()

    csvFile = open(unsame_data_file, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        ids.append(item[0])
        ljw.append(item[1])
        mf.append(item[2])
        contents.append(id_content[item[0]])
    csvFile.close()

    # python2可以用file替代open
    with open("to_label_test.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        # 先写入columns_name
        writer.writerow(["id", "ljw", "mf","content"])
        # 写入多行用writerows
        for i in range(len(ids)):
            writer.writerow([ids[i],ljw[i] ,mf[i] ,contents[i] ])
    return

if __name__=="__main__":
    get_documents_from_csv(test_data_file,unsame_data_file)

