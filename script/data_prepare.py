# -*- encoding: utf-8 -*-
'''
@File    :   data_prepare.py
@Time    :   2020/11/21 19:59:00
@Author  :   Yunlin Lei 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib

import json
import numpy as np
import pandas as pd
import random

lda_label_path = "after_lda_classed_data.json"
labeled_train_data_path = "data/labeled_data.csv"
unlabeled_train_data_path = "data/unlabeled_data.csv"
test_data_path = "data/test_data.csv"
predict_result_path = "predict_result.txt"
label2id = {"教育": 0, "财经": 1, "时政": 2, "房产": 3, "科技": 4,
            "时尚": 5, "家居": 6, "游戏": 7, "体育": 8, "娱乐": 9}
id2label = {"0": "教育", "1": "财经", "2": "时政", "3": "房产", "4": "科技",
            "5": "时尚", "6": "家居", "7": "游戏", "8": "体育", "9": "娱乐"}
split_str = "$#@!"


def main():
    # with open(lda_label_path, "r+", encoding="utf-8") as rf:
    #     lines = rf.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         lda_label_dic = json.loads(line)
    #     rf.close()
    # lda_label = []
    # for i in range(33000):
    #     lda_label.append(lda_label_dic[str(i)])
    lda_label = []
    with open(predict_result_path, "r+", encoding="utf-8") as rf:
        lines = rf.readlines()
        for line in lines:
            line = line.strip()
            lda_label.append(int(line))
    print(len(lda_label))
    unlabeled_data = pd.read_csv(unlabeled_train_data_path)
    unlabeled_data.replace("\n", "。", inplace=True)
    unlabeled_data.replace("\r", "。", inplace=True)
    labeled_data = pd.read_csv(labeled_train_data_path)
    labeled_data.replace("\n", "。", inplace=True)
    labeled_data.replace("\r", "。", inplace=True)
    test_data = pd.read_csv(test_data_path)
    test_data.replace("\n", "。", inplace=True)
    test_data.replace("\r", "。", inplace=True)
    unlabeled_data["pre_label"] = lda_label
    print(unlabeled_data.head(20))
    # unlabeled_data_list = []
    # add_yx_train_data = []
    # add_yl_train_data = []
    # add_ty_train_data = []
    # for index, row in unlabeled_data.iterrows():
    #     unlabeled_data_list.append(row["content"] + split_str + "0")
    #     if(row["pre_label"] == "游戏"):
    #         add_yx_train_data.append(row["content"] + split_str + "7")
    #     if(row["pre_label"] == "体育"):
    #         add_ty_train_data.append(row["content"] + split_str + "8")
    #     if(row["pre_label"] == "娱乐"):
    #         add_yl_train_data.append(row["content"] + split_str + "9")
    # add_ty_train_data = random.sample(add_ty_train_data, 750)
    # add_yl_train_data = random.sample(add_yl_train_data, 750)
    # add_yx_train_data = random.sample(add_yx_train_data, 750)
    # with open("unlabel_data.txt", "w+", encoding="utf-8") as wf:
    #     for line in unlabeled_data_list:
    #         wf.writelines(line + "\n")
    #     wf.close()

    # train_data = []
    # for index, row in labeled_data.iterrows():
    #     train_data.append(row["content"] + split_str + str(label2id[row["class_label"]]))
    # train_data.extend(add_yx_train_data)
    # train_data.extend(add_ty_train_data)
    # train_data.extend(add_yl_train_data)
    # random.shuffle(train_data)

    # #用于取出1/10用于dev集
    # print(len(train_data))
    # print(len(train_data) // 10)
    # indice = list(range(len(train_data)))
    # random.shuffle(indice)
    # dev_length = len(train_data) // 10
    # with open("dev_data.txt", "w+", encoding="utf-8") as wf:
    #     for i in range(dev_length):
    #         line = train_data[indice[i]]
    #         wf.writelines(line + "\n")
    #     wf.close()
    # with open("train_data.txt", "w+", encoding="utf-8") as wf:
    #     for i in range(dev_length, len(train_data)):
    #         line = train_data[indice[i]]
    #         wf.writelines(line + "\n")
    #     wf.close()
    # with open("test_data.txt", "w+", encoding="utf-8") as wf:
    #     for i in range(dev_length, len(train_data)):
    #         line = train_data[indice[i]]
    #         wf.writelines(line + "\n")
    #     wf.close()

    # predict_data = []
    # for index, row in test_data.iterrows():
    #     predict_data.append(row["content"] + split_str + "0")
    # with open("predict_data.txt", "w+", encoding="utf-8") as wf:
    #     for line in predict_data:
    #         wf.writelines(line + "\n")
    #     wf.close()



if __name__ == "__main__":
    main()
