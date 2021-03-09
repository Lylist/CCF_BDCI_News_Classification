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
import pandas as pd
import random

lda_label_path = "after_lda_classed_data.json"
interaction_data_path = "interaction_three_new_class_1000_data_dict.json"
labeled_train_data_path = "data/labeled_data.csv"
new_labeled_train_data_path = "data/labeled_data_content_dict.json"
unlabeled_train_data_path = "data/unlabeled_data.csv"
new_unlabeled_train_data_path = "data/unlabeled_data_content_dict.json"
test_data_path = "data/test_data.csv"
new_test_data_path = "data/test_unlabeled_data_content_dict.json"
predict_result_path = "unlabel_result.txt"

label2id = {"教育": 0, "财经": 1, "时政": 2, "房产": 3, "科技": 4,
            "时尚": 5, "家居": 6, "游戏": 7, "体育": 8, "娱乐": 9}
id2label = {"0": "教育", "1": "财经", "2": "时政", "3": "房产", "4": "科技",
            "5": "时尚", "6": "家居", "7": "游戏", "8": "体育", "9": "娱乐"}
split_str = "$#@!"


def main():
    with open(interaction_data_path, "r+", encoding="utf-8") as rf:
        lines = rf.readlines()
        for line in lines:
            line = line.strip()
            lda_label_dic = json.loads(line)
        rf.close()
    lda_label = []
    for i in range(33000):
        if lda_label_dic.get(str(i)):
            lda_label.append(lda_label_dic[str(i)])
        else:
            lda_label.append("教育")

    filter_labeled_train_data = {}
    with open(new_labeled_train_data_path, "r+", encoding="utf-8") as rf:
        lines = rf.readlines()
        for line in lines:
            line = line.strip()
            filter_labeled_train_data = json.loads(line)
        rf.close()

    filter_unlabeled_train_data = {}
    with open(new_unlabeled_train_data_path, "r+", encoding="utf-8") as rf:
        lines = rf.readlines()
        for line in lines:
            line = line.strip()
            filter_unlabeled_train_data = json.loads(line)
        rf.close()

    filter_test_data = {}
    with open(new_test_data_path, "r+", encoding="utf-8") as rf:
        lines = rf.readlines()
        for line in lines:
            line = line.strip()
            filter_test_data = json.loads(line)
    # lda_label = []
    # with open(predict_result_path, "r+", encoding="utf-8") as rf:
    #     lines = rf.readlines()
    #     for line in lines:
    #         line = line.strip()
    #         lda_label.append(id2label[line])
    #     rf.close()
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
    print(unlabeled_data.groupby(["pre_label"])["id"].count())
    unlabeled_data_list = []
    add_yx_train_data = []
    add_yl_train_data = []
    add_ty_train_data = []
    with open("unlabel_data.txt", "w+", encoding="utf-8") as wf:
        for index, row in unlabeled_data.iterrows():
            content = "".join(filter_unlabeled_train_data[str(row["id"])])
            wf.writelines(content + split_str + str(label2id[row["pre_label"]]) + "\n")
            if (row["pre_label"] == "游戏"):
                add_yx_train_data.append(content + split_str + "7")
            elif (row["pre_label"] == "体育"):
                add_ty_train_data.append(content + split_str + "8")
            elif (row["pre_label"] == "娱乐"):
                add_yl_train_data.append(content + split_str + "9")
            else:
                unlabeled_data_list.append(content + split_str + str(label2id[row["pre_label"]]))
        wf.close()
    # print(add_ty_train_data[0:20])
    # print(add_yl_train_data[0:20])
    # print(add_yx_train_data[0:20])
    add_ty_train_data = random.sample(add_ty_train_data, 1000)
    add_yl_train_data = random.sample(add_yl_train_data, 1000)
    add_yx_train_data = random.sample(add_yx_train_data, 1000)
    # unlabeled_data_list = random.sample(unlabeled_data_list, 20000)

    train_data = []
    for index, row in labeled_data.iterrows():
        content = "".join(filter_labeled_train_data[str(row["id"])])
        train_data.append(content + split_str + str(label2id[row["class_label"]]))
    train_data.extend(add_yx_train_data)
    train_data.extend(add_ty_train_data)
    train_data.extend(add_yl_train_data)
    # train_data.extend(unlabeled_data_list)
    random.shuffle(train_data)

    # 用于取出1/10用于dev集
    print(len(train_data))
    print(len(train_data) // 50)
    indice = list(range(len(train_data)))
    random.shuffle(indice)
    dev_length = len(train_data) // 50
    with open("dev_data.txt", "w+", encoding="utf-8") as wf:
        for i in range(dev_length):
            line = train_data[indice[i]]
            wf.writelines(line + "\n")
        wf.close()
    with open("train_data.txt", "w+", encoding="utf-8") as wf:
        for i in range(dev_length, len(train_data)):
            line = train_data[indice[i]]
            wf.writelines(line + "\n")
        wf.close()
    with open("test_data.txt", "w+", encoding="utf-8") as wf:
        for i in range(dev_length, len(train_data)):
            line = train_data[indice[i]]
            wf.writelines(line + "\n")
        wf.close()

    predict_data = []
    for index, row in test_data.iterrows():
        content = "".join(filter_test_data[str(row["id"])])
        predict_data.append(content + split_str + "0")
    with open("predict_data.txt", "w+", encoding="utf-8") as wf:
        for line in predict_data:
            wf.writelines(line + "\n")
        wf.close()


if __name__ == "__main__":
    main()
