# -*- encoding: utf-8 -*-
'''
@File    :   EDA.py
@Time    :   2020/11/18 21:19:12
@Author  :   Yunlin Lei
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import numpy as np
import pandas as pd

labeled_train_data_path = "data/labeled_data.csv"
unlabeled_train_data_path = "data/unlabeled_data.csv"
test_data_path = "data/test_data.csv"
exp_labels = ["教育", "财经", "时政", "房产", "科技", "时尚", "家居"]
imp_labels = ["游戏", "体育", "娱乐"]
labels = ["教育", "财经", "时政", "房产", "科技", "时尚", "家居", "游戏", "体育", "娱乐"]


def main():
    labeled_train_data = pd.read_csv(labeled_train_data_path)
    true_label_dic = {}
    true_label_in_text = []
    only_true_label_in_text = []
    label_count = {}
    max_label = []
    max_label_is_true = []
    only_true_label_in_text_result = []
    for label in labels:
        label_count[label] = []

    for index, row in labeled_train_data.iterrows():
        if row["class_label"] in true_label_dic:
            true_label_dic[row["class_label"]].append(row["content"])
        else:
            true_label_dic[row["class_label"]] = [row["content"]]

        if row["class_label"] in row["content"]:
            true_label_in_text.append(1)
        else:
            true_label_in_text.append(0)
        maxx = 0
        maxx_label = ""
        labels_in_text = []
        for label in labels:
            cnt = row["content"].count(label)
            if (cnt >= maxx):
                maxx = cnt
                maxx_label = label
            label_count[label].append(cnt)
            if cnt > 0:
                labels_in_text.append(label)
        max_label.append(maxx_label)
        if maxx_label == row["class_label"]:
            max_label_is_true.append(1)
        else:
            max_label_is_true.append(0)

        if len(labels_in_text) == 1 and labels_in_text[0] == row["class_label"]:
            only_true_label_in_text.append(1)
            only_true_label_in_text_result.append(labels_in_text[0])
        else:
            only_true_label_in_text.append(0)
            if len(labels_in_text) == 1:
                only_true_label_in_text_result.append(labels_in_text[0])
            else:
                only_true_label_in_text_result.append("Nah")

    labeled_train_data["true_label_in_text"] = true_label_in_text
    labeled_train_data["only_true_label_in_text"] = only_true_label_in_text
    labeled_train_data["max_label_is_true"] = max_label_is_true
    labeled_train_data["max_label"] = max_label
    labeled_train_data["only_true_label_in_text_result"] = only_true_label_in_text_result

    # print(labeled_train_data["true_label_in_text"].head(3))
    # print("按label出现判断的召回率")
    # print(labeled_train_data.groupby(["class_label"])[
    #       "true_label_in_text"].sum() / 1000)

    # print("按只有一种label出现判断召回率和准确率")
    # print(labeled_train_data.groupby(["class_label"])[
    #       "only_true_label_in_text"].sum() / 1000)

    # # print(labeled_train_data.groupby(["only_true_label_in_text_result"]).count()["id"])
    # print(
    #     labeled_train_data.groupby(
    #         ["only_true_label_in_text_result"])["only_true_label_in_text"].sum() /
    #     labeled_train_data.groupby(
    #         ["only_true_label_in_text_result"]).count()["id"])

    # print("按文章内最大标签判断召回率和准确率")
    # print(labeled_train_data.groupby(["class_label"])[
    #       "max_label_is_true"].sum() / 1000)

    print(
        labeled_train_data.groupby(
            ["max_label"])["max_label_is_true"].sum() /
        labeled_train_data.groupby(
            ["max_label"]).count()["id"])
    # print(labeled_train_data.groupby(
    #         ["max_label"]).count()["id"])
    # print(labeled_train_data.groupby(
    #         ["only_true_label_in_text_result"]).count()["id"])


if __name__ == "__main__":
    main()
