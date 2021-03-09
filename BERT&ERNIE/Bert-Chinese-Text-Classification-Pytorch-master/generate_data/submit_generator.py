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

id2fengxian = {"0": "低风险", "1": "高风险", "2": "高风险", "3": "中风险", "4": "中风险",
                "5": "低风险", "6": "可公开", "7": "低风险", "8": "可公开", "9": "可公开"}


def main():
    result_label = []
    result_name = []
    with open(predict_result_path, "r+", encoding="utf-8") as rf:
        lines = rf.readlines()
        for line in lines:
            line = line.strip()
            result_label.append(line)
            result_name.append(id2label[line])
    print(len(result_label))
    id = 0

    test_data = pd.read_csv(test_data_path)
    test_data["pre_label"] = result_name
    print(test_data.groupby(["pre_label"]).head(5))

    with open("submit.csv", "w+", encoding="utf-8") as wf:
        wf.writelines("id,class_label,rank_label\n")
        for line in result_label:
            wf.writelines(str(id) + "," + id2label[line] + "," + id2fengxian[line] + "\n")
            id += 1
        wf.close()



if __name__ == "__main__":
    main()
    