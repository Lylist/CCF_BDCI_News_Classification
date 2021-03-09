# coding=utf-8
# MF 2020.8.25
import os
import  data_process

import pandas as pd
import  numpy as np
from cntopic import  Topic
import  jieba

unlabeled_lda_model_path="output/unlabeled/model/unlabeled_data_lda_model"
unlabeled_dict_path="output/unlabeled/dictionary.dict"


labeled_lda_model_path = "output/labeled/model/labeled_data_lda_model"
labeled_dict_path = "output/labeled/dictionary.dict"

and_and_unlabeled_data_after_keywords_classed_data_json_path="key_words_result_data/and_and_unlabeled_data_after_keywords_classed_data.json"
and_or_unlabeled_data_after_keywords_classed_data_json_path="key_words_result_data/and_or_unlabeled_data_after_keywords_classed_data.json"
or_or_unlabeled_data_after_keywords_classed_data_json_path="key_words_result_data/or_or_unlabeled_data_after_keywords_classed_data.json"

all_key_words_data_path=[and_and_unlabeled_data_after_keywords_classed_data_json_path,
                         and_or_unlabeled_data_after_keywords_classed_data_json_path
                         ]




id_label_dict={
    0:"财经",
    1:"时政",
    2:"科技",
    3:"游戏",
    4:"娱乐",
    5:"体育",
    6:"时尚",
    7:"家居",
    8:"教育",
    9:"房产"
}

label_id_dict={
    "财经":0,
    "时政":1,
    "科技":2,
    "游戏":3,
    "娱乐":4,
    "体育":5,
    "时尚":6,
    "家居":7,
    "教育":8,
    "房产":9
}

label_keywords_dict={
    "财经":["市场","价格","财经"],
    "时政":["责任编辑","工作","时政"],
    "科技":["手机","发布","科技"],
    "游戏":["玩家","时间","游戏"],
    "娱乐":["电影","导演","娱乐"],
    "体育":["新浪","比赛","体育"],
    "时尚":["系列","设计","时尚"],
    "家居":["原文","产品","家居"],
    "教育":["责编","学生","教育"],
    "房产":["项目","平米","房产"],
}




after_quchong_union_keywords_result_dict_json_path="key_words_result_data/after_quchong_union_keywords_result_dict.json"
two_after_quchong_union_keywords_result_dict_json_path="key_words_result_data/2after_quchong_union_keywords_result_dict.json"
lda_result_path="AllData/after_lda_classed_data.json"

def intersection_keywords_lda_result():
    print("开始进行交操作，生成新的三个类别分类结果")
    intersection_keywords_result_dict={}
    two_intersection_keywords_result_dict={}
    result_keys = ["娱乐", "游戏", "体育"]

    lda_json_data = data_process.read_data_from_json(lda_result_path)
    union_keywords_1_dict=data_process.read_data_from_json(after_quchong_union_keywords_result_dict_json_path)
    union_keywords_2_dict=data_process.read_data_from_json(two_after_quchong_union_keywords_result_dict_json_path)

    for key, value in lda_json_data.items():
        if value not in result_keys:
            continue
        if key in union_keywords_1_dict.keys() and union_keywords_1_dict[key]==value:
            intersection_keywords_result_dict[key]=value

        if key in union_keywords_2_dict.keys() and union_keywords_2_dict[key]==value:
            two_intersection_keywords_result_dict[key]=value

    data_process.write_json(intersection_keywords_result_dict, "intersection_keywords_lda_result");
    data_process.write_json(two_intersection_keywords_result_dict, "2intersection_keywords_lda_result");

    print("交操作完成")
    return


def generate_tiyu_data():
    after_quchong_union_keywords_result_dict = data_process.read_data_from_json("key_words_result_data/after_quchong_union_keywords_result_dict.json")
    two_intersection_keywords_lda_result = data_process.read_data_from_json("intersection/2intersection_keywords_lda_result.json")

    finaly_intersection_result_dict={}
    result_keys = ["娱乐", "游戏", "体育"]
    for key,value in after_quchong_union_keywords_result_dict.items():
        if value !="体育":
            continue
        if  key not in two_intersection_keywords_lda_result.keys():
            finaly_intersection_result_dict[key]=value

        elif key  in two_intersection_keywords_lda_result.keys() and two_intersection_keywords_lda_result[key]=="体育":
            finaly_intersection_result_dict[key] = value

    data_process.write_json(finaly_intersection_result_dict, "tiyu_finaly_intersection_result_dict");

    for key, value in two_intersection_keywords_lda_result.items():
        if value!="体育":
            finaly_intersection_result_dict[key]=value

    data_process.write_json(finaly_intersection_result_dict, "three_finaly_intersection_result_dict");

    print("完成体育类的数据生成以及新三类的构造")

    return

import random
def sample_generate_1000_3_data():
    print("开始生成1000*3 三个新的类的标签数据")
    three_data_dict= data_process.read_data_from_json(
        "intersection/three_finaly_intersection_result_dict.json")


    tiyu_list=[]
    youxi_list = []
    yule_list = []

    for key,value in three_data_dict.items():
        if value=="体育":
            tiyu_list.append(key)
        elif value=="游戏":
            youxi_list.append(key)
        else:
            yule_list.append(key)

    tiyu_list_1000=random.sample(tiyu_list, 1000)
    youxi_list_1000 = random.sample(youxi_list, 1000)
    yule_list_1000 = random.sample(yule_list, 1000)

    print(np.array(tiyu_list_1000).shape)
    print(np.array(youxi_list_1000).shape)
    print(np.array(yule_list_1000).shape)
    three_new_class_1000_data_dict={}
    for key in tiyu_list_1000:
        three_new_class_1000_data_dict[key]="体育"

    for key in youxi_list_1000:
        three_new_class_1000_data_dict[key]="游戏"

    for key in yule_list_1000:
        three_new_class_1000_data_dict[key]="娱乐"

    data_process.write_json(three_new_class_1000_data_dict, "three_new_class_1000_data_dict");
    print("生成1000*3 三个新的类的标签数据OK")
    return


def intersection_bert_lda_result():
    print("开始进行交操作，生成新的三个类别分类结果")
    intersection_bert_lda_result_dict={}
    two_intersection_keywords_result_dict={}
    result_keys = ["娱乐", "游戏", "体育"]

    unlabel_result_dict_path = "unlabel_result_dict.json"
    keywords_result_dict_path="key_words_result_data/or_or_unlabeled_data_after_keywords_classed_data.json"

    unlabel_result_json_data=data_process.read_data_from_json(unlabel_result_dict_path)
    lda_json_data = data_process.read_data_from_json(lda_result_path)
    keywords_json_data=data_process.read_data_from_json(keywords_result_dict_path)

    union_keywords_1_dict=data_process.read_data_from_json(after_quchong_union_keywords_result_dict_json_path)
    union_keywords_2_dict=data_process.read_data_from_json(two_after_quchong_union_keywords_result_dict_json_path)

    for key, value in keywords_json_data.items():
        if key in unlabel_result_json_data.keys() and unlabel_result_json_data[key]==value:
            intersection_bert_lda_result_dict[key]=value

    data_process.write_json(intersection_bert_lda_result_dict, "intersection_bert_or_orkeywords_result_dict");

    print("交操作完成")
    return







if __name__=="__main__":
    intersection_bert_lda_result()
    print()







