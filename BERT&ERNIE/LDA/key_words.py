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

all_key_words_data_path=[and_and_unlabeled_data_after_keywords_classed_data_json_path
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

def resort_by_keywords(data_dict1,data_dict2):
    '''
    :param data_dict1:体育类比较少的dict
    :param data_dict2:体育类比较多的dict
    :return:
    '''
    print("开始按照关键字进行频率重排序.......")
    documents, labels, content_dict_data, label_dict_data = data_process.get_data(False)
    label_dict = {}

    yule_keywords_result_dict = {}
    youxi_keywords_result_dict = {}
    tiyu_keywords_result_dict = {}
    result_keys = ["娱乐", "游戏", "体育"]

    for key,value in data_dict1.items():
        document=content_dict_data[key]
        if value=="娱乐":
            num=document.count("娱乐")+1
            yule_keywords_result_dict[key]=num
        elif value=="游戏":
            num = document.count("游戏") + 1
            youxi_keywords_result_dict[key] = num
        else:
            num = document.count("体育") + 1
            tiyu_keywords_result_dict[key] = num

    for key,value in data_dict2.items():
        if value!="体育":
            continue
        if key not in yule_keywords_result_dict.keys() and key not in youxi_keywords_result_dict.keys() and key not in tiyu_keywords_result_dict.keys():
            document = content_dict_data[key]
            num = document.count("体育") + 1
            tiyu_keywords_result_dict[key] = num

    yule_keywords_result_dict=sorted(yule_keywords_result_dict.items(),key = lambda x:x[1],reverse = True)
    youxi_keywords_result_dict=sorted(youxi_keywords_result_dict.items(), key=lambda x: x[1], reverse=True)
    tiyu_keywords_result_dict=sorted(tiyu_keywords_result_dict.items(), key=lambda x: x[1], reverse=True)

    print("yule_keywords_result_dict: ", len(yule_keywords_result_dict))
    print("youxi_keywords_result_dict: ", len(youxi_keywords_result_dict))
    print("tiyu_keywords_result_dict: ", len(tiyu_keywords_result_dict))
    print("按照关键字key words分类并按照词频排序完成")


    data_process.write_json(yule_keywords_result_dict, "yule_keywords_result_dict_"+str(len(yule_keywords_result_dict)))
    data_process.write_json(youxi_keywords_result_dict, "youxi_keywords_result_dict_"+str(len(youxi_keywords_result_dict)))
    data_process.write_json(tiyu_keywords_result_dict, "tiyu_keywords_result_dict_"+str(len(tiyu_keywords_result_dict)))
    print("已保存成json数据格式")
    return


def calssfi_data_by_keywords():
    print("开始按照关键字进行分类")
    documents, labels, content_dict_data, label_dict_data = data_process.get_data(False)
    label_dict={}

    for i,content in enumerate(documents):
        flag=False
        for key,value in label_keywords_dict.items():
            if value[0] in content and value[1] in content or value[2] in content:#

                label_dict[i]=key
                flag=True
                break
        if flag==False:
            label_dict[i] = "UnKnow"
    print("按照关键字key words分类完成")
    data_process.write_json(label_dict, "and_unlabeled_data_after_keywords_classed_data")
    print("已保存成json数据格式")
    return

def union_keywords_result():
    print("开始进行并操作，生成新的三个类别分类结果")
    union_keywords_result_dict={}
    yule_keywords_result_dict={}
    youxi_keywords_result_dict = {}
    tiyu_keywords_result_dict = {}
    result_keys=["娱乐","游戏","体育"]
    for data_path in all_key_words_data_path:
        json_data=data_process.read_data_from_json(data_path)
        for key,value in json_data.items():
            if value not in result_keys:
                continue
            if value=="体育":
                tiyu_keywords_result_dict[key]="体育"
            elif value=="娱乐":
                yule_keywords_result_dict[key]="娱乐"
            else:
                youxi_keywords_result_dict[key]="游戏"
    #此步骤还没有去除重复
    # data_process.write_json(yule_keywords_result_dict, "1tiyu_keywords_result_dict");
    # data_process.write_json(youxi_keywords_result_dict, "youxi_keywords_result_dict");
    # data_process.write_json(tiyu_keywords_result_dict, "2tiyu_keywords_result_dict");

    #开始去除重复
    for key,value in tiyu_keywords_result_dict.items():
        if key not in yule_keywords_result_dict.keys() and key not in youxi_keywords_result_dict.keys():
            union_keywords_result_dict[key]=value

    for key,value in yule_keywords_result_dict.items():
        if key not in tiyu_keywords_result_dict.keys() and key not in youxi_keywords_result_dict.keys():
            union_keywords_result_dict[key]=value

    for key,value in youxi_keywords_result_dict.items():
        if key not in tiyu_keywords_result_dict.keys() and key not in yule_keywords_result_dict.keys():
            union_keywords_result_dict[key]=value

    data_process.write_json(union_keywords_result_dict, "1after_quchong_union_keywords_result_dict");
    print("去除重复ok")


import random
def generate_1000_3_new_class_data():
    print("开始生成1000*3 三个新的类的标签数据")
    three_data_dict = data_process.read_data_from_json("key_words_result_data/after_quchong_union_keywords_result_dict.json")

    tiyu_list = []
    youxi_list = []
    yule_list = []

    for key, value in three_data_dict.items():
        if value == "体育":
            tiyu_list.append(key)
        elif value == "游戏":
            youxi_list.append(key)
        else:
            yule_list.append(key)

    tiyu_list_1000 = random.sample(tiyu_list, 1000)
    youxi_list_1000 = random.sample(youxi_list, 1000)
    yule_list_1000 = random.sample(yule_list, 1000)

    print(np.array(tiyu_list_1000).shape)
    print(np.array(youxi_list_1000).shape)
    print(np.array(yule_list_1000).shape)
    three_new_class_1000_data_dict = {}
    for key in tiyu_list_1000:
        three_new_class_1000_data_dict[key] = "体育"

    for key in youxi_list_1000:
        three_new_class_1000_data_dict[key] = "游戏"

    for key in yule_list_1000:
        three_new_class_1000_data_dict[key] = "娱乐"

    data_process.write_json(three_new_class_1000_data_dict, "keywords_three_new_class_1000_data_dict");
    print("生成1000*3 三个新的类的标签数据OK")
    return




if __name__=="__main__":
    three_data_dict1 = data_process.read_data_from_json(
        "key_words_result_data/2after_quchong_union_keywords_result_dict.json")
    three_data_dict2 = data_process.read_data_from_json(
        "key_words_result_data/after_quchong_union_keywords_result_dict.json")
    resort_by_keywords(three_data_dict1,three_data_dict2)
    #generate_1000_3_new_class_data()
    print()







