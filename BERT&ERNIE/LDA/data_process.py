# coding=utf-8
# MF 2020.11.20
import numpy as np
import pandas as pd
import jieba
import re
import os
import json
import csv
import codecs
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from collections import Counter

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

labeled_data_file_path="AllData/labeled_data.csv"
unlabeled_data_file_path="AllData/unlabeled_data.csv"
labeled_content_dict_path="AllData/labeled_data_content_dict.json"
labeled_label_dict_path="AllData/labeled_data_labels_dict.json"
unlabeled_content_dict_path="AllData/unlabeled_data_content_dict.json"
stop_words_path="AllData/stop_words.txt"
external_train_data_path="AllData/cnews.train.txt"
external_test_data_path="AllData/cnews.test.txt"

test_data_path="AllData/test_data.csv"

#进行中文分词 输入一篇文章
def chineseSentenceSplit(string): #return [n]  n 为一句评论中分词个数
    #print(string)
    data = clean_str(string)
    seg_list = jieba.cut(data)  # 默认是精确模式
    tokens = [t for t in seg_list if len(t) > 1]  # 剔除单字
    stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='UTF-8').readlines()]
    segList = []
    for s in tokens:
        if s not in stopwords and s != '\t':
            s = clean_str(s)
            segList.append(s)
    return segList


# 句子清洗
def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\r\n", "", string)
    string = re.sub(r"\r", "", string)
    string = re.sub(r"\,", "", string)
    string = re.sub(r"\.", "", string)
    string = re.sub(r"\，", "", string)
    string = re.sub(r"\。", "", string)
    string = re.sub(r"\（", "", string)
    string = re.sub(r"\）", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\“", "", string)
    string = re.sub(r"\”", "", string)

    return string.strip()


def get_test_documents_from_csv(file_path=test_data_path):
    print("开始读取测试test数据并分词...")
    content_json = {}
    label_json = {}
    documents = []
    labels = []
    csvFile = open(file_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        documents.append(chineseSentenceSplit(item[1]))
        content_json[item[0]] = chineseSentenceSplit(item[1])
    csvFile.close()
    print(documents[0])
    print(np.array(documents).shape)
    print(np.array(labels).shape)
    write_json(content_json, "test_unlabeled_data_content_dict")
    print("分词处理结束......")
    return documents, labels, content_json, label_json




def get_documents_from_csv(file_path,labeled=True):
    content_json={}
    label_json={}
    documents=[]
    labels=[]
    csvFile = open(file_path, "r")
    reader = csv.reader(csvFile)

    if labeled==True:
        for item in reader:
            # 忽略第一行
            if reader.line_num == 1:
                continue
            documents.append(chineseSentenceSplit(item[2]))
            labels.append(item[1])
            content_json[item[0]] = chineseSentenceSplit(item[2])
            label_json[item[0]]=item[1]
        csvFile.close()
        print(documents[0])
        print(np.array(documents).shape)
        print(np.array(labels).shape)
        write_json(label_json,"labeled_data_labels_dict")
        write_json(content_json, "labeled_data_content_dict")
        return documents, labels, content_json, label_json
    else:
        for item in reader:
            # 忽略第一行
            if reader.line_num == 1:
                continue
            documents.append(chineseSentenceSplit(item[1]))
            content_json[item[0]] = chineseSentenceSplit(item[1])
        csvFile.close()
        print(documents[0])
        print(np.array(documents).shape)
        print(np.array(labels).shape)
        write_json(content_json, "unlabeled_data_content_dict")
        return documents, labels, content_json, label_json



'''
	writed by MF
    写入 json 函数
'''
def write_json(dic,name):
    jss = json.dumps(dic, ensure_ascii=False)
    json_sessions = open(name+".json", 'w')
    json_sessions.write(jss)
    json_sessions.close()
    print(name+".json OK")

'''
	writed by MF
    从json中获取相应的数据
'''
def read_data_from_json(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        data =json.load(f)
    f.close()
    return data


def transfer_jsondata_list(data):
    documents=[]
    for key,value in data.items():
        documents.append(value)
    return documents


def get_data(labeled=True):
    content_dict_data={}
    documents=[]
    label_dict_data={}
    labels=[]
    if labeled==False:
        content_file_path=unlabeled_content_dict_path
        content_dict_data = read_data_from_json(content_file_path)
        documents = transfer_jsondata_list(content_dict_data)
    else:
        content_file_path=labeled_content_dict_path
        label_file_path=labeled_label_dict_path
        content_dict_data = read_data_from_json(content_file_path)
        documents = transfer_jsondata_list(content_dict_data)
        label_dict_data=read_data_from_json(label_file_path)
        labels=transfer_jsondata_list(label_dict_data)

    return documents,labels,content_dict_data,label_dict_data


def analys_classed_data():
    json_data=read_data_from_json("AllData/labeled_after_lda_classed_data.json")
    label_num={}
    y_pred=[]
    y_true=[]
    for key,value in json_data.items():
        y_pred.append(label_id_dict[value])
        # if value in label_num.keys():
        #     label_num[value]+=1
        # else:
        #     label_num[value]=1

    documents, labels, content_dict_data, label_dict_data =get_data(True)
    for label in labels:
        y_true.append(label_id_dict[label])
    print(y_true)
    print(y_pred)

    print("准确率：",accuracy_score(y_true, y_pred))
    print("精确率：",precision_score(y_true, y_pred, average='micro'))
    print("召回率：",recall_score(y_true, y_pred, average='micro'))
    print("F1值：",f1_score(y_true, y_pred, average='weighted'))
    target_names = ["财经","时政","科技","游戏","娱乐","体育","时尚","家居","教育","房产"]
    print(classification_report(y_true, y_pred, target_names=target_names))


def find_key_words():
    print("开始对外部数据进行词频统计.......")
    documents, labels, content_dict_data, label_dict_data =read_txt_data() #get_data(False)
    #json_data = read_data_from_json("AllData/labeled_after_lda_classed_data.json")

    documents_0=[]
    documents_1=[]
    documents_2=[]
    documents_3 = []
    documents_4 = []
    documents_5 = []
    documents_6 = []
    documents_7 = []
    documents_8 = []
    documents_9 = []
    for i, value in label_dict_data.items():
        i=int(i)

        if label_id_dict[value]==0:
            documents_0.append(documents[i])
        elif label_id_dict[value]==1:
            documents_1.append(documents[i])
        elif label_id_dict[value] == 2:
            documents_2.append(documents[i])
        elif label_id_dict[value] == 3:
            documents_3.append(documents[i])
        elif label_id_dict[value] == 4:
            documents_4.append(documents[i])
        elif label_id_dict[value] == 5:
            documents_5.append(documents[i])
        elif label_id_dict[value] == 6:
            documents_6.append(documents[i])
        elif label_id_dict[value] == 7:
            documents_7.append(documents[i])
        elif label_id_dict[value] == 8:
            documents_8.append(documents[i])
        else :
            documents_9.append(documents[i])

    label_0_words_num={}
    label_1_words_num = {}
    label_2_words_num = {}
    label_3_words_num = {}
    label_4_words_num = {}
    label_5_words_num = {}
    label_6_words_num = {}
    label_7_words_num = {}
    label_8_words_num = {}
    label_9_words_num = {}


    for document in documents_0:
        words_dict=Counter(document)
        for key,value in words_dict.items():
            if key in label_0_words_num.keys():
                label_0_words_num[key]+=1
            else:
                label_0_words_num[key]=1

    for document in documents_1:
        words_dict=Counter(document)
        for key,value in words_dict.items():
            if key in label_1_words_num.keys():
                label_1_words_num[key]+=1
            else:
                label_1_words_num[key]=1

    for document in documents_2:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_2_words_num.keys():
                label_2_words_num[key] += 1
            else:
                label_2_words_num[key] = 1

    for document in documents_3:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_3_words_num.keys():
                label_3_words_num[key] += 1
            else:
                label_3_words_num[key] = 1

    for document in documents_4:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_4_words_num.keys():
                label_4_words_num[key] += 1
            else:
                label_4_words_num[key] = 1

    for document in documents_5:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_5_words_num.keys():
                label_5_words_num[key] += 1
            else:
                label_5_words_num[key] = 1

    for document in documents_6:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_6_words_num.keys():
                label_6_words_num[key] += 1
            else:
                label_6_words_num[key] = 1

    for document in documents_7:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_7_words_num.keys():
                label_7_words_num[key] += 1
            else:
                label_7_words_num[key] = 1

    for document in documents_8:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_8_words_num.keys():
                label_8_words_num[key] += 1
            else:
                label_8_words_num[key] = 1

    for document in documents_9:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_9_words_num.keys():
                label_9_words_num[key] += 1
            else:
                label_9_words_num[key] = 1

    label_0_words_num=sorted(label_0_words_num.items(), key=lambda x: x[1], reverse=True)
    label_1_words_num=sorted(label_1_words_num.items(), key=lambda x: x[1], reverse=True)
    label_2_words_num=sorted(label_2_words_num.items(), key=lambda x: x[1], reverse=True)
    label_3_words_num=sorted(label_3_words_num.items(), key=lambda x: x[1], reverse=True)
    label_4_words_num=sorted(label_4_words_num.items(), key=lambda x: x[1], reverse=True)
    label_5_words_num=sorted(label_5_words_num.items(), key=lambda x: x[1], reverse=True)
    label_6_words_num=sorted(label_6_words_num.items(), key=lambda x: x[1], reverse=True)
    label_7_words_num=sorted(label_7_words_num.items(), key=lambda x: x[1], reverse=True)
    label_8_words_num=sorted(label_8_words_num.items(), key=lambda x: x[1], reverse=True)
    label_9_words_num=sorted(label_9_words_num.items(), key=lambda x: x[1], reverse=True)
    print("外部数据词频统计完毕......")
    write_json(label_0_words_num,"external_label_0_words_num");
    write_json(label_1_words_num, "external_label_1_words_num");
    write_json(label_2_words_num, "external_label_2_words_num");
    write_json(label_3_words_num, "external_label_3_words_num");
    write_json(label_4_words_num, "external_label_4_words_num");
    write_json(label_5_words_num, "external_label_5_words_num");
    write_json(label_6_words_num, "external_label_6_words_num");
    write_json(label_7_words_num, "external_label_7_words_num");
    write_json(label_8_words_num, "external_label_8_words_num");
    write_json(label_9_words_num, "external_label_9_words_num");


def find_external_key_words():
    print("开始对外部数据进行词频统计.......")
    documents, labels, content_dict_data, label_dict_data = read_txt_data()  # get_data(False)
    # json_data = read_data_from_json("AllData/labeled_after_lda_classed_data.json")


    documents_3 = []
    documents_4 = []
    documents_5 = []

    for i, value in label_dict_data.items():
        i = int(i)
        if label_id_dict[value] == 3:
            documents_3.append(documents[i])
        elif label_id_dict[value] == 4:
            documents_4.append(documents[i])
        elif label_id_dict[value] == 5:
            documents_5.append(documents[i])
        else:
            continue
    label_3_words_num = {}
    label_4_words_num = {}
    label_5_words_num = {}
    for document in documents_3:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_3_words_num.keys():
                label_3_words_num[key] += 1
            else:
                label_3_words_num[key] = 1

    for document in documents_4:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_4_words_num.keys():
                label_4_words_num[key] += 1
            else:
                label_4_words_num[key] = 1

    for document in documents_5:
        words_dict = Counter(document)
        for key, value in words_dict.items():
            if key in label_5_words_num.keys():
                label_5_words_num[key] += 1
            else:
                label_5_words_num[key] = 1
    label_3_words_num = sorted(label_3_words_num.items(), key=lambda x: x[1], reverse=True)
    label_4_words_num = sorted(label_4_words_num.items(), key=lambda x: x[1], reverse=True)
    label_5_words_num = sorted(label_5_words_num.items(), key=lambda x: x[1], reverse=True)
    print("外部数据词频统计完毕......")
    write_json(label_3_words_num, "external_label_3_words_num");
    write_json(label_4_words_num, "external_label_4_words_num");
    write_json(label_5_words_num, "external_label_5_words_num");



def read_txt_data(filename=external_test_data_path):
    print("开始读取外部数据.......")
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    content_json = {}
    label_json = {}
    contents, labels = [], []
    start=0
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                if label !="游戏" and label!="娱乐" and label !="体育":
                    continue
                label_json[start] = label
                print(label)
                start+=1
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        word.extend(chineseSentenceSplit(blk))
                contents.append(word)
            except:
                pass
    print("外部数据读取完毕......")

    return contents,labels,content_json, label_json


def analys_keywords_classed_data():

    json_data=read_data_from_json("key_words_result_data/after_quchong_union_keywords_result_dict.json")

    label_num={}

    for key,value in json_data.items():
        if value in label_num.keys():
            label_num[value]+=1
        else:
            label_num[value] = 1

    print(label_num)

def read_txt_label():
    txt_file_path="unlabel_result.txt"
    unlabel_result_dict={}
    with codecs.open(txt_file_path, 'r', encoding='utf-8') as f:
        for i,line in enumerate(f):
            unlabel_result_dict[i]=id_label_dict[int(line.rstrip())]
    write_json(unlabel_result_dict,"unlabel_result_dict")

    return

if __name__=="__main__":
    #print("hhsadhdh"+str(1))
    analys_keywords_classed_data()
    #get_test_documents_from_csv()
    # data=read_data_from_json(unlabeled_content_dict_path)
    # documents=transfer_jsondata_list(data)
    # documents,labels,content_dict_data,label_dict_data=get_data(False)
    #
    # print(np.array(documents).shape)
    # print(documents[0])
    # print()
    # train_txt_file="AllData/cnews.train.txt"
    # labels, contents=read_txt_data(train_txt_file)
    # print(np.array(contents).shape)
    # print(contents[0])
    # print(labels[0])
    print()











def get_alldatsa():
    content_dict_data = {}
    documents = []
    label_dict_data = {}
    labels = []
    content_file_path = unlabeled_content_dict_path
    content_dict_data = read_data_from_json(content_file_path)
    documents_unlabeled = transfer_jsondata_list(content_dict_data)

    content_file_path = labeled_content_dict_path
    label_file_path = labeled_content_dict_path
    content_dict_data = read_data_from_json(content_file_path)
    documents = transfer_jsondata_list(content_dict_data)
    label_dict_data = read_data_from_json(label_file_path)
    labels = transfer_jsondata_list(label_dict_data)


    return documents, labels, content_dict_data, label_dict_data









