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
def train_lda_and_save(lda_name,documents):
    """
    :param lda_name: 保存lda名字
    :param documents: 训练lda的文档数组
    :return:          lda模型
    """
    # ---------------------------------开始训练LDA模型---------------------------------------
    print("开始训练LDA模型")
    topic = Topic(cwd=os.getcwd())  # 构建词典dictionary
    topic.create_dictionary(documents=documents)  # 根据documents数据，构建词典空间
    topic.create_corpus(documents=documents)  # 构建语料（将文本转为文档-词频矩阵）
    topic.train_lda_model(n_topics=10, epochs=20, fname=lda_name)  # 指定n_topic ，构建LDA话题模型
    print("完成lda模型的训练 并存储")
    #topic.visualize_lda()
    return topic


def load_lda_model(documents,lda_path="output/unlabeled/model/labeled_data_lda_model",dict_path="output/unlabeled/dictionary.dict"):
    """
    :param lda_path: lda存储路径
    :return: lda模型
    """
    print("开始加载训练好的lda model")
    topic=Topic(cwd=os.getcwd())
    topic.load_dictionary(dictpath=dict_path)
    topic.create_corpus(documents=documents)
    topic.load_lda_model(modelpath=lda_path)
    print("lda 模型加载完毕")
    return topic


def use_lda_classifi_data(lda_model_path,lda_dict_path):
    print("开始标记未标记数据......")
    documents, labels, content_dict_data, label_dict_data = data_process.get_data(True)
    # model=train_lda_and_save("unlabeled_data_lda_model",documents=documents)
    model = load_lda_model(documents, lda_path=lda_model_path, dict_path=lda_dict_path)
    after_lda_classed_data_dict={}
    start=0
    for document in documents:
        topics=model.get_document_topics(document)
        topics = sorted(topics, key=lambda x: x[1], reverse=True)
        topic=topics[0]#取概率最大的那个主题最为分类类别
        label=id_label_dict[topic[0]]
        after_lda_classed_data_dict[start]=label
        start+=1

    data_process.write_json(after_lda_classed_data_dict,"labeled_after_lda_classed_data")
    print("未标记数据标记成功。")
    return





if __name__=="__main__":

    use_lda_classifi_data(unlabeled_lda_model_path,unlabeled_dict_path)
    #documents, labels, content_dict_data, label_dict_data = data_process.get_data(False)
    # print(documents[0])
    # print(documents[1])

    #model=train_lda_and_save("unlabeled_data_lda_model",documents=documents)
    # model=load_lda_model(documents,lda_path=unlabeled_lda_model_path,dict_path=unlabeled_dict_path)
    # document=jieba.lcut("一年一度的春运即将拉开帷幕，在微博上一些网友呼吁无座火车票实行半价，得到众多网友的热烈响应，大家纷纷说“有道理”。人民日报官方微博等也对此予以关注，河南商报官方微博发起的投票显示，八成网友支持站票半价。(1月14日《河南商报》)　　在微博上，北京大学法学院教授贺卫方通过自己以前“很痛苦”的乘火车经历，认为“站票应卖三分之一价”。我倒没这么贪心，或者说不敢有如此奢望，但是给无座火车票打个折，无论如何都是应该的。　　春运人满为患，铁路运力不足，火车在一定限度内超员未尝不可，但这也意味着站票打折是需要铁道部单独面对并回应的民意，不应推诿和观望。　　“坐火车”与“站火车”是不可同日而语的，尤其是路途遥远的乘客，“站火车”简直是活受罪。换言之，无座乘客虽与有座乘客同处于一列火车上，但他们没有享受到应有的服务，即服务大打折扣，所以无座车票的价格也应该相应打折。乘客与铁路运输单位是平等的民事主体，乘客买票相当于与铁路运输单位订立合同，按照《合同法》的规定，“当事人应当遵循公平原则确定各方的权利和义务”。没有享受到应有服务的无座乘客，却要与有座乘客掏一样的车票钱，这是不公平的。　　以前，大多数列车的每个座位全程只卖给一名乘客，有些乘客中途下车后，运气好的无座乘客还能“捡”到座位；现在随着全国联网售票的推行，大多数列车实行全程对号入座，即沿途车站会将中途下车乘客留下的座位重新出售，这样，绝大多数无座乘客都要“一站到底”，在春运期间尤其是如此，他们的权益因此受到了损害。　　哪些人最可能买不到座位票而只能“站火车”呢？其中之一是农民工。农民工收入很低，有的干了一年还没拿到工钱，回家时兜里没有几个钱，他们是最盼望无座票打折的，也是对无座票不打折最有意见的，只是他们没有话语权。无论是从公平的角度，还是从关爱弱势群体的角度，无座火车票都应该打折，哪怕一张票为农民工省下百八十元，他们也会觉得“站火车”的辛苦是值得的，并会在心里感谢铁道部。　　微博上也有一些网友说，如果无座票实行半价，很多人岂不都要去买无座票？岂不造成列车更加拥挤？如此担忧纯属杞人忧天。一列火车当然先卖座位票，然后再卖无座票，而无座票的数量也是有限的。还有个别网友说，嫌无座票贵就别坐火车呀！这话说得毫无道理。我国铁路实行垄断经营，人们在票价上没有选择余地，只能被动接受。正因为实行垄断经营，人们无法用脚投票，垄断者更应该担负起责任，将票价制定得合理一些，而不能将垄断作为欺负消费者的工具。　　无座火车票降价问题，人们已经呼吁了好多年，铁道部该有所动作了。（作者　晏扬）责任编辑：hdwmn_ctt")# 预测document对应的话题
    # #print(model.get_document_topics(document))
    # topics=model.get_document_topics(document)
    # #topics=topics.sort(key=lambda t:t[1])
    # topics=sorted(topics,key=lambda x:x[1],reverse=True)
    #
    # for topic in topics:
    #     print(topic)
    #     print(topic[0])
    #     print(topic[1])
    # print(model.show_topics())
    # print(model.topic_distribution(raw_documents=documents))




def use_lda_generate_data(lda_model_path,dict_path):
    topic=load_lda_model(lda_model_path,dict_path=dict_path)

    # 准备document
    # document=jieba.lcut("体育游戏真有意思")
    # 预测document对应的话题
    # topic.get_document_topics(document)
    # 显示每种话题与对应的特征词之间的关系
    # topic.show_topics()
    # 话题额分布情况
    # topic.topic_distribution()
    # 可视化
    # topic.visualize_lda()

    return



#--------------------------------使用LDA模型------------------------------------------
#准备document
#document=jieba.lcut("体育游戏真有意思")
#预测document对应的话题
#topic.get_document_topics(document)
#显示每种话题与对应的特征词之间的关系
#topic.show_topics()
#话题额分布情况
#topic.topic_distribution()
#可视化
#topic.visualize_lda()

#存储与导入lda模型   默认是存储的
#topic2=Topic(cwd=os.getcwd())
#topic2.load_dictionary(dictpath="path")
#topic2.create_corpus(documents=documents)
#topic2.load_lda_model(modelpath="path")
