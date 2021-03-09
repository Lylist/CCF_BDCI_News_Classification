# -*- coding: utf-8 -*-
import pandas as pd
import os

f = pd.read_csv("people.csv",names=['id','content','class'])
f2 = open("leibie.txt","a",encoding="utf-8")
print(len(f))
print(f["content"][0])

for i in range(len(f)):
    print(f["content"][i])
    print("0: '财经', 1: '房产', 2: '家居', 3: '教育', 4: '科技', 5: '时尚', 6: '时政',7:'游戏',8:'娱乐',9:'体育'")
    c = input()
    if c == '0':
        f2.write(str(f["id"][i])+',财经\n')
    elif c == '1':
        f2.write(str(f["id"][i])+',房产\n')
    elif c == '2':
        f2.write(str(f["id"][i])+',家居\n')
    elif c == '3':
        f2.write(str(f["id"][i])+',教育\n')
    elif c == '4':
        f2.write(str(f["id"][i])+',科技\n')
    elif c == '5':
        f2.write(str(f["id"][i])+',时尚\n')
    elif c == '6':
        f2.write(str(f["id"][i])+',时政\n')
    elif c == '7':
        f2.write(str(f["id"][i])+',游戏\n')
    elif c == '8':
        f2.write(str(f["id"][i])+',娱乐\n')
    elif c == '999':
        f2.close()
        break
    else:
        f2.write(str(f["id"][i])+',体育')
    print("\033c", end="")
f2.close()
print('done')