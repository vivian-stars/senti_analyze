
import pandas as pd 
label_dic = {'others':0,'news_agriculture':1, 'news_house':2, 'news_edu':3, 'news_sports':4, 'news_tech':5, 'stock':6, 'news_travel':7, 'news_car':8, 'news_entertainment':9, 'news_world':10, 'news_story':11, 'news_culture':12, 'news_military':13, 'news_finance':14, 'news_game':15}

def read_data_toutiao(path):
    data_dic = {'label':[],'content':[]}
    with open(path,'r') as f:
        for line in f:
            info = line.split('_!_')
            if len(info) < 4: continue
            data_dic['label'].append(label_dic[info[2]])
            data_dic['content'].append(info[3])
    data = pd.DataFrame.from_dict(data_dic)  
    return data

def read_data_weibo(path):
    data = pd.read_csv(path)
    return data