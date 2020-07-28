from sklearn.model_selection import train_test_split
import pandas as pd 
import jieba
import numpy
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
label_dic = {'others':0,'news_agriculture':1, 'news_house':2, 'news_edu':3, 'news_sports':4, 'news_tech':5, 'stock':6, 'news_travel':7, 'news_car':8, 'news_entertainment':9, 'news_world':10, 'news_story':11, 'news_culture':12, 'news_military':13, 'news_finance':14, 'news_game':15}

def read_data(path):
    data_dic = {'label':[],'content':[]}
    with open(path,'r') as f:
        for line in f:
            info = line.split('_!_')
            if len(info) < 4: continue
            data_dic['label'].append(label_dic[info[2]])
            data_dic['content'].append(info[3])
    data = pd.DataFrame.from_dict(data_dic)  
    return data

def gen_small_data(data):
    #data = read_data('./toutiao_cat_data.txt')
    data_list = []
    for label in label_dic.values():
        label_data = data.loc[data['label'] == label]
        temp = label_data.sample(int(len(label_data)/10),random_state=1)
        data_list.append(temp)
    small_data = pd.concat(data_list)
    return small_data

def preprocess(content,label):
    #data = read_data(path)
    cw = lambda x: list(jieba.cut(x))
    words = content.apply(cw)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    vocab = tokenizer.word_index
    x_train,x_test,y_train,y_test = train_test_split(words,label,test_size=0.2)
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    x_train_padded_seqs = pad_sequences(x_train_word_ids,maxlen=50)
    x_test_padded_seqs = pad_sequences(x_test_word_ids,maxlen=50)
    #print(x_train_padded_seqs[0])
    return x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab

def gen_label_map(data=None,classes=None):
    if data is not None:
        classes = list(set(data['label']))
    cls_map = {'label':['others'],'id':[0]}
    for i,clss in enumerate(classes):
        cls_map['label'].append(clss)
        cls_map['id'].append(i+1)
    df = pd.DataFrame(cls_map)
    df.to_csv('./cnn/datasets/label_map.csv',index=False)
    return dict((k,v) for k,v in zip(df.label,df.id))

def train_w2v(path):
    data = read_data(path)
    cw = lambda x: list(jieba.cut(x))
    words = data['content'].apply(cw)
    #print(words[0])
    model = Word2Vec(words,size=300)
    #print(model.wv['你好'])
    #print(len(model.wv['你好']))
    model.save('toutiao_cat_w2v.pkl')


if __name__=='__main__':
    #data = read_data('./toutiao_cat_data.txt')
    #label_dic = gen_label_map(data=data)
    #data['label'] = [label_dic[l] for l in data.label]
    #x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data)
    #print(len(x_train_padded_seqs))
    #print(len(x_test_padded_seqs))
    #print(x_test_padded_seqs)
    train_w2v('./cnn/datasets/toutiao_cat_data.txt')