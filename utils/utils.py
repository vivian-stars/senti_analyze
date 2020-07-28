from sklearn.model_selection import train_test_split
import pandas as pd 
import jieba
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
label_dic = {'others':0,'news_agriculture':1, 'news_house':2, 'news_edu':3, 'news_sports':4, 'news_tech':5, 'stock':6, 'news_travel':7, 'news_car':8, 'news_entertainment':9, 'news_world':10, 'news_story':11, 'news_culture':12, 'news_military':13, 'news_finance':14, 'news_game':15}


def gen_small_data(data):
    '''
    取总数据的十分之一，作为调试代码的数据
    '''
    #data = read_data('./toutiao_cat_data.txt')
    data_list = []
    for label in label_dic.values():
        label_data = data.loc[data['label'] == label]
        temp = label_data.sample(int(len(label_data)/10),random_state=1)
        data_list.append(temp)
    small_data = pd.concat(data_list)
    return small_data

def preprocess(content,label):
    '''
    将文字转换成id
    content:训练数据
    label:对应标签
    '''
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

def gen_label_map(data=None,classes=None,path='datasets/label_map.csv'):
    '''
    生成类别与id的映射表
    data:所有的数据，dict或者是DataFrame
    classes:所有的类别
    path:映射表保存的位置
    '''
    if data is not None:
        classes = list(set(data['label']))
    cls_map = {'label':['others'],'id':[0]}
    for i,clss in enumerate(classes):
        cls_map['label'].append(clss)
        cls_map['id'].append(i+1)
    df = pd.DataFrame(cls_map)
    df.to_csv(path,index=False)
    return dict((k,v) for k,v in zip(df.label,df.id))

def train_w2v(words,name='w2v.pkl'):
    '''
    训练词向量，word2vec方式
    words:待训练的数据，已分词，格式如下：[['我','要','去'],[]]
    name:词向量保存的名字
    '''
    #data = read_data(path)
    #cw = lambda x: list(jieba.cut(x))
    #words = data['content'].apply(cw)
    model = Word2Vec(words,size=300)
    #model.save('toutiao_cat_w2v.pkl')
    model.save(name)

def get_embedding(w2v_path,vocab):
    w2v_model=Word2Vec.load(w2v_path)
    # 预训练的词向量中没有出现的词用0向量表示
    embedding_matrix = np.zeros((len(vocab) + 1, 300))
    for word, i in vocab.items():
        try:
            embedding_vector = w2v_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    return embedding_matrix