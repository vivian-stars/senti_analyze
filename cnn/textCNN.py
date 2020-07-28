from keras.models import Sequential,Model
from keras.layers import Embedding,Conv1D,MaxPooling1D,Flatten,Dropout,BatchNormalization,Dense,concatenate,Input
from keras.utils import to_categorical
from sklearn import metrics
from gensim.models import Word2Vec
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from read_data_toutiao import read_data,preprocess,gen_small_data

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)

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
 
#构建TextCNN模型
def textCNN(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embedding_matrix,vocab):
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(50,), dtype='float64')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(len(vocab) + 1, 300, input_length=50, weights=[embedding_matrix], trainable=False)
    #embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=50)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=49)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=48)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(16, activation='softmax')(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
    one_hot_labels = to_categorical(y_train, num_classes=16)  # 将标签转换为one-hot编码
    model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=60)
    #y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
    result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
    y_predict = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    #y_predict = list(map(str, result_labels))
    print(model.summary())
    print('准确率', metrics.accuracy_score(y_test, y_predict))
    print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))

#if __name__=="__main__":
#    data = read_data('./cnn/datasets/toutiao_cat_data.txt')
#    #data = gen_small_data(data)   
#    x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data['content'],data['label'])
#    embedding_matrix = get_embedding('./cnn/datasets/toutiao_cat_w2v.pkl',vocab)
#    textCNN(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embedding_matrix,vocab)