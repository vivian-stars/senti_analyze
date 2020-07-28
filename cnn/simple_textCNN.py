from keras.models import Sequential,Model
from keras.layers import Embedding,Conv1D,MaxPooling1D,Flatten,Dropout,BatchNormalization,Dense,concatenate,Input
from keras.utils import to_categorical
import keras
from sklearn import metrics
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from read_data_toutiao import read_data,preprocess,gen_small_data

#import tensorflow as tf

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)


def simple_textCNN(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab):
    main_input = Input(shape=(50,),dtype='float64')
    #词嵌入，使用预训练的词向量
    embedder = Embedding(len(vocab) + 1,300,input_length=50,trainable=False)
    embed = embedder(main_input)
    #词窗大小分别为3，4，5
    cnn1 = Conv1D(256,3,padding='same',strides=1,activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = Conv1D(256,4,padding='same',strides=1,activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = Conv1D(256,5,padding='same',strides=1,activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=46)(cnn3)
    #合并三个模型的输出向量
    cnn = concatenate([cnn1,cnn2,cnn3],axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(16,activation='softmax')(drop)
    model = Model(inputs=main_input,outputs = main_output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    one_hot_labels = to_categorical(y_train,num_classes=16)
    model.fit(x_train_padded_seqs,one_hot_labels,batch_size=1000,epochs=1)
    
    result = model.predict(x_test_padded_seqs)
    y_predict = np.argmax(result,axis=1)
    #y_predict = list(map(str,result_labels))
    print(model.summary())
    print('准确率：',metrics.accuracy_score(y_test,y_predict))
    print('平均f1-score:',metrics.f1_score(y_test,y_predict,average='weighted'))

#if __name__=="__main__":
#    data = read_data('./cnn/datasets/toutiao_cat_data.txt')
#    data = gen_small_data(data)
#    x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data['content'],data['label'])
#    simple_textCNN(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab)
