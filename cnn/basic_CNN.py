import os
#os.environ["PATH"] += os.pathsep + ''
from keras.models import Sequential
from keras.layers import Embedding,Conv1D,MaxPooling1D,Flatten,Dropout,BatchNormalization,Dense
from keras.utils import to_categorical,plot_model
from sklearn import metrics
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#构建CNN分类模型(LeNet-5)
#模型结构：嵌入-卷积池化*2-dropout-BN-全连接-dropout-全连接

def CNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab):
    model = Sequential()
    #使用embedding层将每个词编码转换为词向量
    model.add(Embedding(len(vocab) + 1,300,input_length=50))
    
    model.add(Conv1D(filters=256,kernel_size=5,padding='same'))
    model.add(MaxPooling1D(3,3,padding='same'))
    model.add(Conv1D(128,5,padding='same'))
    model.add(MaxPooling1D(3,3,padding='same'))
    model.add(Conv1D(64,3,padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    one_hot_labels = to_categorical(y_train,num_classes=16)
    #print('one_hot_labels:{}'.format(one_hot_labels))
    #print(len(one_hot_labels[0]))
    model.fit(x_train_padded_seqs,one_hot_labels,epochs=3,batch_size=800)
    #score2 = model.evaluate(x_test_padded_seqs,y_test,verbose=0)
    #print('score2:{}'.format(score2))
    y_predict = model.predict_classes(x_test_padded_seqs)
    #print('y_predict1:{}'.format(y_predict))
    #y_predict = list(map(str,y_predict))
    #print('y_predict2:{}'.format(y_predict[0]))
    #print('y_test:{}'.format(y_test))
    #print(type(y_test))
    #print(np.array(y_test))
    print(model.summary())
    print('准确率：',metrics.accuracy_score(y_test,y_predict))
    print('平均f1-score:',metrics.f1_score(y_test,y_predict,average='weighted'))

#if __name__=="__main__":
#    data = read_data('./cnn/datasets/toutiao_cat_data.txt')
#    #data = gen_small_data(data)
#    x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data['content'],data['label'])
#    CNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab)