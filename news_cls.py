from utils.read_data import read_data_toutiao,read_data_weibo
from utils.utils import preprocess,gen_small_data,gen_label_map,get_embedding
from cnn.basic_CNN import CNN_model
from cnn.simple_textCNN import simple_textCNN
from cnn.textCNN import textCNN

def use_small_data_with_basicCNN():
    data = read_data_toutiao('./datasets/toutiao/toutiao_cat_data.txt')
    data = gen_small_data(data)
    x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data['content'],data['label'])
    CNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab)

def use_total_data_with_CNN():
    data = read_data_toutiao('./datasets/toutiao/toutiao_cat_data.txt')
    #data = gen_small_data(data)
    x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data['content'],data['label'])
    CNN_model(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab)

def use_small_data_with_simple_textCNN():
    data = read_data_toutiao('./datasets/toutiao/toutiao_cat_data.txt')
    data = gen_small_data(data)
    x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data['content'],data['label'])
    simple_textCNN(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab)

def use_total_data_with_textCNN():
    data = read_data_toutiao('./datasets/toutiao/toutiao_cat_data.txt')
    #data = gen_small_data(data)   
    x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,vocab = preprocess(data['content'],data['label'])
    embedding_matrix = get_embedding('./datasets/toutiao/toutiao_cat_w2v.pkl',vocab)
    textCNN(x_train_padded_seqs,y_train,x_test_padded_seqs,y_test,embedding_matrix,vocab)

if __name__=="__main__":
    use_small_data_with_basicCNN()
    use_total_data_with_CNN()
    use_small_data_with_simple_textCNN()
    use_total_data_with_textCNN()

