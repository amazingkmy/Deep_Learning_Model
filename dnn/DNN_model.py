import sys
import os
import tensorflow as tf

class DNN():
    def __init__(self):
        self.X_tr = None
        self.Y_tr = None
        self.sess = None
        self.params = None
        self.embed_model=None
    def load_params(self, params_file, model_type):
        with open(params_file, "r") as f:
            w = f.read().replace(' ', '').strip().split('\n')
            params_data=[var for var in w if w is not '#']

        # CNN : d_rate,	row_size,	col_size, filter_vec, DNN_vec
        if model_type is "CNN":
            result = [p if ',' not in p else [int(data) for data in p.split(',')] for p in params_data[0].split('\t')]
        self.params = result
        return True
    
    def model_train(self, model_type, X, Y):
        if model_type is "CNN":
            make_CNN_model()


    def make_CNN_model(Y_num):
        [dropout, row_size, col_size, dnn_vec, filter_size] = self.params
        row_size, col_size = int(row_size), int(col_size)

        self.X = tf.placeholder(float32, [None, row_size, col_size], name='X')
        self.Y = tf.placeholder(float32, [None, Y_num], name='Y')
        self.Dropout = tf.placeholder(float32, None, name='Drop')

        # row_size * col_size를 row_size*col_size*1 로 변환? 일반적인 CNN의 설명에 이게 작성되어 넣었지만, 과연 이게 의미가 있을까?
        cnn_net = tf.reshape(self.X, [-1, row_size, col_size, 1])

        W1 = tf.variable(tf.random_normal(filter_size))
        L1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(cnn_net, W1, strides=[1,1,1,1], padding='SAME')), ksize=[1,5,5,1], strides=[1,5,5,1], padding='SAME')

        W2 = tf.variable(tf.random_normal())

