from keras.layers import Dropout
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, LeakyReLU, Dense
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import euclidean_distances as eucl
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.optimizers import Adam
import scipy.stats
import os
from tensorflow.keras import Model



class AttentionLayer(Layer):
    def __init__(self, in_dim, num_neighbors):
        # in_dim: 为样本的原始长度
        super(AttentionLayer, self).__init__()
        # print('Using Concate Method')
        self.in_dim = in_dim
        self.linear = self.add_weight(shape=(self.in_dim, self.in_dim),
                                      initializer=initializers.glorot_uniform)

        self.e_for_neighors = self.add_weight(shape=(self.in_dim, 1),
                                              initializer=initializers.glorot_uniform)
        self.bias = self.add_weight(shape=(num_neighbors,1),
                                    initializer='zeros')


    def call(self, para_neighbors, para_nei_labels):

        para_neighbors = LeakyReLU(0.2)(tf.matmul(para_neighbors, self.linear))
        e_list = tf.matmul(para_neighbors, self.e_for_neighors)
        e_list = LeakyReLU(0.2)(tf.add(e_list, self.bias))
        # e_list_n = tf.slice(e_list, [0,1,0],[e_list.shape[0],e_list.shape[1]-1,e_list.shape[2]])
        # e_list_n = e_list[1:]
        alpha_list = tf.nn.softmax(e_list, axis=1)  # 归一化后的注意力系数 [n_nei, 1]
        alpha_list = tf.reshape(alpha_list, (-1, alpha_list.shape[2], alpha_list.shape[1]))
        # para_nei_labels = tf.slice(para_nei_labels, [0, 1, 0], [para_nei_labels.shape[0], para_nei_labels.shape[1] - 1, para_nei_labels.shape[2]])
        pred_labels = tf.matmul(alpha_list, para_nei_labels)
        pred_labels = tf.squeeze(pred_labels, axis=1)
        return pred_labels


class model(Model):
    def __init__(self, in_dim, num_neighbors):
        super(model, self).__init__()
        self.dense = Dense(int(in_dim), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00000001))
        self.drop = Dropout(0.6)
        self.att_layer = AttentionLayer(in_dim, num_neighbors)

    def call(self, data):
        x = self.dense(data[0])
        x = self.drop(x)
        return self.att_layer(x, data[1])


# 分隔训练集和测试集
def get_index(num_sample, para_k):
    """
    Get the training set index and test set index.
    :param
        para_k:
            The number of k-th fold.
    :return
        ret_tr_idx:
            The training set index, and its type is dict.
        ret_te_idx:
            The test set index, and its type is dict.
    """
    temp_rand_idx = np.random.permutation(num_sample)

    temp_fold = int(np.ceil(num_sample / para_k))
    ret_tr_idx = {}
    ret_te_idx = {}
    for i in range(para_k):
        temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx[i] = temp_tr_idx
        ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
    return ret_tr_idx, ret_te_idx

def find_neighbors(train_samples, train_labels, test_samples, num_nei):
    train_dis_matrix = eucl(train_samples, train_samples)
    for i in range(len(train_samples)):
        train_dis_matrix[i, i] = np.inf
    # 获得每个训练集样本的邻居索引
    train_nei_idx = [] #邻居索引数组
    for row in train_dis_matrix:
        nei_idx = np.argsort(row)[:num_nei]#从小到大排列
        train_nei_idx.append(nei_idx)
    train_nei_idx = np.array(train_nei_idx)
    # 获得每个训练集样本的邻居
    train_nei = [] #邻居
    for idx in train_nei_idx:
        train_nei.append(train_samples[idx])
    train_nei = np.array(train_nei)
    train_nei_labels = []
    for idx in train_nei_idx:
        train_nei_labels.append(train_labels[idx])
    train_nei_labels = np.array(train_nei_labels)
    # 获得测试集的邻居索引
    test_dis_matrix = eucl(test_samples, train_samples)
    test_nei_idx = []
    for row in test_dis_matrix:
        idx = np.argsort(row)[:num_nei]
        test_nei_idx.append(idx)
    test_nei_idx = np.array(test_nei_idx)
    test_nei, test_nei_labels = [], []
    for idx in test_nei_idx:
        test_nei.append(train_samples[idx])
        test_nei_labels.append(train_labels[idx])
    test_nei, test_nei_labels = np.array(test_nei), np.array(test_nei_labels)
    # print(train_nei.shape, train_nei_labels.shape, test_nei.shape, test_nei_labels.shape)
    return train_nei, train_nei_labels, test_nei, test_nei_labels

#带交叉验证的run
def data_cv(path, num_neighbors, para_k):
    data = loadmat(path)
    features = data['features']
    in_dim = len(features[0])
    labels = data['labels']
    n_class = len(labels[0])
    num_sample = len(features)
    idx = np.arange(num_sample)

    np.random.shuffle(idx)

    train_idx_list, test_idx_list = get_index(num_sample=num_sample, para_k=para_k)

    euclidean_list, srensen_list, squaredx_list, kl_list, intersection_list, fidelity_list = [], [], [], [], [], []

    for i in range(para_k):
        train_idx = train_idx_list[i]
        test_idx = test_idx_list[i]

        # 得到训练数据
        train_sample = features[train_idx]
        train_labels = labels[train_idx]
        test_sample = features[test_idx]
        test_labels = labels[test_idx]

        # 得到测试数据
        train_neighbors, train_neighbor_labels, test_neighbors, test_neighbor_labels = find_neighbors(train_sample,train_labels,test_sample,num_neighbors)
        #
        model_1 = model(in_dim=in_dim, num_neighbors=num_neighbors)
        model_1.compile(optimizer=Adam(lr=0.0001),
                        loss=KLDivergence())
        model_1.fit(x=[train_neighbors, train_neighbor_labels],
                    y=train_labels,
                    validation_data=([test_neighbors, test_neighbor_labels], test_labels),
                    batch_size=1,
                    epochs=50,
                    verbose=1)

        pred_labels = model_1.predict(x=[test_neighbors, test_neighbor_labels], batch_size=1)

        # kl_ = kl_loss(pred_labels, test_labels)

        euclidean_, srensen_, squaredx_, kl_, intersection_, fidelity_ = evaluate(pred_labels, test_labels)

        euclidean_list.append(euclidean_)
        srensen_list.append(srensen_)
        squaredx_list.append(squaredx_)
        kl_list.append(kl_)
        intersection_list.append(intersection_)
        fidelity_list.append(fidelity_)

    print('KL: ave: %lf  std: %lf' % (np.mean(kl_list), np.std(kl_list)))
    print('Euclidean: ave: %lf  std: %lf' % (np.mean(euclidean_list), np.std(euclidean_list)))
    print('Srensen: ave: %lf  std: %lf' % (np.mean(srensen_list), np.std(srensen_list)))
    print('Squared X: ave: %lf  std: %lf' % (np.mean(squaredx_list), np.std(squaredx_list)))
    print('intersection: ave: %lf  std: %lf' % (np.mean(intersection_list), np.std(intersection_list)))
    print('fidelity: ave: %lf  std: %lf' % (np.mean(fidelity_list), np.std(fidelity_list)))

#评价指标
def evaluate(pred, val):
    euclidean_ = euclidean_loss(pred, val)
    srensen_ = srensen_loss(pred, val)
    squaredx_ = squaredx_loss(pred, val)
    kl_ = kl_loss(pred, val)
    intersection_ = intersection_loss(pred, val)
    fidelity_ = fidelity_loss(pred, val)

    return euclidean_, srensen_, squaredx_, kl_, intersection_, fidelity_

def kl(P, Q):
    return scipy.stats.entropy(P, Q)


# kl散度loss
def kl_loss(pred, val):
    sum_ = 0
    for i, pred_i in enumerate(pred):
        sum_ += kl(val[i], pred_i)
    return sum_ / len(pred)


# 欧式loss
def euclidean_loss(pred, val):
    sum_ = 0
    for i in range(len(pred)):
        sum_ += np.sqrt(np.sum(np.power((pred[i] - val[i]), 2)))
    return sum_ / len(pred)


# srensen
def srensen_loss(pred, val):
    sum_ = 0
    for i in range(len(pred)):
        sum_ += np.sum(np.abs(pred[i] - val[i])) / np.sum(np.abs(pred[i] + val[i]))
    return sum_ / len(pred)


def squaredx_loss(pred, val):
    sum_ = 0
    for i in range(len(pred)):
        sum_ += np.sum((pred[i] - val[i]) ** 2 / (pred[i] + val[i]))
    return sum_ / len(pred)


def intersection_loss(pred, val):
    sum_ = 0
    for i in range(len(pred)):
        sum__ = 0
        for j in range(len(pred[i])):
            sum__ += np.min((pred[i][j], val[i][j]))
        sum_ += sum__
    return sum_ / len(pred)


def fidelity_loss(pred, val):
    sum_ = 0
    for i in range(len(pred)):
        sum_ += np.sum(np.sqrt(pred[i] * val[i]))
    return sum_ / len(pred)



if __name__ == '__main__':

     data_cv('datasets/Yeast_alpha.mat', 73, 5)
     print("Yeast_alpha.mat")

    # data_cv('datasets/Yeast_cold.mat', 73, 5)
    # print("Yeast_cold.mat")
    # # # #
    # data_cv('datasets/Yeast_dtt.mat', 73,5)
    # print("Yeast_dtt.mat")
    #
    # data_cv('datasets/Yeast_heat.mat', 73,5)
    # print("Yeast_heat.mat")
    #
    # data_cv('datasets/Yeast_spoem.mat', 73,5)
    # print("Yeast_spoem.mat")
    #
    # data_cv('datasets/Human_Gene.mat', 73,5)
    # print("Human_Gene.mat")
    # # #
    # data_cv('datasets/SJAFFE.mat', 73,5 )
    # print("SJAFFE.mat")
    #
    # data_cv('datasets/SBU_3DFE.mat', 73,5)
    # print("SBU_3DFE.mat")

    # print("Movie.mat")
    # data_cv('datasets/Movie.mat', 73,5)




