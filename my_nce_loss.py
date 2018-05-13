# -- coding: UTF-8 --
"""实现负采样，功能类似于tf.nn.nce_loss"""
import numpy as np
import tensorflow as tf


def negative_sampling(num_neg_sam, vocabulary_size, pos_sam, p):
    """
    从分布p中进行负采样
    :param num_neg_sam: 需要采样的负样本数量
    :param vocabulary_size: 词汇表大小
    :param pos_sam: 正样本序号，用于计算期望采样数量
    :param p: 负采样概率
    :return: 负样本序号neg_sam[num_neg_sam], 正样本和负样本的采样数量的期望e_pos[num_pos_sam, 1], e_neg[num_neg_sam]
    """
    assert num_neg_sam <= vocabulary_size

    neg_sam = tf.py_func(np.random.choice, [[i for i in range(vocabulary_size)], num_neg_sam, False, p], tf.int32)
    neg_sam = tf.convert_to_tensor(neg_sam)

    num_pos_sam = pos_sam.get_shape().as_list()[0]  # 正样本数量，即batch_size
    e_pos = tf.expand_dims(tf.gather(p, pos_sam), 1) * num_neg_sam
    e_neg = tf.gather(p, neg_sam) * num_pos_sam
    return neg_sam, e_pos, e_neg


def calc_logits_and_labels(context_weights, context_biases, input_embeddings, pos_sam, neg_sam_list):
    """
    计算cross_entropy_loss函数的输入值
    :param context_weights: 单词作为context时的嵌入向量
    :param context_biases: 添加在内积项上的偏置
    :param input_embeddings: input单词的嵌入向量
    :param pos_sam: input单词的真context序号
    :param neg_sam_list: negtive_sampling函数生成的采样结果，(neg_sam, e_pos, e_neg)
    :return: logits和labels，大小均为[batch_size, num_neg_sam+1]，直接作为交叉熵损失函数的输入
    """
    neg_sam, e_pos, e_neg = neg_sam_list  # 解析参数

    pos_weights = tf.nn.embedding_lookup(context_weights, pos_sam)  #查找真假context样本的向量和偏置
    neg_weights = tf.nn.embedding_lookup(context_weights, neg_sam)
    pos_biases = tf.gather(context_biases, pos_sam)
    neg_biases = tf.gather(context_biases, neg_sam)

    logits_pos = tf.reduce_sum(tf.multiply(input_embeddings, pos_weights), axis=1, keep_dims=True) \
                 + tf.expand_dims(pos_biases, 1)  #[batch_size, 1]
    logits_neg = tf.matmul(input_embeddings, neg_weights, transpose_b=True) + neg_biases  #[batch_size, num_neg_sam]
    logits_pos -= tf.log(e_pos)
    logits_neg -= tf.log(e_neg)

    labels_pos = tf.ones_like(logits_pos, dtype=tf.int32)
    labels_neg = tf.zeros_like(logits_neg, dtype=tf.int32)

    logits = tf.concat([logits_pos, logits_neg], -1)  # [batch_size, num_neg_sam+1]
    labels = tf.concat([labels_pos, labels_neg], -1)
    return logits, labels


def cross_entropy_loss(logits, labels):
    """
    交叉熵损失，为防止指数项的上溢出，使用公式max(x, 0) - x * z + log(1 + exp(-abs(x)))
    :param logits: tensor
    :param labels: 与logits相同形状，元素为0或1，代表标号
    :return: 与logits相同大小，logits和labels的逐元素交叉熵损失
    """
    return tf.maximum(logits, 0) - logits * tf.cast(labels, tf.float32) + tf.log(1 + tf.exp(-tf.abs(logits)))


def nce_loss(num_neg_sam, vocabulary_size, context_weights, context_biases, input_embeddings, train_contexts, p):
    """
    计算NCE损失
    :param num_neg_sam: 需要采样的负样本数量
    :param vocabulary_size: 词汇表大小
    :param context_weights: 单词作为context时的嵌入向量
    :param context_biases: 添加在内积项上的偏置
    :param input_embeddings: input单词的嵌入向量
    :param train_contexts: input单词的真context序号，placeholder
    :param p: 负采样概率
    :return: [batch_size]，代表每一对样本(input,context)的NCE损失
    """
    pos_sam = train_contexts[:, 0]  #真context序号，[batch_size]
    neg_sam_list = negative_sampling(num_neg_sam, vocabulary_size, pos_sam, p)  #负采样

    logits, labels = calc_logits_and_labels(context_weights, context_biases, input_embeddings, pos_sam, neg_sam_list)
    ce_loss = cross_entropy_loss(logits, labels)  # 交叉熵损失

    return tf.reduce_sum(ce_loss, axis=1, keep_dims=True)