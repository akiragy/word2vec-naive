# -- coding: UTF-8 --
import numpy as np
import tensorflow as tf
import zipfile
import collections
import my_nce_loss


class Config(object):
    """配置文件"""
    def __init__(self):
        self.embedding_size = 128  # 嵌入向量的维度
        self.skip_window = 1  # 单侧窗口大小
        self.num_skip = 2  # 从一个窗口内采样的样本数量

        self.max_epoch = 100001  # 迭代次数
        self.print_loss_step = 2000  # 输出训练误差的间隔步长
        self.batch_size = 128  # 训练中的batch大小
        self.num_neg_sam = 64  # 一次负采样的采样数量

        self.filename = "data/text8.zip"  # 语料库文件
        self.vocabulary_size = 50000  # 词汇表大小


class Word2vec_naive(object):

    def __init__(self):
        self.config = Config()
        self.start_index = 0  # 记录当前采样位置
        self.sess = None

        self.read_data_and_build_dataset()
        print("成功构建语料库...")

    def read_data_and_build_dataset(self):
        """读取语料库，生成词汇表dict_vocabulary，对语料库进行one-hot编码得到新语料库corpus"""
        with zipfile.ZipFile(self.config.filename) as f:
            raw_corpus = tf.compat.as_str(f.read(f.namelist()[0])).split()  # 读取原始语料库数据

        self.count = [['UNK', -1]]  # count代表词汇表中各词的词频，UNK代表所有未被选入词汇表的单词
        self.count.extend(collections.Counter(raw_corpus).most_common(self.config.vocabulary_size - 1))

        self.dict_vocabulary = {}  # 对词汇表中的单词按照词频编号，词频越大编号越小
        for i, (word, _) in enumerate(self.count):
            self.dict_vocabulary[word] = i

        self.corpus = []  # 将raw_corpus中的词汇替换为词汇表中的编号，相当于one-hot编码
        unk_count = 0  # 统计UNK的数量
        for word in raw_corpus:
            index = self.dict_vocabulary.get(word, 0)
            if index == 0:
                unk_count += 1
            self.corpus.append(index)
        self.count[0][1] = unk_count

        self.dict_vocabulary_reversed = dict(zip(self.dict_vocabulary.values(), self.dict_vocabulary.keys()))

        self.p_negative_sampling = [(np.log(i + 2) - np.log(i + 1)) / np.log(self.config.vocabulary_size + 1)
                                    for i in range(self.config.vocabulary_size)]  # 负采样概率

    def generate_batch(self):
        """生成一个batch的训练样本，包括input单词和context单词"""
        assert self.config.batch_size % self.config.num_skip  == 0
        assert self.config.num_skip <= 2 * self.config.skip_window

        if self.start_index + self.config.skip_window > len(self.corpus) - 1:
            self.start_index = 0
        while  self.start_index - self.config.skip_window < 0:
            self.start_index += 1

        sk, ns, bs = self.config.skip_window, self.config.num_skip, self.config.batch_size
        buffer = collections.deque(maxlen = 2 * sk + 1)  # input单词左右各一个skip_window内的样本
        buffer.extend(self.corpus[self.start_index-sk : self.start_index+sk+1])
        span = [v for v in range(2 * sk + 1) if v != sk]  # 左右各一个window的序号

        batch_inputs, batch_contexts = np.ndarray(shape=[bs], dtype=np.int32), np.ndarray(shape=[bs, 1], dtype=np.int32)

        for i in range(bs // ns):

            input_sample = buffer[sk]  # 当前的input单词
            context_window = [buffer[i] for i in span]  # 包含当前所有context单词的窗口
            context_sample = np.random.choice(context_window, size=ns, replace=False)  # 从窗口内采样context

            for j in range(ns):
                batch_inputs[i * ns + j] = input_sample
                batch_contexts[i * ns + j, 0] = context_sample[j]

            if self.start_index + sk == len(self.corpus) - 1:
                self.start_index = sk
                buffer.extend(self.corpus[self.start_index-sk : self.start_index+sk+1])
            else:
                self.start_index += 1
                buffer.append(self.corpus[self.start_index + sk])
        return batch_inputs, batch_contexts

    def train_word2vec(self):
        """训练word2vec模型"""
        train_inputs = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size])
        train_contexts = tf.placeholder(dtype=tf.int32, shape=[self.config.batch_size, 1])

        # 单词作为input时的嵌入向量
        embeddings = tf.Variable(tf.random_uniform([self.config.vocabulary_size, self.config.embedding_size], -1, 1))
        input_embeddings = tf.nn.embedding_lookup(embeddings, train_inputs)

        # 单词作为context时的嵌入向量
        context_weights = tf.Variable(tf.truncated_normal([self.config.vocabulary_size,self.config.embedding_size],
                                                          stddev=1.0 / np.sqrt(self.config.embedding_size)))
        context_biases = tf.Variable(tf.zeros([self.config.vocabulary_size]), trainable=False)

        loss = tf.reduce_mean(
            my_nce_loss.nce_loss(
                self.config.num_neg_sam,
                self.config.vocabulary_size,
                context_weights,
                context_biases,
                input_embeddings,
                train_contexts,
                self.p_negative_sampling
            )
        )

        my_opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_step = my_opt.minimize(loss)

        # 开始训练
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        average_loss = 0
        for epoch in range(self.config.max_epoch):
            batch_inputs, batch_contexts = self.generate_batch()  # 生成一个batch的训练样本
            feed_dict0 = {train_inputs: batch_inputs, train_contexts: batch_contexts}

            _, loss_val = self.sess.run([train_step, loss], feed_dict=feed_dict0)
            average_loss += loss_val

            if epoch % self.config.print_loss_step == 0:
                if epoch != 0:
                    average_loss /= self.config.print_loss_step
                print("Loss at step", epoch + 1, "is", average_loss)
                average_loss = 0

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        embeddings_norm = embeddings / norm  # 将嵌入向量标准化为单位长度
        self.embeddings_final = self.sess.run(embeddings_norm)  # 最终保存的嵌入向量


if __name__ == "__main__":
    w = Word2vec_naive()
    w.train_word2vec()