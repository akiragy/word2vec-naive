# word2vec-naive
word2vec的极简版本，只保留最基础的功能，剖析并学习word2vec_basic.py的产物。

使用了自己实现的nce_loss而不是tensorflow内置的tf.nn.nce_loss函数，经测试两者效果几乎一致。

tensorflow官方实现的basic版本：

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
