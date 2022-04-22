

## 第一条

WordNet是什么？简述其构成。其和同义词库有什么区别？（10分）

WordNet是一个在20世纪80年代由Princeton大学的著名认知心理学家George Miller团队构建的一个大型的英文词汇数据库。名词、动词、形容词和副词以同义词集合（synsets）的形式存储在这个数据库中。每一个synset代表了一个同义词集合。各个synsets之间通过语义关系和词性关系等边连接。

与同义词库的区别: WordNet描述了words之间的各种复杂关系, 包括了上位词和下位词关系，整体-部分关系, 而同义词库只是单纯的按照词义相近的规则对words进行索引，没有考虑到其他复杂的关系。



## 第二条

简述词汇语义的分布式思想，word2vec方法如何建模词汇关系？（10分）

在分布式表示（Distributed Representation）中，一个实体（词汇）是由多个元素表征的，同样，一个元素也不仅只涉及到一个实体。

基于分布式表示的思想，我们构建一个密集向量（Dense Vector）来表征一个词汇。使用向量点积表征词汇含义相似度，这就是Word2Vec方法的基本思想。即

- 将词典中的每一个词都用分布式实数词特征向量表示以对抗维度灾难；
- 基于该特征向量表达词序的联合概率函数；
- 同时学习词特征向量与概率函数的参数。



## 第三条

写出CBOW模型详细公式和训练过程。

详细公式：

第一步就是去计算隐藏层$h$的输出。如下：
$$
h=\frac{1}{C} W \cdot\left(\sum_{i=1}^{C} x_{i}\right)
$$
该输出就是输入向量的加权平均。这里的隐藏层与skip-gram的隐藏层明显不同。

第二部就是计算在输出层每个结点的输入。如下：
$$
u_{j}=v_{w j}^{\prime} \cdot h
$$
其中$v_{w j}^{\prime}$是输出矩阵$W^{\prime}$的第j列。

最后我们计算输出层的输出，输出$y_j$如下：
$$
hy_{c, j}=p\left(w_{y, j} \mid w_{1}, \ldots, w_{c}\right)=\frac{\exp \left(u_{j}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(u^{\prime} j\right)}
$$
训练过程：

首先就是定义损失函数，这个损失函数就是给定输入上下文的输出单词的条件概率，一般都是取对数，如下所示：
$$
\begin{array}{c}E = -\log p\left(w_{O} \mid w_{I}\right) = -v_{w o}^{T} \cdot h-\log \sum_{j=1}^{V} \exp \left(v_{w_{j}^{\prime}}^{T} \cdot h\right)\end{array}
$$
接下来就是对上面的概率求导，具体推导过程可以去看BP算法，我们得到输出权重矩阵$W^{\prime}$的更新规则：

$$
w^{\prime}(\text { new })=w_{i j}^{\prime}(\text { old })-\eta \cdot\left(y_{j}-t_{j}\right) \cdot h_{i}
$$

同理权重$W$的更新规则如下：

$$
h_{i}w^{(\text {new })}=w_{i j}^{(\text {old })}-\eta \cdot \frac{1}{C} \cdot E H
$$

## 第四条

简述词向量学习的全局矩阵分解法和局部上下文窗口法的主要做法和各自优缺点。

全局矩阵分解法:

矩阵分解法就是通过统计一个事先指定大小的窗口内的词共现次数,以词周边的共现词的次数做为当前word的vector。具体来说,我们通过从大量的语料文本中构建一个词共现矩阵来统计次数。因为这个共现矩阵维度过大而且非常稀疏(Huge and Sparse), 所以我们通过SVD奇异值分解的方法来进行进行降维, 从而得到词向量的稠密表示。

优点: 能够有效的利用全局信息。

缺点: 高频成分多为噪声或者停用词之类的, 他们对预测结果的贡献和他们的权重不成正比, 会在很大程度上影响任务的准确率。



局部上下文窗口法(也就是word2vec, 这个方法通过两个基本的假设来建模词汇关系):

1. word2vec模型使用稠密向量来表示词汇，而不是独热编码。

2. 中心词可以用其上下文来表示.

优点: 在词汇类比任务中具有很好的效果。

缺点: 不能很好的利用全局信息。



## 第五条

什么是语言模型？ n-gram语言模型是什么，如何计算？写出公式。

语言模型是根据语言客观事实而进行的语言抽象数学建模，是一种对应关系。语言模型与语言客观事实之间的关系，如同数学上的抽象直线与具体直线之间的关系。

N-Gram是一种基于统计语言模型的算法。它的基本思想是将文本里面的内容按照字节进行大小为N的滑动窗口操作，形成了长度是N的字节片段序列。

如果我们有一个由 m 个词组成的序列（或者说一个句子），我们希望算得概率 $p\left(w_{1}, w_{2}, \ldots, w_{m}\right)$，根据链式规则，可得
$$
p\left(w_{1}, w_{2}, \ldots, w_{m}\right)=p\left(w_{1}\right) * p\left(w_{2} \mid w_{1}\right) * p\left(w_{3} \mid w_{1}, w_{2}\right) \ldots \ldots p\left(w_{m} \mid w_{1}, \ldots, w_{m-1}\right)
$$
这个概率显然并不好算，不妨利用马尔科夫链的假设，即当前这个词仅仅跟前面几个有限的词相关，因此也就不必追溯到最开始的那个词，这样便可以大幅缩减上述算式的长度。即
$$
p\left(w_{1}, w_{2}, \ldots, w_{m}\right)=p\left(w_{i} \mid w_{i-n+1, \ldots, w_{i-1}}\right)
$$


## 第六条

简述N-gram语言模型的稀疏问题和解决方法。

由于自然语言组合的多样, 我们语料库中不可能覆盖所有词汇可能的共现情况. 因此某词汇与上下文共现次数为0的情况

解决方法: 平滑化。平滑处理的算法有很多，例如：加1法、加法平滑方法、Good-Turing估计法、Katz平滑方法、Jelinek-Mercer平滑方法、Witten-Bell平滑方法等，其中Good-Turing估计法是很多平滑技术的核心，于1953年有古德（I.J.Good）引用图灵（Turing）的方法而提出来的，取名古德-图灵估计法。基本思想是：用观察计数较高的N元语法数重新估计概率量的大小，并把它指派给那些具有零计数或者较低计数的N元语法。



## 第七条

数学公式推导

https://blog.csdn.net/itplus/article/details/37969519

知乎博客：

https://zhuanlan.zhihu.com/p/108987941
https://zhuanlan.zhihu.com/p/26306795

吴恩达，李宏毅教学视频
https://www.bilibili.com/video/BV164411b7dx

Paper：
https://arxiv.org/pdf/1301.3781.pdf
https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
https://nlp.stanford.edu/pubs/glove.pdf
https://aclanthology.org/Q15-1016/
https://aclanthology.org/D15-1036/



## 第八条

关于第一章和第二章知识点，写下你仍然存在的问题

Q1:
word2vec和glove各有所长，能不能设计一个多任务的学习来同时进行embedding的优化呢

Q2:
word2vec的huffman编码树是否可以改用其他的数据结构来优化分类呢，比如二叉树



## 补充

详细公式我们设词汇表中第i个词的词向量为$v_i$ ,设中心词和周围词的序号分别为$c$ 和$o$，则内积$v_o^Tv_c$可以一定程度上表示两个词的相似程度，然后我们可以使用一个Softmax函数，来将其转化成概率值，即：

$$
P(w_o|w_c) = \frac{exp(v_o^Tv_c)}{\sum^{V}_{i} exp(v_i^Tv_c)}
$$

我们在训练过程时发现，上面这个式子求导不太方便，我们如果对中心词和周围词采用两套词向量分别表示,即中心词用$v_c$ 表示，周围词用$u_o$ 表示，求导就会容易很多，所以我们更常见的写法是下面这种： 

$$
P(w_o|w_c) = \frac{exp(u_o^Tv_c)}{\sum^{V}_{i} exp(u_i^Tv_c)}
$$

于是，对于一个中心词$w_c$和一个上下文词$w_o$，其损失函数就可以由上面的概率值的负对数来表示： 
$$
\begin{align} J_{single} &= -log(P(w_o|w_c)) \\ &= -log(\frac{exp(u_o^Tv_c)}{\sum^{V}_{i} exp(u_i^Tv_c)})\nonumber \end{align}
$$

那么，整个窗口内损失函数，就是把窗口内各上下文词与中心词计算损失再累加： 

$$
\begin{align} J_{window} &= \sum_{w_o \in window}J_{single} \\& = -\sum_{w_o \in window}log(P(w_o|w_c))\nonumber \end{align}
$$

如果要计算在整个语料中的损失，那就是再遍历所有的中心词，再累加： 
$$
\begin{align} J_{corpus} &= \frac{1}{V}\sum_{w_c \in corpus}J_{window} \\ &= -\frac{1}{V}\sum_{w_c \in corpus}\sum_{w_o \in window}log(P(w_o|w_c))\nonumber \end{align}
$$
其中V为词汇表大小。

最后再用随机梯度下降可以完成最后的训练过程

