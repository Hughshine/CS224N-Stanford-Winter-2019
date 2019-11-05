# assignment2 matrix derivation & word2vec & SGD 

> 虽然说要做深度学习很久了，但是这个周末才真正走出一点舒适区，开始敲代码，融会贯通一点点
> 之前对BP有些模糊，还有论文材料里莫名其妙的矩阵维度弄得脑壳疼
> 简单做一些记录啦。每一小节最后给出了reference，应该会比我复述的更清楚~ 有优先级啦。


## 基础知识：矩阵求导

高数的基础知识是标量与标量的求导。进入机器学习的世界，论文或代码都是以向量为整体考虑。因而要说明学术界的规范是比较重要的。

求导有自变量、因变量，分别可以为标量、向量、矩阵，所以共可能有9种情况。本文我们设定向量都为列向量。

考虑标量对向量的求导，很容易认为向量

> “另外三种向量对矩阵的求导，矩阵对向量的求导，以及矩阵对矩阵的求导我们在后面再讲。” 没有提及。

> ref:
> 1. [刘建平Pinard：机器学习中的矩阵向量求导(一) 求导定义与求导布局](https://www.cnblogs.com/pinard/p/10750718.html)
> 2. [wikipedia: Matrix Caculas(layout conventions)](https://en.wikipedia.org/wiki/Matrix_calculus#Layout_conventions)
> 3. [一个矩阵求导的简述](https://zlearning.netlify.com/math/matrix/matrix-gradient.html)
> > 不过这篇博客应该用的是分母布局

> 有时间可以了解下 泛函分析

## 基础知识：BP

## 三层的网络：word2vec

### "bigram" language model

### CBOW

### skip-gram

### furthre optimization: hierarchical softmax & negative sampling

对于网络的优化，最重要是知道损失函数，并推导出每一层权重矩阵的梯度变化，和backprop的梯度。

#### hierarchical softmax

#### negative sampling

## CS224N-assignment2

### word2vec

#### naive softmax loss

#### skip-gram loss (sum up naive softmax loss)

#### negative sampling

### SGD

