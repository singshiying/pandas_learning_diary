# 一.python基础

## 1.列表推导式与条件赋值

- 函数的定义：def
- 空数组的定义：[ ]
- 为数组添加数据：L.append


```python
L=[]

def my_func(x):
    return 2*x

for i in range(5):
    L.append(my_func(i))
    
L
```




    [0, 2, 4, 6, 8]



- 列表推导式：一层嵌套与多层嵌套


```python
[my_func(i) for i in range(5)]
```




    [0, 2, 4, 6, 8]



列表推导式：[ * for i * ]，其中第一个 * 为映射函数，第二个为迭代对象。


```python
[m+'_'+n for m in ['a','b'] for n in ['c','d']]
```




    ['a_c', 'a_d', 'b_c', 'b_d']



- 条件赋值： value = a if condition else b 


```python
value ='cat' if 2>1 else 'dog'
value
```




    'cat'




```python
a,b ='cat','dog'
condition =2>1
if condition:
    value=a
else:
    value=b
    
value
```




    'cat'



- 条件赋值加列表推导


```python
L=[1,2,3,4,5,6,7]
[i if i<=5 else 5 for i in L]
```




    [1, 2, 3, 4, 5, 5, 5]



## 2.匿名函数与map方法

- 匿名函数：lambda


```python
my_func=lambda x:2*x
my_func(3)
```




    6




```python
multi_para_func=lambda a,b:a+b
multi_para_func(1,2)
```




    3



- map函数返回map对象，需用list转化成列表


```python
list(map(lambda x:2*x,range(5)))
```




    [0, 2, 4, 6, 8]




```python
list(map(lambda x,y:str(x)+'_'+y,range(5),list('abcde')))
```




    ['0_a', '1_b', '2_c', '3_d', '4_e']



## 3.zip对象与enumerate方法

- zip函数打包迭代对象，返回zip对象，需用tuple/list得到打包结果


```python
L1,L2,L3=list('abc'),list('def'),list('hij')
list(zip(L1,L2,L3))
```




    [('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]




```python
L1='abc'
L1
```




    'abc'




```python
L1=list('abc')
L1
```




    ['a', 'b', 'c']




```python
tuple(zip(L1,L2,L3))
```




    (('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j'))



- zip函数加循环


```python
for i,j,k in zip(L1,L2,L3):
    print(i,j,k)
```

    a d h
    b e i
    c f j
    


```python
L=list('abcd')
for index,value in zip(range(len(L)),L):
    print(index,value)
```

    0 a
    1 b
    2 c
    3 d
    

- enumerate迭代时绑定迭代元素的遍历序号


```python
for index,value in enumerate(L):
    print(index,value)
```

    0 a
    1 b
    2 c
    3 d
    

- 建立字典映射：dict


```python
dict(zip(L1,L2))
```




    {'a': 'd', 'b': 'e', 'c': 'f'}



- zipped解压函数


```python
zipped=list(zip(L1,L2,L3))
zipped
```




    [('a', 'd', 'h'), ('b', 'e', 'i'), ('c', 'f', 'j')]




```python
list(zip(*zipped)) #三个元组分别对应原来的列表
```




    [('a', 'b', 'c'), ('d', 'e', 'f'), ('h', 'i', 'j')]



# 二.Numpy基础

## 1.np数组的构造

- `np.array`最一般的构造数组的方法


```python
import numpy as np

np.array([1,2,3])
```




    array([1, 2, 3])



- 等差数列：`np.linspace`，`np.arange`


```python
np.linspace(1,5,11)
```




    array([1. , 1.4, 1.8, 2.2, 2.6, 3. , 3.4, 3.8, 4.2, 4.6, 5. ])




```python
np.arange(1,10,2)
```




    array([1, 3, 5, 7, 9])



- 特殊矩阵：`zeros`（全是0），`eye`（单位矩阵），`full`（全是自己定义的某个数），`tile`


```python
np.zeros((2,3))
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
np.eye(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
np.eye(3,k=-2)# 偏移主对角线-2个单位的伪单位矩阵
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [1., 0., 0.]])




```python
np.full((2,3),10)
```




    array([[10, 10, 10],
           [10, 10, 10]])




```python
np.full((2,3),[1,2,3])
```




    array([[1, 2, 3],
           [1, 2, 3]])




```python
np.tile([1,2],(2,3))
```




    array([[1, 2, 1, 2, 1, 2],
           [1, 2, 1, 2, 1, 2]])



逗号隔开的是每列数据，这是一个六行两列的数据

- 随机矩阵：`np.random`

最常用的随机生成函数为 rand, randn, randint, choice ，它们分别表示0-1均匀分布的随机数组、标准正态的随机数组、随机整数组和随机列表抽样


```python
np.random.rand(3) # 生成服从0-1均匀分布的三个随机数
```




    array([0.84633156, 0.22848166, 0.73045607])




```python
np.random.rand(2,3) # 注意这里传入的不是元组，每个维度大小分开输入
```




    array([[0.91852663, 0.28141611, 0.69065354],
           [0.40065741, 0.29022665, 0.96962347]])



在区间a，b上均匀分布


```python
a,b=5,15
(b-a)*np.random.rand(3)+a
```




    array([ 6.69513983, 14.15546683, 14.7755908 ])



正态分布`np.random.randn`


```python
np.random.randn(3)
```




    array([-0.07623673,  0.45768267,  0.52496699])




```python
np.random.randn(2,3)
```




    array([[-0.12438315, -0.37783527, -1.90077733],
           [ 2.03283755,  0.82230168,  2.11644765]])



可以定义方差与均值


```python
sigma,mu=2.5,3
mu+np.random.randn(3)*sigma
```




    array([1.56356183, 4.40403646, 1.44854361])



随机整数数组，最小值，最大值，维度


```python
low,high,size=5,25,(2,2)
np.random.randint(low,high,size)
```




    array([[ 7,  9],
           [15, 23]])



抽样


```python
my_list=['a','b','c','d']
np.random.choice(my_list,2,replace=False,p=[0.1,0.7,0.1,0.1])#可以定义每个数据抽样的概率
```




    array(['b', 'c'], dtype='<U1')




```python
np.random.choice(my_list,(3,3))
```




    array([['d', 'c', 'b'],
           ['d', 'b', 'd'],
           ['a', 'a', 'a']], dtype='<U1')



- 打散原列表


```python
np.random.permutation(my_list)
```




    array(['a', 'd', 'c', 'b'], dtype='<U1')



- 随机种子


```python
np.random.seed(0)
```


```python
np.random.rand()
```




    0.5488135039273248



## 2.np数组的变形与合并

- 转置：T


```python
np.zeros((2,3)).T
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.]])



- 合并操作： r_, c_


```python
np.r_[np.zeros((2,3)),np.zeros((2,3))]
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
np.c_[np.zeros((2,3)),np.zeros((2,3))]
```




    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])



- 维度变换： reshape

C模式和F模式，分别是逐行与逐列的顺序进行填充读取


```python
target=np.arange(8).reshape(2,4)
target
```




    array([[0, 1, 2, 3],
           [4, 5, 6, 7]])



-1代表默认


```python
target.reshape(4,-1)
```




    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])




```python
target.reshape((4,2),order='C')
```




    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])




```python
target.reshape((4,2),order='F')
```




    array([[0, 2],
           [4, 6],
           [1, 3],
           [5, 7]])




```python
target=np.ones((3,1))
target
```




    array([[1.],
           [1.],
           [1.]])



reshape(-1)很常用，转化为1维数据


```python
target.reshape(-1)

```




    array([[1., 1., 1.]])




```python
target.T
```




    array([[1., 1., 1.]])



## 3.np数组的切片和索引

数组的切片模式支持使用 slice 类型的 start:end:step 切片，还可以直接传入列表指定某个维度的索引进行切片
索引和切片的详细分析可见
>https://www.cnblogs.com/mengxiaoleng/p/11616869.html


```python
target=np.arange(9).reshape(3,3)
target
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
target[:-1,[1,2]]#-1代表行数，除了最后一个取全部，[1,2]代表第1与第2列
```




    array([[1, 2],
           [4, 5]])



- np.ix_在对应维度上用布尔索引


```python
target[np.ix_([True, False, True], [True, False, True])]
```




    array([[0, 2],
           [6, 8]])




```python
target[np.ix_([1,2], [True, False, True])]
```




    array([[3, 5],
           [6, 8]])




```python
new = target.reshape(-1)
new
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8])




```python
new[new%2==0]#取出除以2余数为0的数
```




    array([0, 2, 4, 6, 8])



## 4.常用函数

- where


```python
a=np.array([-1,1,-1,0])
np.where(a>0,a,5)
```




    array([5, 1, 5, 5])



- nonzero返回非零数的索引


```python
a=np.array([-2,-5,0,1,3,-1])
np.nonzero(a)
```




    (array([0, 1, 3, 4, 5], dtype=int64),)



- argmax返回最大数的索引，argmin返回最小数的索引


```python
a.argmax()
```




    4




```python
a.argmin()
```




    1



- any表示当序列至少存在一个True或者非零元素，返回True，否则返回False


```python
a=np.array([0,1])
a.any()
```




    True



- all 指当序列元素 全为 True 或非零元素时返回 True ，否则返回 False


```python
a.all()
```




    False



- cumpro表示累乘函数


```python
a=np.array([1,2,3])
a.cumprod()
```




    array([1, 2, 6], dtype=int32)



- cumsum表示累加函数，返回同长度的数组


```python
a.cumsum()
```




    array([1, 3, 6], dtype=int32)



- diff表示与前一个数做差，返回长度比原数组短1


```python
np.diff(a)
```




    array([1, 1])



- 统计函数：max，min，mean，median，std，var，sum，quantile


```python
target=np.arange(5)
target
```




    array([0, 1, 2, 3, 4])




```python
target.max() #最大值
```




    4




```python
np.quantile(target,0.5) #中位数
```




    4.0




```python
target1=np.array([1,3,5,9])
target2=np.array([1,5,3,-9])
np.cov(target1,target2) #协方差
```




    array([[ 11.66666667, -16.66666667],
           [-16.66666667,  38.66666667]])




```python
np.corrcoef(target1,target2) #相关系数
```




    array([[ 1.        , -0.78470603],
           [-0.78470603,  1.        ]])




```python
target = np.arange(1,10).reshape(3,-1)
target
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])



当 axis=0 时结果为列的统计指标，当 axis=1 时结果为行的统计指标


```python
target.sum(0) #每列的数字相加
```




    array([12, 15, 18])




```python
target.sum(1) #每行的数字相加
```




    array([ 3, 12, 21])



## 5.广播机制

- 标量与数组


```python
res=3*np.ones((2,2))+1 #3乘以一个2x2的数组，是数组的每一个数都乘以3
res
```




    array([[4., 4.],
           [4., 4.]])




```python
res=1/res
res
```




    array([[0.25, 0.25],
           [0.25, 0.25]])



- 二维数组与二维数组


```python
res=np.ones((3,2))
res
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])




```python
res*np.array([2,3]) #3x2的数组乘1x2的数组
```




    array([[ 8., 12.],
           [ 8., 12.]])




```python
res*np.array([[2],[3],[4]])
```




    array([[2., 2.],
           [3., 3.],
           [4., 4.]])




```python
np.array([[2],[3],[4]])
```




    array([[2],
           [3],
           [4]])



- 一维数组与二维数组


```python
np.ones(3)+np.ones((2,3))
```




    array([[2., 2., 2.],
           [2., 2., 2.]])




```python
np.ones(3) + np.ones((2,1))
```




    array([[2., 2., 2.],
           [2., 2., 2.]])




```python
np.ones(1) + np.ones((2,3))
```




    array([[2., 2., 2.],
           [2., 2., 2.]])



## 6.向量与矩阵的计算

- 向量内积：dot


```python
a=np.array([1,2,3])
b=np.array([1,3,5])
a.dot(b) #1+6+15
```




    22



- 向量范数与矩阵范数：np.linalg.norm

向量和矩阵的相关公式可见
>https://blog.csdn.net/qq_15807167/article/details/54631261


```python
matrix_target=np.arange(4).reshape(-1,2)
matrix_target
```




    array([[0, 1],
           [2, 3]])




```python
np.linalg.norm(matrix_target,'fro') #frobenius范数，即矩阵元素绝对值的平方和再开平方
#（1+4+9）**（1/2）
```




    3.7416573867739413




```python
np.linalg.norm(matrix_target,np.inf) #行和范数，即所有矩阵行向量绝对值之和的最大值
#2+3
```




    5.0




```python
np.linalg.norm(matrix_target,1)#列和范数，即所有矩阵列向量绝对值之和的最大值
#1+3
```




    4.0




```python
np.linalg.norm(matrix_target,-1)#即所有矩阵列向量绝对值之和的最小值
#0+2
```




    2.0




```python
np.linalg.norm(matrix_target, 2) #2范数，谱范数，即(A.T).dot(A)的最大特征值的开平方
```




    3.702459173643833




```python
np.linalg.norm(matrix_target, -2) #即(A.T).dot(A)的最小特征值的开平方
```




    0.5401815134754528




```python
(matrix_target.T).dot(matrix_target)
```




    array([[ 4,  6],
           [ 6, 10]])



- 矩阵乘法：@


```python
a = np.arange(4).reshape(-1,2)
a
```




    array([[0, 1],
           [2, 3]])




```python
b = np.arange(-4,0).reshape(-1,2)
b
```




    array([[-4, -3],
           [-2, -1]])




```python
a@b#第[0]行向量dot第[:,0]列向量得到[0][0]的数据
#0*-4+1*-2=-2
#0*-3+1*-1=-1
```




    array([[ -2,  -1],
           [-14,  -9]])



## 三.练习

EX1:利用列表推导式写矩阵乘法

一般的矩阵乘法根据公式，可以由三重循环写出：


```python
M1 = np.random.rand(2,3)
M1
```




    array([[0.891773  , 0.96366276, 0.38344152],
           [0.79172504, 0.52889492, 0.56804456]])




```python
M2 = np.random.rand(3,4)
M2
```




    array([[0.44767829, 0.75221083, 0.94991427, 0.81705698],
           [0.93094131, 0.47506013, 0.71931657, 0.30904959],
           [0.06958711, 0.36547353, 0.69791523, 0.27462211]])




```python
res=np.empty((M1.shape[0],M2.shape[1]))
res
```




    array([[3.5, 1.5, 0.5, 4.5],
           [1. , 5. , 3. , 9. ]])




```python
for i in range(M1.shape[0]):
    for j in range(M2.shape[1]):
        item=0
        for k in range(M1.shape[1]):
            item+=M1[i][k]*M2[k][j]
        res[i][j]=item
res
```




    array([[1.9342174 , 0.99017907, 1.21538516, 1.0064396 ],
           [1.62714412, 0.72994502, 0.97250281, 0.60077883]])




```python
((M1@M2 - res) < 1e-15).all()
```




    True



请将其改写为列表推导式的形式:


```python
res=np.array([sum(M1[i][k]*M2[k][j] for k in range(M1.shape[1])) for i in range(M1.shape[0]) for j in range(M2.shape[1])]).reshape((2,4))
res
```




    array([[1.9342174 , 0.99017907, 1.21538516, 1.0064396 ],
           [1.62714412, 0.72994502, 0.97250281, 0.60077883]])




```python
 (M1@M2-res<1e-15).all()
```




    True



EX2:更新矩阵

设矩阵 Am×n ，现在对 A 中的每一个元素进行更新生成矩阵 B ，更新方法是 Bij=Aij∑k=1n1Aik ，例如下面的矩阵为 A ，则 B2,2=5×(1/4+1/5+1/6)=37/12 ，请利用 Numpy 高效实现。


```python
A=np.arange(1,10).reshape((3,3))
A
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
B=np.ones((A.shape[0],A.shape[1]))
B
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])



第一种方法：三重循环


```python
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        item=0
        for k in range(A.shape[1]):
            item+=1/A[i][k]
        B[i][j]=A[i][j]*item
B
```




    array([[1.83333333, 3.66666667, 5.5       ],
           [2.46666667, 3.08333333, 3.7       ],
           [2.65277778, 3.03174603, 3.41071429]])



第二种方法：列表推导式


```python
B=np.array([A[i][j]*sum(1/A[i][k] for k in range(A.shape[1])) for i in range(A.shape[0]) for j in range(A.shape[1])]).reshape((A.shape[0],A.shape[1])) 
B
```




    array([[1.83333333, 3.66666667, 5.5       ],
           [2.46666667, 3.08333333, 3.7       ],
           [2.65277778, 3.03174603, 3.41071429]])



第三种方法：利用矩阵运算


```python
B=(1/A).sum(1).reshape(-1,1)*A
B
```




    array([[1.83333333, 3.66666667, 5.5       ],
           [2.46666667, 3.08333333, 3.7       ],
           [2.65277778, 3.03174603, 3.41071429]])



EX3:卡方统计量


```python
np.random.seed(0)
A=np.random.randint(10,20,(8,5))
A
```




    array([[15, 10, 13, 13, 17],
           [19, 13, 15, 12, 14],
           [17, 16, 18, 18, 11],
           [16, 17, 17, 18, 11],
           [15, 19, 18, 19, 14],
           [13, 10, 13, 15, 10],
           [12, 13, 18, 11, 13],
           [13, 13, 17, 10, 11]])



第一种方法：根据给出公式


```python
B=A.sum(1).reshape(-1,1)*A.sum(0).reshape(1,-1)/A.sum()
B
```




    array([[14.14211438, 13.08145581, 15.20277296, 13.67071057, 11.90294627],
           [15.18197574, 14.04332756, 16.32062392, 14.67590988, 12.77816291],
           [16.63778163, 15.38994801, 17.88561525, 16.08318891, 14.0034662 ],
           [16.42980936, 15.19757366, 17.66204506, 15.88214905, 13.82842288],
           [17.67764298, 16.35181976, 19.0034662 , 17.08838821, 14.87868284],
           [12.68630849, 11.73483536, 13.63778163, 12.26343154, 10.67764298],
           [13.93414211, 12.88908146, 14.97920277, 13.46967071, 11.72790295],
           [13.3102253 , 12.31195841, 14.3084922 , 12.86655113, 11.20277296]])



第二种方法：运算处理后


```python
x2=((A-B)**2/B).sum()
x2
```




    11.842696601945802



EX4:改进矩阵计算的性能


```python
np.random.seed(0)
m, n, p = 100, 80, 50
B = np.random.randint(0, 2, (m, p))
B
```




    array([[0, 1, 1, ..., 1, 0, 1],
           [0, 1, 1, ..., 1, 1, 0],
           [1, 0, 0, ..., 1, 1, 1],
           ...,
           [1, 0, 0, ..., 1, 1, 0],
           [0, 0, 0, ..., 1, 1, 0],
           [0, 1, 1, ..., 1, 1, 1]])




```python
U = np.random.randint(0, 2, (p, n))
U
```




    array([[1, 0, 1, ..., 1, 1, 0],
           [0, 0, 0, ..., 0, 0, 1],
           [0, 1, 0, ..., 0, 1, 1],
           ...,
           [1, 0, 1, ..., 1, 0, 1],
           [0, 0, 1, ..., 0, 0, 0],
           [1, 1, 0, ..., 0, 0, 1]])




```python
Z = np.random.randint(0, 2, (m, n))
Z
```




    array([[1, 0, 0, ..., 1, 0, 0],
           [0, 1, 1, ..., 1, 1, 0],
           [1, 0, 1, ..., 0, 1, 1],
           ...,
           [0, 1, 0, ..., 1, 1, 0],
           [1, 0, 0, ..., 1, 1, 0],
           [0, 0, 0, ..., 0, 0, 1]])




```python
def solution(B=B, U=U, Z=Z):
    L_res = []
    for i in range(m):
        for j in range(n):
            norm_value = ((B[i]-U[:,j])**2).sum()
            L_res.append(norm_value*Z[i][j])
    return sum(L_res)

solution(B, U, Z)
```




    100566



第一种方法：列表推导式


```python
R=sum([((B[i]-U[:,j])**2).sum()*Z[i][j] for i in range(m) for j in range(n)])
R
```




    100566



第二种方法：矩阵运算


```python
R=(((B**2).sum(1).reshape(-1,1)+(U**2).sum(0).reshape(1,-1)-2*B@U)*Z).sum()
R
```




    100566



EX5:连续整数的最大长度

输入一个整数的 Numpy 数组，返回其中递增连续整数子数组的最大长度。例如，输入 [1,2,5,6,7]，[5,6,7]为具有最大长度的递增连续整数子数组，因此输出3；输入[3,2,1,2,3,4,6]，[1,2,3,4]为具有最大长度的递增连续整数子数组，因此输出4。请充分利用 Numpy 的内置函数完成。（提示：考虑使用 nonzero, diff 函数）


```python
A=np.random.randint(1,20,10)
res=np.diff(A) 
#第一步：与前一个数做差，若为1，则说明是递增的
print(res) 
print(np.diff(A)!=1)
#第二步：若为1（原数组递增），则数组为0（False），不为1（原数组不递增），则数组为1（True）
print(np.nonzero(np.diff(A)!=1))
#第三步：选出不为0的索引号
print(np.diff(np.nonzero(np.diff(A)!=1)))
#第四步：将索引号与前一个做差
#如果数据是5,7,差为2，说明6的位置是递增的，说明原数组有两个数递增
#如果数据为5，6，差为1.说明原数组没有递增的（只有1个数递增）
#但是这样存在一个问题，当递增的数位于原数组的开头或者末尾的时候
#第三步的索引号不能囊括开头或者末尾
#所以要在第三步前给数组前面和最后面都加上一个1，这样索引号才能囊括所有
```

    [-11   0  -2   8  -4  -3  16  -4  -7]
    [ True  True  True  True  True  True  True  True  True]
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int64),)
    [[1 1 1 1 1 1 1 1]]
    


```python
f = lambda x:np.diff(np.nonzero(np.r_[1,np.diff(x)!=1,1])).max()
f(A)
```




    2


