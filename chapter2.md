# 第二章 pandas基础


```python
import numpy as np
import pandas as pd
```


```python
pd.__version__
```




    '1.1.5'



## 一、文件的读取和写入

### 1.文件读取

pandas可以读取的文件格式有很多，这里主要介绍读取 csv, excel, txt 文件。

- csv文件：`pd.read.csv`


```python
df_csv=pd.read_csv('D:\\datawhale\\joyful-pandas\\data\\my_csv.csv')
#windows系统的目录用 \\ 隔开，linux系统用/隔开
```

用%可以在pwershell中运行代码


```python
%pwd
```




    'D:\\datawhale'




```python
df_csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
      <td>2020/1/5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>



- txt文件：`pd.read_table`


```python
df_txt=pd.read_table('D:\\datawhale\\joyful-pandas\\data\\my_table.txt')
df_txt
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple 2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana 2020/1/2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange 2020/1/5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon 2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>



- excel文件：pd.read_excel


```python
df_excel = pd.read_excel('D:\\datawhale\\joyful-pandas\\data\\my_excel.xlsx')
df_excel
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
      <td>2020/1/5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>



- 公共参数：
`header=None` 表示第一行不作为列名， `index_col` 表示把某一列或几列作为索引，索引的内容将会在第三章进行详述， `usecols` 表示读取列的集合，默认读取所有的列， `parse_dates` 表示需要转化为时间的列，关于时间序列的有关内容将在第十章讲解， `nrows` 表示读取的数据行数。


```python
pd.read_table('D:\\datawhale\\joyful-pandas\\data\\my_table.txt', header=None)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>col1</td>
      <td>col2</td>
      <td>col3</td>
      <td>col4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple 2020/1/1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana 2020/1/2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange 2020/1/5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon 2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv('D:\\datawhale\\joyful-pandas\\data\\my_csv.csv',index_col=['col1', 'col2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
    <tr>
      <th>col1</th>
      <th>col2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>3</th>
      <th>b</th>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
    <tr>
      <th>6</th>
      <th>c</th>
      <td>2.5</td>
      <td>orange</td>
      <td>2020/1/5</td>
    </tr>
    <tr>
      <th>5</th>
      <th>d</th>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020/1/7</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_table('D:\\datawhale\\joyful-pandas\\data\\my_table.txt', usecols=['col1', 'col2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv('D:\\datawhale\\joyful-pandas\\data\\my_csv.csv', parse_dates=['col5'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>c</td>
      <td>2.5</td>
      <td>orange</td>
      <td>2020-01-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>d</td>
      <td>3.2</td>
      <td>lemon</td>
      <td>2020-01-07</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_excel('D:\\datawhale\\joyful-pandas\\data\\my_excel.xlsx', nrows=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>a</td>
      <td>1.4</td>
      <td>apple</td>
      <td>2020/1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>b</td>
      <td>3.4</td>
      <td>banana</td>
      <td>2020/1/2</td>
    </tr>
  </tbody>
</table>
</div>



- txt文件分隔符不是空格时， `read_table` 有一个分割参数 `sep`，它使得用户可以自定义分割符号，进行 txt 数据的读取


```python
pd.read_table('D:\\datawhale\\joyful-pandas\\data\\my_table_special_sep.txt')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1 |||| col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TS |||| This is an apple.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GQ |||| My name is Bob.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WT |||| Well done!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PT |||| May I help you?</td>
    </tr>
  </tbody>
</table>
</div>



在使用 read_table 的时候需要注意，参数 sep 中使用的是**正则表达式**，因此需要对 | 进行转义变成 \ | 


```python
pd.read_table('D:\\datawhale\\joyful-pandas\\data\\my_table_special_sep.txt',sep='\|\|\|\|',engine='python')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TS</td>
      <td>This is an apple.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GQ</td>
      <td>My name is Bob.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WT</td>
      <td>Well done!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PT</td>
      <td>May I help you?</td>
    </tr>
  </tbody>
</table>
</div>



### 2.数据写入

- 写csv与excel文件

一般在数据写入中，最常用的操作是把 index 设置为 False 


```python
df_csv.to_csv('D:\\datawhale\\joyful-pandas\\data\\my_csv_saved.csv', index=False)
df_excel.to_excel('D:\\datawhale\\joyful-pandas\\data\\my_excel_saved.xlsx', index=False)
```

- 写txt文件

pandas 中没有定义 to_table 函数，但是 to_csv 可以保存为 txt 文件，并且允许自定义分隔符，常用制表符 \ t 分割（tab的意思）


```python
df_txt.to_csv('D:\\datawhale\\joyful-pandas\\data\\my_txt_saved.txt', sep='\t', index=False)
```

- 写markdown与latex文件


```python
print(df_csv.to_markdown())
```

    |    |   col1 | col2   |   col3 | col4   | col5     |
    |---:|-------:|:-------|-------:|:-------|:---------|
    |  0 |      2 | a      |    1.4 | apple  | 2020/1/1 |
    |  1 |      3 | b      |    3.4 | banana | 2020/1/2 |
    |  2 |      6 | c      |    2.5 | orange | 2020/1/5 |
    |  3 |      5 | d      |    3.2 | lemon  | 2020/1/7 |
    


```python
print(df_csv.to_latex())
```

    \begin{tabular}{lrlrll}
    \toprule
    {} &  col1 & col2 &  col3 &    col4 &      col5 \\
    \midrule
    0 &     2 &    a &   1.4 &   apple &  2020/1/1 \\
    1 &     3 &    b &   3.4 &  banana &  2020/1/2 \\
    2 &     6 &    c &   2.5 &  orange &  2020/1/5 \\
    3 &     5 &    d &   3.2 &   lemon &  2020/1/7 \\
    \bottomrule
    \end{tabular}
    
    

## 二、基本数据结构

pandas 中具有两种基本的数据存储结构，存储一维 values 的 Series 和存储二维 values 的 DataFrame ，在这两种结构上定义了很多的属性和方法。

### 1.Series

- 构造series

Series 一般由四个部分组成，分别是序列的值 data 、索引 index 、存储类型 dtype 、序列的名字 name 。其中，索引也可以指定它的名字，默认为空。


```python
s=pd.Series(data=[100,'a',{'dic1':5}],index=pd.Index(['id1',20,'third'],name='my_idx'),dtype='object',name='my_name')
s
```




    my_idx
    id1              100
    20                 a
    third    {'dic1': 5}
    Name: my_name, dtype: object



>object 代表了一种混合类型，正如上面的例子中存储了整数、字符串以及 Python 的字典数据结构。此外，目前 pandas 把纯字符串序列也默认认为是一种 object 类型的序列，但它也可以用 string 类型存储

- 对于这些属性(values,index,dtype,name)，可以通过**.**的方式获取


```python
s.values
```




    array([100, 'a', {'dic1': 5}], dtype=object)




```python
s.index
```




    Index(['id1', 20, 'third'], dtype='object', name='my_idx')




```python
s.dtype
```




    dtype('O')




```python
s.name
```




    'my_name'



- 获取序列的长度


```python
s.shape
```




    (3,)



- 取出索引对应的值


```python
s['third']
```




    {'dic1': 5}



### 2.DataFrame

- DataFrame的构造

DataFrame 在 Series 的基础上增加了列索引，一个数据框可以由二维的 data 与行列索引来构造


```python
data = [[1, 'a', 1.2], [2, 'b', 2.2], [3, 'c', 3.2]]
df = pd.DataFrame(data = data,index = ['row_%d'%i for i in range(3)],columns=['col_0', 'col_1', 'col_2'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row_0</th>
      <td>1</td>
      <td>a</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>row_1</th>
      <td>2</td>
      <td>b</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>row_2</th>
      <td>3</td>
      <td>c</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



一般而言，更多的时候会采用从列索引名到数据的映射来构造数据框，同时再加上行索引


```python
df = pd.DataFrame(data = {'col_0': [1,2,3], 'col_1':list('abc'),'col_2':[1.2,2.2,3.2]},index=['row_%d'%i for i in range(3)])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row_0</th>
      <td>1</td>
      <td>a</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>row_1</th>
      <td>2</td>
      <td>b</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>row_2</th>
      <td>3</td>
      <td>c</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



- DataFrame的读取


```python
df['col_0']
```




    row_0    1
    row_1    2
    row_2    3
    Name: col_0, dtype: int64




```python
df[['col_0','col_1']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row_0</th>
      <td>1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>row_1</th>
      <td>2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>row_2</th>
      <td>3</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



- DataFrame的属性


```python
df.values
```




    array([[1, 'a', 1.2],
           [2, 'b', 2.2],
           [3, 'c', 3.2]], dtype=object)




```python
df.index
```




    Index(['row_0', 'row_1', 'row_2'], dtype='object')




```python
df.columns
```




    Index(['col_0', 'col_1', 'col_2'], dtype='object')




```python
df.dtypes
```




    col_0      int64
    col_1     object
    col_2    float64
    dtype: object




```python
df.shape
```




    (3, 3)




```python
df.T#做转置
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_0</th>
      <th>row_1</th>
      <th>row_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>col_0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>col_1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
    </tr>
    <tr>
      <th>col_2</th>
      <td>1.2</td>
      <td>2.2</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



## 三、常用基本函数


```python
df = pd.read_csv('D:\\datawhale\\joyful-pandas\\data\\learn_pandas.csv')
df.columns
```




    Index(['School', 'Grade', 'Name', 'Gender', 'Height', 'Weight', 'Transfer',
           'Test_Number', 'Test_Date', 'Time_Record'],
          dtype='object')




```python
df = df[df.columns[:7]] #只使用前7列
```

### 1.汇总函数

- head,tail函数：返回前n行与后n行，默认n为5


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Grade</th>
      <th>Name</th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Freshman</td>
      <td>Gaopeng Yang</td>
      <td>Female</td>
      <td>158.9</td>
      <td>46.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Peking University</td>
      <td>Freshman</td>
      <td>Changqiang You</td>
      <td>Male</td>
      <td>166.5</td>
      <td>70.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>School</th>
      <th>Grade</th>
      <th>Name</th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>Transfer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Senior</td>
      <td>Chengqiang Chu</td>
      <td>Female</td>
      <td>153.9</td>
      <td>45.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Shanghai Jiao Tong University</td>
      <td>Senior</td>
      <td>Chengmei Shen</td>
      <td>Male</td>
      <td>175.3</td>
      <td>71.0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Tsinghua University</td>
      <td>Sophomore</td>
      <td>Chunpeng Lv</td>
      <td>Male</td>
      <td>155.7</td>
      <td>51.0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>



- info,describe函数：返回表的信息概况与数值列对应的主要统计量


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   School    200 non-null    object 
     1   Grade     200 non-null    object 
     2   Name      200 non-null    object 
     3   Gender    200 non-null    object 
     4   Height    183 non-null    float64
     5   Weight    189 non-null    float64
     6   Transfer  188 non-null    object 
    dtypes: float64(2), object(5)
    memory usage: 11.1+ KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>183.000000</td>
      <td>189.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>163.218033</td>
      <td>55.015873</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.608879</td>
      <td>12.824294</td>
    </tr>
    <tr>
      <th>min</th>
      <td>145.400000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>157.150000</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>161.900000</td>
      <td>51.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>167.500000</td>
      <td>65.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>193.900000</td>
      <td>89.000000</td>
    </tr>
  </tbody>
</table>
</div>



>info, describe 只能实现较少信息的展示，如果想要对一份数据集进行全面且有效的观察，特别是在列较多的情况下，推荐使用 pandas-profiling 包

### 2.特征统计函数

- 统计函数：sum,mean,median,var,std,max,min,idmax,idmin,count,quantile


```python
df_demo = df[['Height', 'Weight']]#取出高度和体重这两列
df_demo.mean()#平均值
```




    Height    163.218033
    Weight     55.015873
    dtype: float64




```python
df_demo.max()#最大值
```




    Height    193.9
    Weight     89.0
    dtype: float64




```python
df_demo.idxmax()#最大值的索引
```




    Height    193
    Weight      2
    dtype: int64




```python
df_demo.quantile(0.75)#分位数
```




    Height    167.5
    Weight     65.0
    Name: 0.75, dtype: float64




```python
df_demo.count()#非缺失值个数
```




    Height    183
    Weight    189
    dtype: int64



上面这些所有的函数，由于操作后返回的是标量，所以又称为聚合函数，它们有一个公共参数 axis ，默认为0代表逐列聚合，如果设置为1则表示逐行聚合


```python
df_demo.mean(axis=1).head()#身高加体重的均值，这里只列出前五行
```




    0    102.45
    1    118.25
    2    138.95
    3     41.00
    4    124.00
    dtype: float64



### 3.唯一值函数

- unique与nunique函数：得到唯一值组成的列表和唯一值的个数


```python
df['School'].unique()#查看schoo列的唯一值
```




    array(['Shanghai Jiao Tong University', 'Peking University',
           'Fudan University', 'Tsinghua University'], dtype=object)




```python
df['School'].nunique()#number of unique
```




    4



- value_counts：得到唯一值及其对应的频数


```python
df['School'].value_counts()
```




    Tsinghua University              69
    Shanghai Jiao Tong University    57
    Fudan University                 40
    Peking University                34
    Name: School, dtype: int64



- drop_duplicates：观察多个列组合的唯一值

drop_duplicates 等价于把 duplicated 为 True 的对应行剔除。


```python
df_demo = df[['Gender','Transfer','Name']]#取出三列
df_demo.drop_duplicates(['Gender', 'Transfer'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Transfer</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>N</td>
      <td>Gaopeng Yang</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>N</td>
      <td>Changqiang You</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Female</td>
      <td>NaN</td>
      <td>Peng You</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Male</td>
      <td>NaN</td>
      <td>Xiaopeng Shen</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Male</td>
      <td>Y</td>
      <td>Xiaojuan Qin</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Female</td>
      <td>Y</td>
      <td>Gaoli Feng</td>
    </tr>
  </tbody>
</table>
</div>



关键参数是 keep ，默认值 first 表示每个组合保留第一次出现的所在行， last 表示保留最后一次出现的所在行， False 表示把所有重复组合所在的行剔除。


```python
df_demo.drop_duplicates(['Gender', 'Transfer'], keep='last')#性别和transfer组合的唯一值
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Transfer</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>147</th>
      <td>Male</td>
      <td>NaN</td>
      <td>Juan You</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Male</td>
      <td>Y</td>
      <td>Chengpeng You</td>
    </tr>
    <tr>
      <th>169</th>
      <td>Female</td>
      <td>Y</td>
      <td>Chengquan Qin</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Female</td>
      <td>NaN</td>
      <td>Yanmei Qian</td>
    </tr>
    <tr>
      <th>197</th>
      <td>Female</td>
      <td>N</td>
      <td>Chengqiang Chu</td>
    </tr>
    <tr>
      <th>199</th>
      <td>Male</td>
      <td>N</td>
      <td>Chunpeng Lv</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_demo.drop_duplicates(['Name', 'Gender'],keep=False).head()
# 保留只出现过一次的性别和姓名组合
#只列出前五行
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Transfer</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>N</td>
      <td>Gaopeng Yang</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>N</td>
      <td>Changqiang You</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>N</td>
      <td>Mei Sun</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>N</td>
      <td>Gaojuan You</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Female</td>
      <td>N</td>
      <td>Xiaoli Qian</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['School'].drop_duplicates() # 在Series上也可以使用
```




    0    Shanghai Jiao Tong University
    1                Peking University
    3                 Fudan University
    5              Tsinghua University
    Name: School, dtype: object



- duplicated函数：返回了是否为唯一值的布尔列表，其 keep 参数与后者一致。其返回的序列，把重复元素设为 True ，否则为 False 


```python
df_demo.duplicated(['Gender', 'Transfer']).head()
```




    0    False
    1    False
    2     True
    3     True
    4     True
    dtype: bool




```python
df['School'].duplicated().head() # 在Series上也可以使用
```




    0    False
    1    False
    2     True
    3    False
    4     True
    Name: School, dtype: bool



### 4.替换函数

一般而言，替换操作是针对某一个列进行的，因此下面的例子都以 Series 举例。 pandas 中的替换函数可以归纳为三类：映射替换、逻辑替换、数值替换。

- replace函数


```python
df['Gender'].replace({'Female':0, 'Male':1}).head()#字典构造
```




    0    0
    1    1
    2    1
    3    0
    4    1
    Name: Gender, dtype: int64




```python
df['Gender'].replace(['Female', 'Male'], [0, 1]).head()#传入两个列表进行替换
```




    0    0
    1    1
    2    1
    3    0
    4    1
    Name: Gender, dtype: int64



指定 method 参数为 ffill 则为用前面一个最近的未被替换的值进行替换， bfill 则使用后面最近的未被替换的值进行替换


```python
s = pd.Series(['a', 1, 'b', 2, 1, 1, 'a'])
s.replace([1, 2], method='ffill')
```




    0    a
    1    a
    2    b
    3    b
    4    b
    5    b
    6    a
    dtype: object




```python
s.replace([1, 2], method='bfill')
```




    0    a
    1    b
    2    b
    3    a
    4    a
    5    a
    6    a
    dtype: object



>str.replace进行正则替换

- where与mask函数： where 函数在传入条件为 False 的对应行进行替换，而 mask 在传入条件为 True 的对应行进行替换，当不指定替换值时，替换为缺失值


```python
s = pd.Series([-1, 1.2345, 100, -50])
s.where(s<0)
```




    0    -1.0
    1     NaN
    2     NaN
    3   -50.0
    dtype: float64




```python
s.where(s<0, 100)
```




    0     -1.0
    1    100.0
    2    100.0
    3    -50.0
    dtype: float64




```python
s.mask(s<0)
```




    0         NaN
    1      1.2345
    2    100.0000
    3         NaN
    dtype: float64




```python
s.mask(s<0, -50)#用-50遮盖掉小于0的部分
```




    0    -50.0000
    1      1.2345
    2    100.0000
    3    -50.0000
    dtype: float64




```python
s_condition= pd.Series([True,False,False,True])
s_condition= pd.Series([True,False,False,True],index=s.index)
#传入的条件布尔序列需要与被调用的series索引一致（index）
s.mask(s_condition, -50)
```




    0    -50.0000
    1      1.2345
    2    100.0000
    3    -50.0000
    dtype: float64



- round,abs,clip函数：表示取整，取绝对值和截断


```python
s = pd.Series([-1, 1.2345, 100, -50])
s.round(2)#取小数点后面两位数
```




    0     -1.00
    1      1.23
    2    100.00
    3    -50.00
    dtype: float64




```python
s.abs()
```




    0      1.0000
    1      1.2345
    2    100.0000
    3     50.0000
    dtype: float64




```python
s.clip(0, 2) # 前两个数分别表示上下截断边界
#小于0的用0替换，大于2的用2替换
```




    0    0.0000
    1    1.2345
    2    2.0000
    3    0.0000
    dtype: float64



在 clip 中，超过边界的只能截断为边界值，如果要把超出边界的替换为自定义的值，应当如何做？


```python
a=s.mask(s<0,-10)
a.mask(s>2,10)#用where达到一样的效果
```




    0   -10.0000
    1     1.2345
    2    10.0000
    3   -10.0000
    dtype: float64



### 5.排序函数

- sort_values：值排序

先利用 set_index 方法把年级和姓名两列作为索引


```python
df_demo = df[['Grade', 'Name', 'Height','Weight']].set_index(['Grade','Name'])
df_demo.describe
```




    <bound method NDFrame.describe of                           Height  Weight
    Grade     Name                          
    Freshman  Gaopeng Yang     158.9    46.0
              Changqiang You   166.5    70.0
    Senior    Mei Sun          188.9    89.0
    Sophomore Xiaojuan Sun       NaN    41.0
              Gaojuan You      174.0    74.0
    ...                          ...     ...
    Junior    Xiaojuan Sun     153.9    46.0
    Senior    Li Zhao          160.9    50.0
              Chengqiang Chu   153.9    45.0
              Chengmei Shen    175.3    71.0
    Sophomore Chunpeng Lv      155.7    51.0
    
    [200 rows x 2 columns]>



对身高进行排序，默认参数 ascending=True 为升序：


```python
df_demo.sort_values('Height').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Junior</th>
      <th>Xiaoli Chu</th>
      <td>145.4</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Gaomei Lv</th>
      <td>147.3</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <th>Peng Han</th>
      <td>147.8</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Changli Lv</th>
      <td>148.7</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <th>Changjuan You</th>
      <td>150.5</td>
      <td>40.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_demo.sort_values('Height', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Senior</th>
      <th>Xiaoqiang Qin</th>
      <td>193.9</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>Mei Sun</th>
      <td>188.9</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>Gaoli Zhao</th>
      <td>186.5</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <th>Qiang Han</th>
      <td>185.3</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Qiang Zheng</th>
      <td>183.9</td>
      <td>87.0</td>
    </tr>
  </tbody>
</table>
</div>



排序中，经常遇到多列排序的问题，比如在体重相同的情况下，对身高进行排序，并且保持身高降序排列，体重升序排列


```python
 df_demo.sort_values(['Weight','Height'],ascending=[True,False]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sophomore</th>
      <th>Peng Han</th>
      <td>147.8</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Senior</th>
      <th>Gaomei Lv</th>
      <td>147.3</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Junior</th>
      <th>Xiaoli Chu</th>
      <td>145.4</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>Sophomore</th>
      <th>Qiang Zhou</th>
      <td>150.5</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>Freshman</th>
      <th>Yanqiang Xu</th>
      <td>152.4</td>
      <td>38.0</td>
    </tr>
  </tbody>
</table>
</div>



- sort_index：索引排序

索引排序的用法和值排序完全一致，只不过元素的值在索引中，此时需要指定索引层的名字或者层号，用参数 level 表示。另外，需要注意的是字符串的排列顺序由字母顺序决定


```python
df_demo.sort_index(level=['Grade','Name'],ascending=[True,False]).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
    <tr>
      <th>Grade</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Freshman</th>
      <th>Yanquan Wang</th>
      <td>163.5</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>Yanqiang Xu</th>
      <td>152.4</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Yanqiang Feng</th>
      <td>162.3</td>
      <td>51.0</td>
    </tr>
    <tr>
      <th>Yanpeng Lv</th>
      <td>NaN</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>Yanli Zhang</th>
      <td>165.1</td>
      <td>52.0</td>
    </tr>
  </tbody>
</table>
</div>



### 6.apply函数

apply 的参数是一个以序列为输入的函数


```python
df_demo = df[['Height', 'Weight']]
def my_mean(x):
    res = x.mean()
    return res

df_demo.apply(my_mean)
```




    Height    163.218033
    Weight     55.015873
    dtype: float64




```python
df_demo.apply(lambda x:x.mean())
```




    Height    163.218033
    Weight     55.015873
    dtype: float64




```python
df_demo.apply(lambda x:x.mean(), axis=1).head()
#若指定 axis=1 ，那么每次传入函数的就是行元素组成的 Series 
#其结果与之前的逐行均值结果一致，即身高和体重的平均
```




    0    102.45
    1    118.25
    2    138.95
    3     41.00
    4    124.00
    dtype: float64



这里再举一个例子： mad 函数返回的是一个序列中偏离该序列均值的绝对值大小的均值，例如序列1,3,7,10中，均值为5.25，每一个元素偏离的绝对值为4.25,2.25,1.75,4.75，这个偏离序列的均值为3.25。


```python
df_demo.apply(lambda x:(x-x.mean()).abs().mean())
```




    Height     6.707229
    Weight    10.391870
    dtype: float64




```python
df_demo.mad()
```




    Height     6.707229
    Weight    10.391870
    dtype: float64



## 四、窗口对象

pandas 中有3类窗口，分别是滑动窗口 `rolling` 、扩张窗口 `expanding` 以及指数加权窗口 `ewm` 。

### 1.滑窗对象

对一个序列使用 .rolling 得到滑窗对象，其最重要的参数为窗口大小 window 


```python
s = pd.Series([1,2,3,4,5])
roller = s.rolling(window = 3)
roller
```




    Rolling [window=3,center=False,axis=0]



在得到了滑窗对象后，能够使用相应的聚合函数进行计算，需要注意的是窗口包含当前行所在的元素


```python
roller.mean()
#0:nan+nan+1/3
#1:nan+1+2/3
#2:1+2+3/3
#3：2+3+4/3
#4：3+4+5/3
```




    0    NaN
    1    NaN
    2    2.0
    3    3.0
    4    4.0
    dtype: float64




```python
roller.sum()
```




    0     NaN
    1     NaN
    2     6.0
    3     9.0
    4    12.0
    dtype: float64




```python
s2 = pd.Series([1,2,6,16,30])
roller.cov(s2)
#2：1，2，3与1,2,6
#ave1=(1+2+3)/3=2,ave2=(1+2+6)/3=3
#cov=(2+0+3)/(3-1)=2.5
#3：2,3,4与2,6,16
#4:3,4,5与6,16,30
```




    0     NaN
    1     NaN
    2     2.5
    3     7.0
    4    12.0
    dtype: float64




```python
print(np.cov([1,2,3],[1,2,6]))
print(np.cov([2,3,4],[2,6,16]))
print(np.cov([3,4,5],[6,16,30]))
```

    [[1.  2.5]
     [2.5 7. ]]
    [[ 1.  7.]
     [ 7. 52.]]
    [[  1.          12.        ]
     [ 12.         145.33333333]]
    


```python
roller.corr(s2)
```




    0         NaN
    1         NaN
    2    0.944911
    3    0.970725
    4    0.995402
    dtype: float64



- 滑动窗口加apply


```python
roller.apply(lambda x:x.mean())
```




    0    NaN
    1    NaN
    2    2.0
    3    3.0
    4    4.0
    dtype: float64



- shift，diff，pct_change类滑窗函数

公共参数为 periods=n ，默认为1，分别表示取向前第 n 个元素的值、与向前第 n 个元素做差、与向前第 n 个元素相比计算增长率。这里的 n 可以为负，表示反方向的类似操作。


```python
s = pd.Series([1,3,6,10,15])
s.shift(2)#取向前第 n 个元素的值
s.rolling(3).apply(lambda x:list(x)[0])
```




    0    NaN
    1    NaN
    2    1.0
    3    3.0
    4    6.0
    dtype: float64




```python
s.shift(-1)
```




    0     3.0
    1     6.0
    2    10.0
    3    15.0
    4     NaN
    dtype: float64




```python
s.diff(3)#与向前第 n 个元素做差
#3:10-1 4:15-3 
s.rolling(4).apply(lambda x:list(x)[-1]-list(x)[0])
```




    0     NaN
    1     NaN
    2     NaN
    3     9.0
    4    12.0
    dtype: float64




```python
s.diff(-2)
```




    0   -5.0
    1   -7.0
    2   -9.0
    3    NaN
    4    NaN
    dtype: float64




```python
s.pct_change()#与向前第 n 个元素相比计算增长率
#1:(3-1)/1
#2:(6-3)/3
s.rolling(2).apply(lambda x:list(x)[-1]/list(x)[0]-1)
```




    0         NaN
    1    2.000000
    2    1.000000
    3    0.666667
    4    0.500000
    dtype: float64



#### 【练一练】

`rolling`对象的默认窗口方向都是向前的，某些情况下用户需要向后的窗口，例如对1,2,3设定向后窗口为2的`sum`操作，结果为3,5,NaN，此时应该如何实现向后的滑窗操作？（提示：使用`shift`）



```python
s=pd.Series([1,2,3,4,5])
s.shape[0]
s.rolling(2).sum()
s[::-1]+s[::-1].shift(-1)#先把原数组逆序，再将逆序后的shift-1
#两者相加就可以求出窗口为2的向后滑窗
#若窗口为3，就再shift-2一次
```




    4    9.0
    3    7.0
    2    5.0
    1    3.0
    0    NaN
    dtype: float64



### 2.扩张窗口

扩张窗口又称累计窗口，可以理解为一个动态长度的窗口，其窗口的大小就是从序列开始处到具体操作的对应位置，其使用的聚合函数会作用于这些逐步扩张的窗口上。具体地说，设序列为a1, a2, a3, a4，则其每个位置对应的窗口即[a1]、[a1, a2]、[a1, a2, a3]、[a1, a2, a3, a4]。


```python
s = pd.Series([1, 3, 6, 10])
s.expanding().mean()
```




    0    1.000000
    1    2.000000
    2    3.333333
    3    5.000000
    dtype: float64



#### 【练一练】
`cummax, cumsum, cumprod`函数是典型的类扩张窗口函数，请使用`expanding`对象依次实现它们。



```python
s=pd.Series([1,3,5,7,9])
print(s.cummax())
print(s.cumsum())#累加
print(s.cumprod())#累乘
```

    0    1
    1    3
    2    5
    3    7
    4    9
    dtype: int64
    0     1
    1     4
    2     9
    3    16
    4    25
    dtype: int64
    0      1
    1      3
    2     15
    3    105
    4    945
    dtype: int64
    


```python
print(s.expanding().max())
print(s.expanding().sum())
s.expanding().apply(lambda x:x.prod())
#s.expanding().prod()报错
```

    0    1.0
    1    3.0
    2    5.0
    3    7.0
    4    9.0
    dtype: float64
    0     1.0
    1     4.0
    2     9.0
    3    16.0
    4    25.0
    dtype: float64
    




    0      1.0
    1      3.0
    2     15.0
    3    105.0
    4    945.0
    dtype: float64



## 五、练习
### Ex1：口袋妖怪数据集

现有一份口袋妖怪的数据集，下面进行一些背景说明：

* `#`代表全国图鉴编号，不同行存在相同数字则表示为该妖怪的不同状态

* 妖怪具有单属性和双属性两种，对于单属性的妖怪，`Type 2`为缺失值
* `Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed`分别代表种族值、体力、物攻、防御、特攻、特防、速度，其中种族值为后6项之和


```python
df = pd.read_csv('D:\\datawhale\\joyful-pandas\\data\\pokemon.csv')
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



1. 对`HP, Attack, Defense, Sp. Atk, Sp. Def, Speed`进行加总，验证是否为`Total`值。


```python
dftotal=df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)
((df['Total']-dftotal)<1e-15).all()
```




    True



2. 对于`#`重复的妖怪只保留第一条记录，解决以下问题：


* 求第一属性的种类数量和前三多数量对应的种类
* 求第一属性和第二属性的组合种类
* 求尚未出现过的属性组合

- a


```python
df_demo=df.drop_duplicates(['#'],keep='first')#去除重复的
df_demo['Type 1'].value_counts().head(3)#前三多的种类为水，一般与草
```




    Water     105
    Normal     93
    Grass      66
    Name: Type 1, dtype: int64




```python
df_demo['Type 1'].nunique()#第一属性的种类有18种
```




    18



- b


```python
df_demo.drop_duplicates(['Type 1','Type 2']).shape[0]
#第一属性和第二属性组合不重复的种类有143种
```




    143



- c：type1 与type2的没出现在列表里面的组合


```python
#所有组合
L_full = [i+' '+j for i in df['Type 1'].unique() for j in (df['Type 1'].unique().tolist() + [''])]
#问题？
#[i+' '+j for i in df['Type 1'].unique() for j in (df['Type 2'].unique().tolist() + [''])]
#报错？TypeError: can only concatenate str (not "float") to str
#因为type2里面有nan值。将nan替换成''，如下所示即可
len(L_full)
```




    342




```python
len([i+' '+j for i in df['Type 1'].unique() for j in (df['Type 2'].replace(np.nan, '').unique())])
```




    342




```python
#列表里面的组合
L_part = [i+' '+j for i, j in zip(df['Type 1'], df['Type 2'].replace(np.nan, ''))]
#打包就可以了？
print(len(L_part))
print(len(set(L_part)))
```

    800
    154
    


```python
#对比两者的差距
res = set(L_full).difference(set(L_part))#L_full对应L_part的差集，L_full里面有，但L_part里面没有
len(res)
#set去除了重复的数据，故而set(L_part)有154个，L_part有800个
```




    188




```python
df['Type 1'].nunique()
```




    18




```python
df['Type 2'].nunique()
```




    18



3. 按照下述要求，构造`Series`：


* 取出物攻，超过120的替换为`high`，不足50的替换为`low`，否则设为`mid`
* 取出第一属性，分别用`replace`和`apply`替换所有字母为大写
* 求每个妖怪六项能力的离差，即所有能力中偏离中位数最大的值，添加到`df`并从大到小排序

- a

利用mask迭代


```python
df['Attack'].mask(df['Attack']>120,'high').mask(df['Attack']<50,'low').mask((df['Attack']>=50)&(df['Attack']<=120),'mid').head()
```




    0    low
    1    mid
    2    mid
    3    mid
    4    mid
    Name: Attack, dtype: object



- b


```python
df['Type 1'].replace({i:i.upper() for i in df['Type 1'].unique()}).head()
#建立字典映射
#str.upper()是将所有字符中的小写字母换成大写字母
#str.lower()是将所有字符中的大写字母换成小写字母
#str.capitalize()是把第一个字母转化成大写字母，其余转化成小写字母
#str.title()是把每个单词的第一个字母转化成大写，其他为小写
```




    0    GRASS
    1    GRASS
    2    GRASS
    3    GRASS
    4     FIRE
    Name: Type 1, dtype: object




```python
df['Type 1'].apply(lambda x:x.upper()).head()
```




    0    GRASS
    1    GRASS
    2    GRASS
    3    GRASS
    4     FIRE
    Name: Type 1, dtype: object



- c

第一步：取出六项能力的离差，两种定义离差的方法


```python
df['deviation']=df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].apply(lambda x:np.max((x-x.median()).abs()),axis=1).head()
```


```python
df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].apply(lambda x:np.max(x)-x.median(),axis=1).head()
```




    0    16.0
    1    17.5
    2    17.5
    3    13.0
    4    14.0
    dtype: float64




```python
df.head(2)#添加了一列
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>17.5</td>
    </tr>
  </tbody>
</table>
</div>



第二步：依据离差从大到小排序


```python
df.sort_values('deviation',ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>Name</th>
      <th>Type 1</th>
      <th>Type 2</th>
      <th>Total</th>
      <th>HP</th>
      <th>Attack</th>
      <th>Defense</th>
      <th>Sp. Atk</th>
      <th>Sp. Def</th>
      <th>Speed</th>
      <th>deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>VenusaurMega Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>625</td>
      <td>80</td>
      <td>100</td>
      <td>123</td>
      <td>122</td>
      <td>120</td>
      <td>80</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Ivysaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>405</td>
      <td>60</td>
      <td>62</td>
      <td>63</td>
      <td>80</td>
      <td>80</td>
      <td>60</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Venusaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>525</td>
      <td>80</td>
      <td>82</td>
      <td>83</td>
      <td>100</td>
      <td>100</td>
      <td>80</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Bulbasaur</td>
      <td>Grass</td>
      <td>Poison</td>
      <td>318</td>
      <td>45</td>
      <td>49</td>
      <td>49</td>
      <td>65</td>
      <td>65</td>
      <td>45</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Charmander</td>
      <td>Fire</td>
      <td>NaN</td>
      <td>309</td>
      <td>39</td>
      <td>52</td>
      <td>43</td>
      <td>60</td>
      <td>50</td>
      <td>65</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>



## Ex2：指数加权窗口
1. 作为扩张窗口的`ewm`窗口

在扩张窗口中，用户可以使用各类函数进行历史的累计指标统计，但这些内置的统计函数往往把窗口中的所有元素赋予了同样的权重。事实上，可以给出不同的权重来赋给窗口中的元素，指数加权窗口就是这样一种特殊的扩张窗口。

其中，最重要的参数是`alpha`，它决定了默认情况下的窗口权重为$w_i=(1−\alpha)^i,i\in\{0,1,...,t\}$，其中$i=t$表示当前元素，$i=0$表示序列的第一个元素。

从权重公式可以看出，离开当前值越远则权重越小，若记原序列为$x$，更新后的当前元素为$y_t$，此时通过加权公式归一化后可知：

$$
\begin{split}y_t &=\frac{\sum_{i=0}^{t} w_i x_{t-i}}{\sum_{i=0}^{t} w_i} \\
&=\frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
+ (1 - \alpha)^{t} x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
+ (1 - \alpha)^{t-1}}\\\end{split}
$$

对于`Series`而言，可以用`ewm`对象如下计算指数平滑后的序列：


```python
np.random.seed(0)
s = pd.Series(np.random.randint(-1,2,30).cumsum())
s.ewm(alpha=0.2).mean().head()
```




    0   -1.000000
    1   -1.000000
    2   -1.409836
    3   -1.609756
    4   -1.725845
    dtype: float64



请用 expanding 窗口实现。


```python
def ewm_func(x,alpha=0.2):
    weight=(1-alpha)**np.arange(x.shape[0])[::-1]#定义权重，离t越远，权重越小
    res=(weight*x).sum()/weight.sum()
    return res

s.expanding().apply(ewm_func).head()
```




    0   -1.000000
    1   -1.000000
    2   -1.409836
    3   -1.609756
    4   -1.725845
    dtype: float64



2. 作为滑动窗口的`ewm`窗口

从第1问中可以看到，`ewm`作为一种扩张窗口的特例，只能从序列的第一个元素开始加权。现在希望给定一个限制窗口`n`，只对包含自身最近的`n`个窗口进行滑动加权平滑。请根据滑窗函数，给出新的`wi`与`yt`的更新公式，并通过`rolling`窗口实现这一功能。

窗口权重为$w_i=(1−\alpha)^i,i\in\{0,1,...,n\}$，$y_t$为：
$$
\begin{split}y_t &=\frac{\sum_{i=0}^{n-1} w_i x_{t-i}}{\sum_{i=0}^{n-1} w_i} \\
&=\frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
+ (1 - \alpha)^{n-1} x_{t-(n-1)}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
+ (1 - \alpha)^{n-1}}\\\end{split}
$$



```python
s.rolling(window=4).apply(ewm_func).head()
#窗口固定为4时，每次传入到ewm_func里面的都是四个数的数列，权重就是0.8**3...0.8
#0.8**3*数列的第一个数(x_t-3)...0.8**数列的最后一个数（x_t）
```




    0         NaN
    1         NaN
    2         NaN
    3   -1.609756
    4   -1.826558
    dtype: float64


