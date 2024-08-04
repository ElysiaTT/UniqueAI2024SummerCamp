# 第一周任务感想
# task1实现心得

## 运算结果
![运行结果](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task1/yunxingjieguo.png)
## 数据处理部分

对于数据的缺失，删除 'Name' 和 'Ticket' 列，避免对后面归一化的字符串造成影响

Cabin影响不大，所以用0表示没有记录，1代表有记录

embarked数据只缺失两个，所以选择众数填补。

age的值缺失较多，并且均值有代表性，所以选择用均值填补。

## one-hot编码部分

Embarked列有三个可能的值：C、Q和S，通过One-Hot编码，每个可能的值都将成为一个独立的特征;sex与Pclass同理

![缺失值处理与one-hot编码](https://github.com/ElysiaTT/UniqueAI2024SummerCamp/blob/main/task1/chuliguocheng.png)

## 预测回归部分

不使用深度学习库

首先定义特征 X 和目标变量 y，再添加偏置项并初始化参数，sigmoid函数与定义损失函数。就可以实现梯度下降，最后计算准确率
## 代码如下
```
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
data = pd.read_csv('titanic_data.csv')

# 删除 'Name' 和 'Ticket' 列
data.drop(columns=['Name', 'Ticket'], inplace=True)

# 使用均值填充 'Age' 列中的缺失值
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data['Age'] = imp.fit_transform(data[['Age']])
data['Age'] = data['Age'].round(1)

# 将 'Cabin' 标记为0，1
data['Cabin'] = data['Cabin'].notna().astype(int)

# 使用众数填充 'Embarked' 列中的缺失值
mode = data['Embarked'].mode()[0]
data['Embarked'].replace(np.nan, mode, inplace=True)

# 归一化
scaler = MinMaxScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# 对 'Pclass'、'Sex' 和 'Embarked' 进行 One-Hot 编码
#Embarked列有三个可能的值：C、Q和S，通过One-Hot编码，每个可能的值都将成为一个独立的特征
data = pd.get_dummies(data, columns=['Sex', 'Pclass', 'Embarked'])
data[['Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3','Embarked_C','Embarked_Q','Embarked_S']] = data[['Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3','Embarked_C','Embarked_Q','Embarked_S']].astype(int)

print(data.head(10))
data.to_csv("onehot_encoded_data.csv", index=False)

# 定义特征 X 和目标变量 y
X = data.drop(columns=['Survived']).values.astype(float)
y = data['Survived'].values.astype(float)

# 添加偏置项
X = np.hstack((np.ones((X.shape[0], 1)), X))

# 初始化参数
theta = np.zeros(X.shape[1])

# 定义 sigmoid 函数
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 定义损失函数
def compute_loss(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    loss = (1 / (2 * m)) * np.sum((h - y)**2)
    return loss

# 梯度下降函数
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    loss_history = []

    for i in range(num_iters):
        gradient = (1 / m) * (X.T @ (sigmoid(X @ theta) - y))
        theta -= alpha * gradient
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)

    return theta, loss_history

# 设置梯度下降参数
alpha = 0.01
num_iters = 1000

# 训练逻辑回归模型
theta, loss_history = gradient_descent(X, y, theta, alpha, num_iters)

# 预测函数
def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

# 计算训练集上的准确率
predictions = predict(X, theta)
accuracy = np.mean(predictions == y) * 100
print(f'Training Accuracy: {accuracy:.1f}%')
```
## 改进
后续可以通过调参改进准确度
## 心得
本次学习了csv的读取与提交的python函数，并学习了logistics回归的手动实现方法，也明白了模型的构建需要不断调整与多方面改进。
