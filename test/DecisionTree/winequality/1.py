import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 加载数据
df = pd.read_excel('./winequality.xlsx')
df = df.rename(columns={'quality label': 'label'})

# 确定特征列和标签列
X = df[df.columns[:-1]]
y = df['label']

# 对所有特征进行离散化处理
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_discretized = discretizer.fit_transform(X)

# 分割数据集为训练集和验证集
sed = np.random.randint(1000)
print(f'sed is : {sed}')
X_train, X_valid, y_train, y_valid = train_test_split(
    X_discretized, y, test_size=0.5, random_state=sed)

# 构建决策树模型
tree = DecisionTreeClassifier(criterion='entropy', random_state=sed)  # 使用信息熵作为划分标准，类似于C4.5

# 训练模型
tree.fit(X_train, y_train)

# 预测并评估模型
y_pred = tree.predict(X_valid)

# 计算多个评测指标
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred, average='weighted')  # 使用'weighted'来考虑不平衡的数据集
recall = recall_score(y_valid, y_pred, average='weighted')
f1 = f1_score(y_valid, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_valid, y_pred)

# 打印评估结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)

# 提供更详细的分类报告，包含每个类的精确率、召回率、F1值和支持度
class_report = classification_report(y_valid, y_pred)
print('Classification Report:')
print(class_report)
