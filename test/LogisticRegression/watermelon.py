import numpy as np
import pandas as pd
from hym.LogisticRegression import BiClassfier
import matplotlib.pyplot as plt

data = {
    "色泽": ["青绿", "乌黑", "乌黑", "青绿", "浅白", "青绿", "乌黑", "乌黑", "乌黑", "青绿", "浅白", "浅白", "青绿", "浅白", "乌黑", "浅白", "青绿"],
    "根蒂": ["蜷缩", "蜷缩", "蜷缩", "蜷缩", "蜷缩", "稍蜷", "稍蜷", "稍蜷", "稍蜷", "硬挺", "硬挺", "蜷缩", "稍蜷", "稍蜷", "稍蜷", "蜷缩", "蜷缩"],
    "敲声": ["浊响", "沉闷", "浊响", "沉闷", "浊响", "浊响", "浊响", "浊响", "沉闷", "清脆", "清脆", "浊响", "浊响", "沉闷", "浊响", "浊响", "沉闷"],
    "纹理": ["清晰", "清晰", "清晰", "清晰", "清晰", "清晰", "稍糊", "清晰", "稍糊", "清晰", "模糊", "模糊", "稍糊", "稍糊", "清晰", "模糊", "稍糊"],
    "脐部": ["凹陷", "凹陷", "凹陷", "凹陷", "凹陷", "稍凹", "稍凹", "稍凹", "稍凹", "平坦", "平坦", "平坦", "凹陷", "凹陷", "稍凹", "平坦", "稍凹"],
    "触感": ["硬滑", "硬滑", "硬滑", "硬滑", "硬滑", "软粘", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "硬滑", "软粘", "硬滑", "硬滑"],
    "密度": [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
    "含糖率": [0.46, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103],
    "好瓜": ["是", "是", "是", "是", "是", "是", "是", "是", "否", "否", "否", "否", "否", "否", "否", "否", "否"]
}

df = pd.DataFrame(data)
df_encoded = df.copy()
for column in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[column], _ = pd.factorize(df_encoded[column])

numpy_array = df_encoded.to_numpy()
X = df_encoded.drop(columns=["好瓜"]).to_numpy()  # Features
y = df_encoded["好瓜"].to_numpy()  # Labels

# use last 2 features
clr = BiClassfier(X[:, -8:], y, lr=0.015, epoch=10000)
clr.fit()
L = clr.history()
print(clr.w, clr.b)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

plt.plot(L, label='loss')
plt.title('loss vs epoch for clr')
plt.tick_params(axis='both', which='both', direction='in')
plt.show()

# plot dicision edge
xx = X[:, -2:]
labels = y

# Create a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(xx[labels == 0][:, 0], xx[labels == 0][:, 1], color='blue', label='Label 0')
plt.scatter(xx[labels == 1][:, 0], xx[labels == 1][:, 1], color='red', label='Label 1')
plt.title('Scatter plot of points labeled as 0 and 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.tick_params(axis='both', which='both', direction='in')

ww = clr.w.flatten()[-2:]
print(ww)
print(clr.b)

ix = np.linspace(0, 1, 100)
plt.plot(ix, -(ix * ww[0] + clr.b) / ww[1])

# Show the plot
plt.show()


# mtc = Metrics(y_pred=clr(X[:, -8:]), y=y, classes=2)
# print(mtc)
