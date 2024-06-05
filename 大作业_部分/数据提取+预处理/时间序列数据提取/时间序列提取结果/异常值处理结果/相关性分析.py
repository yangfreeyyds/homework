import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('billbill.csv')

# 选择需要进行相关性分析的列
columns_to_evaluate = ['like_count', 'video_play_count', 'video_danmaku', 'video_comment']

# 计算相关性矩阵
correlation_matrix = data[columns_to_evaluate].corr()

# 打印相关性矩阵
print(correlation_matrix)

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('相关性分析/result.png')
plt.show()
