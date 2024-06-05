import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('billbill.csv', encoding='utf-8-sig')

# 使用IQR方法进行异常值处理
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]

# 检查每一列的异常值
like_count_outliers = detect_outliers_iqr(df['like_count'])
video_play_count_outliers = detect_outliers_iqr(df['video_play_count'])
video_danmaku_outliers = detect_outliers_iqr(df['video_danmaku'])
video_comment_outliers = detect_outliers_iqr(df['video_comment'])

# 处理异常值，可以选择删除或替换为中位数等
for column in ['like_count', 'video_play_count', 'video_danmaku', 'video_comment']:
    outliers = detect_outliers_iqr(df[column])
    df[column] = np.where((df[column] < outliers.min()) | (df[column] > outliers.max()), df[column].median(), df[column])

# 将结果保存到新的CSV文件
df.to_csv('异常值处理结果/billbill.csv', index=False, encoding='utf-8-sig')

# 打印处理后的数据
print("处理后的每个月的累加数据：")
print(df)

# 绘制箱线图和小提琴图
plt.figure(figsize=(16, 10))

# 综合的箱线图
plt.subplot(5, 2, 1)
sns.boxplot(data=df[['like_count', 'video_play_count', 'video_danmaku', 'video_comment']])
plt.title('Box Plot of Like Count, Video Play Count, Video Danmaku, and Video Comment')

# 综合的小提琴图
plt.subplot(5, 2, 2)
sns.violinplot(data=df[['like_count', 'video_play_count', 'video_danmaku', 'video_comment']])
plt.title('Violin Plot of Like Count, Video Play Count, Video Danmaku, and Video Comment')

# 按列分别绘制箱线图
columns = ['like_count', 'video_play_count', 'video_danmaku', 'video_comment']
for i, column in enumerate(columns):
    plt.subplot(5, 2, i+3)
    sns.boxplot(data=df[column])
    plt.title(f'Box Plot of {column}')

# 按列分别绘制小提琴图
columns = ['like_count', 'video_play_count', 'video_danmaku', 'video_comment']
for i, column in enumerate(columns):
    plt.subplot(5, 2, i+7)
    sns.violinplot(data=df[column])
    plt.title(f'Violin Plot of {column}')

# 显示图表
plt.tight_layout()

# 保存图片到指定地址
plt.savefig('异常值处理结果/result.png', dpi=300)

plt.show()
