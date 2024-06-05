import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('billbill.csv', encoding='utf-8-sig')

# 选择需要进行聚类分析的列
data = df[['like_count', 'video_play_count', 'video_danmaku', 'video_comment']]

# 检查并处理重复数据
data_dedup = data.drop_duplicates().copy()

# 数据标准化处理
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_dedup)

# 使用手肘法确定最佳聚类数
def determine_optimal_clusters(data, max_k):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('聚类分析结果/elbow_method.png', dpi=300)
    plt.show()
    return inertia

# 确定最佳聚类数
max_k = min(10, len(data_dedup) - 1)  # 设置max_k为数据样本数减1，以避免聚类数超过样本数
inertia = determine_optimal_clusters(scaled_data, max_k)
optimal_k = np.argmax(np.diff(inertia)) + 2  # 手肘法找出“肘部”位置

# 使用最佳聚类数进行KMeans聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_data)
data_dedup['cluster'] = kmeans.labels_

# 将去重后的数据和聚类结果合并回原始数据
df = df.merge(data_dedup[['like_count', 'video_play_count', 'video_danmaku', 'video_comment', 'cluster']],
              on=['like_count', 'video_play_count', 'video_danmaku', 'video_comment'],
              how='left')

# 创建输出目录
output_dir = '聚类分析结果'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 将结果保存到新的CSV文件
df.to_csv(os.path.join(output_dir, 'billbill_clusters.csv'), index=False, encoding='utf-8-sig')

# 打印聚类结果
print("聚类结果：")
print(df)

# 可视化聚类结果
plt.figure(figsize=(16, 10))

# 绘制聚类结果的散点图
sns.scatterplot(data=df, x='like_count', y='video_play_count', hue='cluster', palette='viridis')
plt.title('KMeans Clustering of Like Count and Video Play Count')
plt.xlabel('Like Count')
plt.ylabel('Video Play Count')
plt.legend(title='Cluster')
plt.savefig(os.path.join(output_dir, 'like_video_cluster.png'), dpi=300)
plt.show()

sns.scatterplot(data=df, x='video_danmaku', y='video_comment', hue='cluster', palette='viridis')
plt.title('KMeans Clustering of Video Danmaku and Video Comment')
plt.xlabel('Video Danmaku')
plt.ylabel('Video Comment')
plt.legend(title='Cluster')
plt.savefig(os.path.join(output_dir, 'danmaku_comment_cluster.png'), dpi=300)
plt.show()
