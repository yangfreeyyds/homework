# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# 总体情绪分布数据
emotion_labels = ['Positive', 'Neutral', 'Negative']
emotion_sizes = [241, 170, 69]
emotion_colors = ['#66b3ff', '#ffcc99', '#ff9999']
emotion_explode = (0.1, 0, 0)  # 突出显示积极情绪

# 积极情绪分段数据
positive_labels = ['General (0-10)', 'Moderate (10-20)', 'High (20+)']
positive_sizes = [159, 69, 13]
positive_colors = ['#66b3ff', '#99ff99', '#ffcc99']

# 消极情绪分段数据
negative_labels = ['General (-10-0)', 'Moderate (-20--10)', 'High (<-20)']
negative_sizes = [51, 12, 2]
negative_colors = ['#ff9999', '#ffcc99', '#66b3ff']

# 创建饼图：总体情绪分布
plt.figure(figsize=(10, 6))
plt.pie(emotion_sizes, explode=emotion_explode, labels=emotion_labels, colors=emotion_colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Overall Emotion Distribution')
plt.axis('equal')  # 保持饼图为圆形
plt.savefig('情绪可视化(1).png')
plt.show()

# 创建柱状图：积极情绪分段统计
plt.figure(figsize=(10, 6))
plt.bar(positive_labels, positive_sizes, color=positive_colors)
plt.title('Positive Emotion Segments')
plt.xlabel('Positive Emotion Segments')
plt.ylabel('Count')
for i in range(len(positive_labels)):
    plt.text(i, positive_sizes[i] + 3, f'{positive_sizes[i]}', ha='center')
plt.savefig('情绪可视化(2).png')
plt.show()

# 创建柱状图：消极情绪分段统计
plt.figure(figsize=(10, 6))
plt.bar(negative_labels, negative_sizes, color=negative_colors)
plt.title('Negative Emotion Segments')
plt.xlabel('Negative Emotion Segments')
plt.ylabel('Count')
for i in range(len(negative_labels)):
    plt.text(i, negative_sizes[i] + 1, f'{negative_sizes[i]}', ha='center')
plt.savefig('情绪可视化(3).png')
plt.show()
