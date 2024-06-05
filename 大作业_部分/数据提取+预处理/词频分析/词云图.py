import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 读取txt文件中的数据
file_path = 'result--一般性行处理-去重_分词后_词频.txt'  # 请替换为实际文件路径
with open(file_path, 'r', encoding='gbk') as file:
    lines = file.readlines()
    print(lines)
# 解析前50个关键词及其频率
words_freq = {}
for line in lines[:50]:
    word, freq = line.split()
    words_freq[word] = int(freq)

# 创建词云
# 创建词云，指定中文字体
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    font_path='C:/Windows/Fonts/simhei.ttf'
).generate_from_frequencies(words_freq)

# 显示词云图
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 50 Keywords Word Cloud')
plt.savefig('词云图.png')
plt.show()
