import pandas as pd

# 读取CSV文件
df = pd.read_csv('../原始数据/billbill.csv', encoding='gbk')

# 读取第五列的时间戳并转换为日期时间格式
create_time = df.iloc[:, 4]
df['create_time'] = pd.to_datetime(create_time, unit='s')

# 读取各列数据
df['like_count'] = df.iloc[:, 8]  # 点赞量
df['video_play_count'] = df.iloc[:, 9]  # 播放量
df['video_danmaku'] = df.iloc[:, 10]  # 弹幕量
df['video_comment'] = df.iloc[:, 11]  # 评论量
df['title'] = df.iloc[:,2]  #标题

# 按时间排序
df = df.sort_values(by='create_time')

# 将时间戳转换为年月格式
df['year_month'] = df['create_time'].dt.to_period('M')

# 按年月分组并累加各列
grouped_df = df.groupby('year_month').agg({
    'like_count': 'sum',
    'video_play_count': 'sum',
    'video_danmaku': 'sum',
    'video_comment': 'sum'
}).reset_index()

# 设置Pandas选项以显示完整的DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 查看结果
print("每个月的累加数据：")
print(grouped_df)

# 将结果保存到CSV文件
grouped_df.to_csv('时间序列提取结果/billbill.csv', index=False, encoding='utf-8-sig')

# 如果需要详细的每个月数据，按时间排序输出
for period, group in df.groupby('year_month'):
    print(f"\n{period} 月的数据：")
    print(group[['create_time', 'like_count', 'video_play_count', 'video_danmaku', 'video_comment']])
