import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('billbill.csv')

# 选择需要进行因子分析的列
columns_to_evaluate = ['like_count', 'video_play_count', 'video_danmaku', 'video_comment']

# 数据标准化处理
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[columns_to_evaluate])

# 因子分析
factor_analysis = FactorAnalysis(n_components=1, random_state=42)
factor_scores = factor_analysis.fit_transform(scaled_data)

# 将因子得分转换为DataFrame
factor_scores_df = pd.DataFrame(factor_scores, columns=['factor_score'])

# 将因子得分添加回原始数据
data['factor_score'] = factor_scores_df['factor_score']

# 按时间排序
data = data.sort_values(by='year_month')

# 打印结果
print(data[['year_month', 'factor_score']])

# 保存结果到新的CSV文件
data.to_csv('因子分析综合评价结果/billbill.csv', index=False, encoding='utf-8-sig')

# 可视化因子得分
plt.figure(figsize=(12, 6))
plt.plot(data['year_month'], data['factor_score'], marker='o')
plt.xticks(rotation=90)
plt.xlabel('Year-Month')
plt.ylabel('Factor Score')
plt.title('Factor Score Over Time')
plt.tight_layout()
plt.savefig('因子分析综合评价结果/factor_score_over_time.png', dpi=300)
plt.show()
