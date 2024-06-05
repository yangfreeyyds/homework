
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import numpy as np
import statsmodels

plt.style.use('classic')  # 设置样式表风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import warnings

# 忽略运行时警告信息
warnings.filterwarnings('ignore')
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error  # MAE

# In[2]:

data = pd.read_csv('billbill.csv')
print(data.head())

# In[3]:

y1 = data['factor_score'].astype(float).squeeze()
ts = pd.Series(y1)

# # 划分训练集、测试集、预测集

# In[40]:


# 前80%作为训练集，后20%作为测试集，未来20个作为预测集
train = ts[:int(len(ts) * 0.8)]
test = ts[int(len(ts) * 0.8):]
step = 20  # 往后预测20个
print("train:", len(train))
print("test:", len(test))

# # 原始数据序列图

# In[5]:


# 绘制原始数据图表
plt.figure(figsize=(15, 8))
plt.plot(ts)
plt.title('Original Data')
plt.savefig('时间序列预测/Original Data.png', dpi=300)
plt.show()

# # 统计量

# In[6]:


# 计算均值、标准差、最大值和最小值
mean = np.mean(ts)
std = np.std(ts)
max_value = np.max(ts)
min_value = np.min(ts)

# 输出结果
print("均值:", mean)
print("标准差:", std)
print("最大值:", max_value)
print("最小值:", min_value)

# # 白噪声检验

# In[7]:


from statsmodels.stats.diagnostic import acorr_ljungbox


# 白噪声检验
def test_stochastic(ts, lags=24, alpha=0.05):  # 返回统计量和p值  lags为检验的延迟数
    p_value = acorr_ljungbox(ts, lags=lags)  # lags可自定义
    print(p_value)
    if np.max(p_value['lb_pvalue']) < alpha:
        return '该序列不是白噪声序列'
    else:
        return '该序列是白噪声序列'
    return p_value
    # 返回统计量和p值,lb_pvalue>0.05则为白噪声序列


# In[8]:


print(test_stochastic(ts, lags=3))

# # 提取序列的趋势、季节和随机效应（残差）

# In[9]:


# 分解成趋势（trend）季节性（seasonality）和残差（residual）****************************
import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(ts, period=7, model="add")
fig = res.plot()
# 调整图的大小
fig = plt.gcf()
fig.set_size_inches(15, 8)


# # 自相关ACF，偏自相关PACF图像

# In[10]:


def draw_acf_pacf(ts, lags=12):
    f = plt.figure()
    ax1 = f.add_subplot(211)
    plot_acf(ts, ax=ax1, lags=lags, color='royalblue')
    plt.title("自相关性", fontdict={'weight': 'normal', 'size': 15})
    ax2 = f.add_subplot(212)
    plot_pacf(ts, ax=ax2, lags=lags, color='royalblue')
    plt.title("偏自相关性", fontdict={'weight': 'normal', 'size': 15})
    # 调整图的大小
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    # plt.subplots_adjust(hspace=0.5)
    plt.savefig('时间序列预测/自相关 性偏自相关性.png', dpi=300)
    plt.show()


# In[11]:


draw_acf_pacf(ts, lags=32)


# # ADF平稳性检验

# In[12]:


# Dickey-Fuller test（ADF平稳性检验）:
def teststationarity(ts, max_lag=None):
    dftest = statsmodels.tsa.stattools.adfuller(ts, maxlag=max_lag)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    if (dfoutput['Test Statistic'] < dfoutput['Critical Value (1%)']):
        print("ADF平稳性检验:")
        print("****** Test Statistic < Critical Value (1%) ******\n             此序列为平稳序列")
    return dfoutput


# In[13]:


print(teststationarity(ts))  # 平稳性检验

# # ADF单位根检验哪阶差分之后数据平稳

# In[14]:


# 进行差分处理
diff1 = ts.diff().dropna()  # 一阶差分
diff2 = diff1.diff().dropna()  # 二阶差分
diff3 = diff2.diff().dropna()  # 三阶差分

# In[15]:


print(teststationarity(diff1))  # 平稳性检验

# In[16]:


print(teststationarity(diff2))  # 平稳性检验

# # 差分后平稳的数据

# In[18]:


# 绘制差分后平稳的数据图表
plt.figure(figsize=(15, 8))
plt.plot(diff2)
plt.title('Differenced Data')
plt.savefig('时间序列预测/Differenced Data.png', dpi=300)
plt.show()
# 差分阶数d
d = 2

# # 寻找最优ARIMA模型

# In[19]:


# 自动寻找最优 ARIMA 或 SARIMA 模型并训练
p_min, p_max = 0, 3
q_min, q_max = 0, 3
best_aic = np.inf
best_bic = np.inf
best_order = None
for p in range(p_min, p_max + 1):
    for q in range(q_min, q_max + 1):
        try:
            # 创建ARIMA模型对象，并指定参数p、d、q
            model = sm.tsa.arima.ARIMA(ts, order=(p, d, q))  # 二阶差分平稳，所以d最好取2
            result = model.fit()
            if result.aic < best_aic and result.bic < best_bic:
                best_aic = result.aic
                best_bic = result.bic
                best_order = (p, d, q)
        except:
            continue

print("Best AIC: ", best_aic)
print("Best BIC: ", best_bic)
print("Best order: ", best_order)

# # 模型检验

# ## 残差是检验模型是否成功捕捉所有信息的有效指标，若残差是白噪声，则模型比较好

# In[20]:


resid = result.resid  # 残差
plt.figure(figsize=(15, 8))
plt.plot(resid)

# #  残差白噪声检验，若残差序列不是白噪声，则其中还含有部分有用信息

# In[21]:


#  残差白噪声检验，若残差序列不是白噪声，则其中还含有有用信息
print(test_stochastic(resid, 5))

# # 残差正态性检验，白噪声服从正态分布

# In[22]:


from scipy import stats

x = stats.normaltest(resid)  # 检验序列残差是否为正态分布    pvalue<  0.05  拒绝原假设 认为残差符合正态分布

# In[23]:


if x.pvalue < 0.05:
    print('pvalue = {} < 0.05, 故残差符合正态分布'.format(x.pvalue))

# In[24]:


# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2)

# 绘制第一个子图
ax1.hist(resid, bins=50)
ax1.set_title("残差正态分布图")

# 绘制第二个子图
stats.probplot(resid, dist="norm", plot=plt)
ax2.set_title("残差概率图")

# 调整子图之间的间距
plt.tight_layout()

# 调整图形大小
fig.set_size_inches(15, 8)
plt.savefig('时间序列预测/残差概率图和残差正态分布图.png', dpi=300)
# 显示图形
plt.show()

# # DW残差序列自相关检验 ，白噪声之间不存在自相关关系

# In[25]:


from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(result.resid.values)  ##DW检验：靠近2——正常；靠近0——正自相关；靠近4——负自相关

# In[26]:


dw  # dw更加接近2，表明残差独立不存在自相关性

# # 以最优参数建立ARIMA模型

# In[27]:


model = sm.tsa.arima.ARIMA(endog=train, order=best_order)
# 输出模型的摘要信息
results = model.fit()
print(results.summary())

# # 预测

# In[28]:


predtrain = results.predict(start=1, end=len(train))
predtrain = np.array(predtrain)

# In[29]:


# 对测试集数据进行预测验证
inputdata = train
predtest = []
for i in range(0, (len(test))):
    history = inputdata
    model = sm.tsa.arima.ARIMA(history, order=best_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(1)
    predtest.append(float(yhat))
    inputdata = np.append(inputdata, [test.iloc[i]])
predtest = np.array(predtest)

# In[46]:


# 预测未来
inputdata = ts
pred = []
for i in range(0, step):
    history = inputdata
    model = sm.tsa.arima.ARIMA(history, order=best_order)
    model_fit = model.fit()
    yhat = model_fit.forecast(1)
    pred.append(float(yhat))
    inputdata = np.append(inputdata, [yhat])
pred = np.array(pred)

# In[47]:


x = np.concatenate((predtrain, predtest, pred))
y = np.concatenate((predtrain, predtest))
z = predtrain

# In[48]:


# 计算置信区间（使用标准差方法）
forecast_error1 = results.resid
forecast_std1 = np.std(forecast_error1)
z_score = 1.96  # 95% 置信区间对应的Z值

# In[49]:


t = np.concatenate((predtest, pred))
forecast_lower = t - z_score * forecast_std1
forecast_upper = t + z_score * forecast_std1

# In[50]:


plt.figure(figsize=(15, 8))
plt.xlabel('时间', fontsize=15)
plt.plot(ts, label='真实值')
plt.plot(x, label='未来预测值')
plt.plot(y, label='测试集预测值', linestyle='-')
plt.plot(z, label='训练集预测值', linestyle='-')
plt.fill_between(np.arange(len(z), len(x)), forecast_lower, forecast_upper, color='gray', alpha=0.3,
                 label='95%置信区间')
plt.legend(fontsize=15)
plt.savefig('时间序列预测/预测结果.png', dpi=300)
plt.show()
# In[51]:


# 将数组转换为 DataFrame
df = pd.DataFrame(pred)
df.columns = ['未来预测值']
# 将 DataFrame 写入 CSV 文件
file_name = '时间序列预测/output.csv'
df.to_csv(file_name, index=False)


# # 模型评估

# In[52]:


def metrics_sklearn(y_valid, y_pred_, model='model'):
    r2 = r2_score(y_valid, y_pred_)  # R^2评价
    print(model + '_R2：{}'.format(r2))

    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_))  # rmse评价
    print(model + '_rmse：{}'.format(rmse))

    mae = mean_absolute_error(y_valid, y_pred_)  # mae评价
    print(model + '_mae：{}'.format(mae))

    mse = mean_squared_error(y_valid, y_pred_)  # mae评价
    print(model + '_mse：{}'.format(mse))

    assess = pd.DataFrame([[r2, rmse, mae, mse]], columns=['R2', 'rmse', 'mae', 'mse'], index=[model])
    return assess


# In[53]:


w = metrics_sklearn(test, predtest)

# In[ ]:




