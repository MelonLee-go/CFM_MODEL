import pandas as pd

# 读取数据
order_data = pd.read_csv('OnlineRetail.csv', encoding='gbk')
# 数据清洗
# 1、处理用户ID和消费时间出现的空值（提出）
order_data.dropna(subset=['CustomerID', 'InvoiceDate'], inplace=True)
# 2、处理购买数量和商品单价出现的空值（填充）
order_data.fillna(value={'Quantity': order_data['Quantity'].mean(),
                         'UnitPrice': order_data['UnitPrice'].mean()},
                  inplace=True)
# 3、处理购买数量小于等于0
queryQuantity = order_data.loc[:, 'Quantity'] > 0
order_data = order_data.loc[queryQuantity, :]
# 4、处理商品单价小于0.01元
queryUnitPrice = order_data.loc[:, 'UnitPrice'] >= 0.01
order_data = order_data.loc[queryUnitPrice, :]

# 转换数据类型
order_data['CustomerID'] = order_data['CustomerID'].astype('int')
order_data['Quantity'] = order_data['Quantity'].astype('int')
order_data['InvoiceDate'] = pd.to_datetime(order_data['InvoiceDate'],
                                           format='%d-%m-%Y %H:%M')
order_data['UnitPrice'] = order_data['UnitPrice'].astype('double')
# 计算R(Recency)：最后一次交易距离今天的间隔
max_dt = order_data['InvoiceDate'].max()
order_data['R'] = (max_dt - order_data['InvoiceDate']).dt.days
R = order_data.groupby(by=['CustomerID'])['R'].agg([('R', 'min')])
# 计算F(Frequency)：用户消费总次数
order_data['F'] = order_data['Quantity']
F = order_data.groupby(by=['CustomerID'])['CustomerID'].agg([('F', 'count')])
# 计算M(Monetary)：用户消费总金额
order_data['M'] = order_data['Quantity'] * order_data['UnitPrice']
M = order_data.groupby(by=['CustomerID'])['M'].agg([('M', 'sum')])
# 合并RFM为一张表，并使用固定函数计算用户层级
RFM_Table = R.join(F).join(M)


# RFM计算函数
def RFM_cal(x):
    # 存储的是三个字符串形式的0或者1
    level = x.map(lambda temp: '1' if temp >= 0 else '0')
    label = level.R + level.F + level.M
    dic = {
        '111': '重要价值客户',
        '011': '重要保持客户',
        '101': '重要挽留客户',
        '001': '重要发展客户',
        '110': '一般价值客户',
        '010': '一般保持客户',
        '100': '一般挽留客户',
        '000': '一般发展客户'
    }
    ans = dic[label]
    return ans


RFM_Table['label'] = RFM_Table.apply(lambda x: x - x.mean()).apply(RFM_cal, axis=1)
# 输出
print(RFM_Table)
print(RFM_Table.groupby(by=['label'])['label'].agg([('label', 'count')]))
RFM_Table.to_csv("result.csv", encoding='gbk')
