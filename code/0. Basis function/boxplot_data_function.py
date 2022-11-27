import math
import numpy as np


# 计算四分位数、上限、下限的统计函数，分别定义。以下的上下指大小，上为大，下为小。
# 下四分位数计算
def four_down(data):
    data = np.array(data)  # 数据data转化为数组
    loc = math.ceil(len(data) * 0.25 + 1)
    return data[loc]


# 中位数计算
def four_mid(data):
    return np.median(data)


# 上四分位数计算
def four_up(data):
    data = np.array(data)  # 数据data转化为数组
    loc = math.ceil(len(data) * 0.75 + 1)
    return data[loc]


# 正常值上限计算
def up_line(data, m):
    iqr = four_up(data) - four_down(data)  # 上下四分位间距
    max_line = np.max(data)
    return min(four_up(data) + m * iqr, max_line)


# 正常值下限计算
def down_line(data, m):
    iqr = four_up(data) - four_down(data)  # 上下四分位间距
    min_line = np.min(data)
    return max(four_down(data) - m * iqr, min_line)


# 按数据某一列的数据大小，进行全体数据的重排列
def value(data, i):
    return data.sort_values(by=i, ascending=True)


# 寻找在上下限之外的异常数据
def find_bug_data(data, up, down):
    num_bug = 0  # 初始化num_bug
    bug_data = np.array([])
    bug_data_loc = np.array([])
    for i in range(len(data)):
        if data[i] > up or data[i] < down:  # 如果data在上下限之外，则标定为异常数据
            num_bug += 1  # 异常数据个数+1
            a = "%.2f" % (data[i])  # 将异常数据转化为2位小数的形式
            bug_data = np.append(bug_data, a)  # 将异常数据添加到bug_data中
            bug_data_loc = np.append(bug_data_loc, i)  # 将异常数据的行数位置添加到bug_data_loc中
    print('共计', len(data), '个数据，其中有', num_bug, '个异常数据，具体如下')  # 打印数据总量、异常数据的数量
    print(bug_data)  # 打印异常数据的具体值
    return bug_data_loc  # 函数返回值为异常数据的位置
