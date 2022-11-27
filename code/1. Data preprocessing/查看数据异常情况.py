import numpy as np
import pandas as pd
import boxplot_data_function as bdf

# 读取CSV文件中的数据
data_noValues = pd.read_csv("C:\\Users\\MI\\Desktop\\污泥水热炭数据汇总-直删1.csv")

# 手动给出数据标签，后期考虑继续优化为直接读取CSV文件中的标签
a = np.array(['HHV', 'CR', 'ER'])


# 主函数
def main(range):
    print()
    print('以下为', range, '倍范围')
    print()
    bug_data_loc_total = np.array([])  # 设置一个装载异常数据行号的数组
    for i in a:
        data_values = bdf.value(data_noValues, i)  # 将数据按某一列的大小进行排序
        print(i, ':')
        print('下四分位数是：', bdf.four_down(data_values[i]),
              '，中位数是：', bdf.four_mid(data_values[i]),
              '，上四分位数是：', bdf.four_up(data_values[i]))
        print('正常值下限是：', round(bdf.down_line(data_values[i], range), 2),
              '，正常值上限是：', round(bdf.up_line(data_values[i], range), 2))
        bug_data_loc_total = np.append(bug_data_loc_total,
                                       bdf.find_bug_data(data_values[i],
                                                        bdf.up_line(data_values[i], range),
                                                        bdf.down_line(data_values[i], range)))
        print()
    order_loc = np.unique(bug_data_loc_total)  # 删除重复行号，并排列
    print('异常数据行共有', order_loc.size, '，位置如下：')
    print(order_loc)
    print()
    data_new = data_noValues.drop(data_noValues.index[order_loc.astype('int64')], axis=0)
    print(data_new)
    frame = pd.DataFrame({'HHV': data_new['HHV'],
                          'CR': data_new['CR'],
                          'ER': data_new['ER'],
                          })
    # frame.to_csv("C:\\Users\\MI\\Desktop\\BOD5预测\\第一份数据整理.csv", index=False, sep=',')


# 分别运行1.5倍范围和3倍范围的主函数
main(1.5)
# main(3)


