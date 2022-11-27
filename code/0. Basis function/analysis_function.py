# 按照分类名name，查找数据DataFrame类型的data中，各项数据的平均值
def groupby_find(data, name):
    data_1 = data.groupby(name).agg(['mean'])
    print(data_1)


# 读取数据，将地址为data_file的CSV文件数据转化为DataFrame数据结构，并读取其中列为name的数据
def reader_file(data_file, name):
    import pandas as pd
    file = pd.read_csv(data_file, encoding="utf-8", error_bad_lines=False)
    data_1 = pd.DataFrame(file, columns=name)
    return data_1


# 读取数据，将文件数据转化为DataFrame数据结构，并读取其中列为name的数据，与上一个相比少了读取CSV文件一步
def reader(data, name):
    import pandas as pd
    data_1 = pd.DataFrame(data, columns=name)
    return data_1


# 在DataFrame类型数据data中，选择指定行列数据。实际上这里的 type 可以替换为参数放在函数形参中。
def select(data, name, label, x):
    data_2 = data[name].loc[(x['type'] == label)]
    return data_2


# 数据分析，读取文件数据版，data为数据文件地址
def analysis_parameter_file(data, key_name, eva_name):
    for i in range(len(key_name)):
        print('\n')
        name = [key_name[i]]+[eva_name[1]]  # 此处的eva_name包含rmse和r2，但在查看数据时r2更能体现模型的预测能力，因此只取了r2
        data_1 = reader_file(data, name)  # 从数据文件地址读取列名为name的数据
        groupby_find(data_1, key_name[i])  # 分类计算


# 数据分析，直接读取数据版，data为DataFrame数据。后续超参数数据分析中，这个函数用的较多
def analysis_parameter(data, key_name, eva_name):
    for i in range(len(key_name)):
        print('\n')
        name = [key_name[i]]+eva_name  # 此处的eva_name可以包含rmse和r2
        data_1 = reader(data, name)  # 从数据中读取列名为name的数据
        groupby_find(data_1, key_name[i])  # 分类计算
