def groupby_find(data, name):
    data_1 = data.groupby(name).agg(['mean'])
    print(data_1)


def reader(data, name):
    import pandas as pd
    data_1 = pd.DataFrame(data, columns=name)
    return data_1


def analysis_parameter(data, key_name, eva_name):
    for i in range(len(key_name)):
        print('\n')
        name = [key_name[i]]+eva_name
        data_1 = reader(data, name)
        groupby_find(data_1, key_name[i])
