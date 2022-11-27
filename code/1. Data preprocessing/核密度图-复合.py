# 制作子图底板
def multi_plot(file):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(24, 12), dpi=300)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        KdePlot(x, name[i + 1], name_unit[i], type_sludge, color)
    plt.tight_layout(pad=1.08)
    plt.savefig('C:\\Users\\MI\\Desktop\\BT邀稿\\Fig\\核密度图\\{变量}.jpg'.format(变量=file), dpi=300)  # 存储图片
    plt.show()


# 绘制核密度曲线图
def KdePlot(data, name, unit, line_name, color):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 绘制核密度分布直方图
    # plt.figure(figsize=(8, 6), dpi=300)
    # sns.set(style='white', font='SimHei')        # 设置边框颜色、设置中文字体
    for i in range(len(line_name)):
        sns.kdeplot(data=select(data, name, line_name[i]),  # 指定绘图数据
                    color=color[i],  # 设置绘图颜色
                    shade=True)  # 绘制密度曲线

    # plt.title(name)                       # 设置图片标题
    plt.legend(line_name, fontsize=20)  # 设置标签
    plt.xticks(fontsize=20)  # 设置x轴刻度大小
    plt.yticks(fontsize=20)  # 设置y轴刻度大小
    plt.xlabel(xlabel=name + ' (' + unit + ')', labelpad=10, fontsize=25)  # 设置 x 轴标签
    plt.ylabel('Date Density', labelpad=10, fontsize=25)  # 设置 y 轴标签
    plt.grid(visible=True, color='gray', linewidth=1)  # 设置参考线
    plt.subplots_adjust(left=0.2, bottom=0.15, top=0.9, right=0.9)


# 读取数据
def reader(data, name):
    import pandas as pd
    file = pd.read_csv(data, encoding="utf-8", error_bad_lines=False)
    data = pd.DataFrame(file, columns=name)
    return data


# 选择指定行列数据
def select(data, name, label):
    data = data[name].loc[(x['type'] == label)]
    return data


# 删除特定标签的数据
def delete(data, name, label):
    data = data.loc[data[name] != label]
    return data


# 代码执行部分
data = 'C:\\Users\\MI\\Desktop\\BT邀稿\\Fig-1-data.csv'
name = ['type', 'TS', 'VM', 'Ash', 'O/C', 'H/C', 'HHV']
name_unit = ['%', '%', '%', 'ratio', 'ratio', 'MJ/kg']
file_name = ['TS', 'VM', 'Ash', 'O比C', 'H比C', 'HHV']
type_sludge = ['PS', 'AS', 'SS', 'ADS']
type = ['type']
color = ['b', 'r', 'g', 'y']
x = reader(data, name)

multi_plot('6合1')
