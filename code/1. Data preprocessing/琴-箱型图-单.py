import matplotlib.pyplot as plt


def violin_box(data, name, unit, num):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 设置绘图风格
    ax = plt.gca()
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号的处理
    # plt.figure(figsize=(8, 5), dpi=300)
    # plt.grid(visible=True, axis='y', color='gray', linewidth=1)
    sns.violinplot(x='type',
                   y=name,  # 指定y轴的数据
                   data=data,  # 指定绘图的数据集
                   scale="width",
                   palette='pastel',
                   linewidth=2,
                   )
    plt.xticks(fontsize=25)  # 设置x轴刻度大小
    plt.yticks(fontsize=25)  # 设置y轴刻度大小
    plt.xlabel(xlabel='Type of sludge', labelpad=10, fontsize=30)
    plt.ylabel(ylabel=name + ' (' + unit + ')', labelpad=10, fontsize=30)
    plt.text(0.03, 0.9, '({})'.format(num), horizontalalignment="left", fontsize='30',
             transform=ax.transAxes)  # bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.9),


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
data = 'C:\\Users\\MI\\Desktop\\BT邀稿\\Fig-1-data-1.csv'
# name = ['type', 'C(Ash)', 'H(Ash)', 'O(Ash)', 'N(Ash)']
name = ['type', 'C', 'H', 'O', 'N']
name_unit = ['%', '%', '%', '%']
type_sludge = ['PS', 'AS', 'SS', 'ADS']
type = ['type']
color = ['b', 'r', 'g', 'y']
num_f = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
x = reader(data, name)

plt.figure(figsize=(16, 12), dpi=300)
for i in range(4):
    plt.subplot(2, 2, i + 1)  # facecolor="whitesmoke"
    violin_box(x, name[i + 1], name_unit[i], num_f[i])
plt.tight_layout(pad=1.15)
plt.savefig('C:\\Users\\MI\\Desktop\\BT邀稿\\Fig\\{变量}-Ash-free.jpg'.format(变量='4合1琴图'), dpi=300)
plt.show()
