import matplotlib.pyplot as plt


def violin_box(data_violin, name_violin, unit, name_y):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    sns.violinplot(x='type',
                   y=name_violin,
                   data=data_violin,
                   scale="width",
                   palette='pastel',
                   linewidth=2,
                   )
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel(xlabel='Type of sludge', labelpad=10, fontsize=30)
    plt.ylabel(ylabel=name_y + unit, labelpad=10, fontsize=30)



def reader(data_read, name_read):
    import pandas as pd
    file = pd.read_csv(data_read, encoding="utf-8", error_bad_lines=False)
    data_read = pd.DataFrame(file, columns=name_read)
    return data_read


data = '../0. Data file/Data-sludge.csv'
name = ['C(ash-free)', 'type']
name_sludge = ['C']
name_unit = ['(%)']
type_sludge = ['PS', 'AS', 'SS', 'ADS']
color = ['b', 'r', 'g', 'y']
num_f = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
x = reader(data, name)

plt.figure(figsize=(8, 6), dpi=300)
violin_box(x, name[0], name_unit[0], name_sludge[0])
plt.tight_layout(pad=1.2)
plt.savefig('../5. Fig/{name}.jpg'.format(name='violin_plot_ash_free'), dpi=300)
plt.show()
