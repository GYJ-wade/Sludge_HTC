import matplotlib.pyplot as plt


def violin_box(data_violin, name_violin, unit, num):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax = plt.gca()
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
    plt.ylabel(ylabel=name_violin + unit, labelpad=10, fontsize=30)
    plt.text(0.03, 0.9, '({})'.format(num), horizontalalignment="left", fontsize='30',
             transform=ax.transAxes)


def reader(data_read, name_read):
    import pandas as pd
    file = pd.read_csv(data_read, encoding="utf-8", error_bad_lines=False)
    data_read = pd.DataFrame(file, columns=name_read)
    return data_read


data = '../0. Data file/Data-sludge.csv'
name = ['type', 'TS', 'OM', 'Ash', 'HHV', 'C', 'O/C', 'H/C', 'N/C', 'Type']
name_unit = [' (%)', ' (%)', ' (%)', ' (MJ/kg)', ' (%)', '', '', '']
type_sludge = ['PS', 'AS', 'SS', 'ADS']
color = ['b', 'r', 'g', 'y']
num_f = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
x = reader(data, name)

plt.figure(figsize=(32, 12), dpi=300)
for i in range(8):
    plt.subplot(2, 4, i + 1)
    violin_box(x, name[i + 1], name_unit[i], num_f[i])
plt.tight_layout(pad=1.2)
plt.savefig('../5. Fig/{name}.jpg'.format(name='violin_plot'), dpi=300)
plt.show()
