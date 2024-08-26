import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.tree import DecisionTreeClassifier

class BinMonotonicityChecker:
    def __init__(self, df, target, max_iterations=10, min_merge=1):
        self.df = df
        self.target = target
        self.max_iterations = max_iterations
        self.min_merge = min_merge
        self.results = {}

    def check_monotonicity(self, df_grouped):
        """
        检查分箱后的数据是否单调。
        """
        is_monotonic = df_grouped['badprob'].is_monotonic_increasing or df_grouped['badprob'].is_monotonic_decreasing
        return is_monotonic

    def merge_bins(self, variable):
        """
        合并不满足单调性的箱体，使用启发式方法优先合并最不单调的部分。
        """
        iterations = 0
        while not self.check_monotonicity(self.binned_data) and iterations < self.max_iterations:
            iterations += 1
            merged_count = 0
            merged = False

            # 计算差值并找到最不单调的箱体
            differences = np.diff(self.binned_data['badprob'].values)
            non_monotonic_indices = np.where(differences < 0)[0]

            if len(non_monotonic_indices) > 0:
                index_to_merge = non_monotonic_indices[0]  # 选择最靠前的非单调部分
                self.binned_data.loc[index_to_merge, variable] = self.binned_data.loc[index_to_merge + 1, variable]
                merged_count += 1
                merged = True

            if merged and merged_count >= self.min_merge:
                self.binned_data = self.binned_data.groupby(variable).agg({
                    'count': 'sum',
                    'good': 'sum',
                    'bad': 'sum',
                    'count_distr': 'sum',
                    'woe': 'mean',
                    'bin_iv': 'sum',
                    'total_iv': 'max',
                    'badprob': 'mean'
                }).reset_index()
            else:
                break

    def adjust_bins(self):
        """
        自动调整分箱直到单调。
        """
        variables = self.df['variable'].unique()
        for variable in variables:
            self.binned_data = self.df[self.df['variable'] == variable].copy()
            self.binned_data = self.binned_data.groupby('bin').agg({
                'count': 'sum',
                'good': 'sum',
                'bad': 'sum',
                'count_distr': 'sum',
                'woe': 'mean',
                'bin_iv': 'sum',
                'total_iv': 'max',
                'badprob': 'mean'
            }).reset_index()

            self.merge_bins(variable)

            if not self.check_monotonicity(self.binned_data):
                self.redefine_bins(variable)

            # 将处理结果存储在字典中
            self.results[variable] = self.binned_data

    def redefine_bins(self, variable):
        """
        使用决策树重新定义分割点。
        """
        clf = DecisionTreeClassifier(max_leaf_nodes=5)
        clf.fit(self.df[[variable]], self.df[self.target])
        self.df['bins'] = clf.apply(self.df[[variable]])

        self.binned_data = self.df.groupby('bins').agg({
            variable: 'mean',
            'count': 'sum',
            'good': 'sum',
            'bad': 'sum',
            'count_distr': 'sum',
            'woe': 'mean',
            'bin_iv': 'sum',
            'total_iv': 'max',
            'badprob': 'mean'
        }).reset_index()

    def plot_histogram(self, variable, col_name, label_bad='响应样本',
                       label_good='未响应样本', label='响应率%', rotation=45):
        """
        绘制双轴直方图和响应率曲线。
        """
        df = self.results[variable]
        x_list = df['bin']  # X轴
        y_value = df['bad']  # Y1轴数值
        y11_value = df['good']  # Y1轴数值
        y12_value = df['bad']  # Y1轴数值

        y1 = y_value.max() * (1 + 0.30)  # Y1轴取值上限
        y2_value = df['badprob']  # Y2轴数值
        y2 = y2_value.max() * (1 + 0.20)  # Y2轴取值上限
        iv = df['total_iv'].values[0].round(3)

        def to_percent(temp, position):
            return '%2.1f' % (100 * temp) + "%"

        plt.figure(figsize=(12, 8), constrained_layout=True)
        sns.set(style="whitegrid")
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))  # 设置纵轴百分比
        ax = plt.subplot(111, facecolor='lightgrey')
        plt.ylim(0, y1)

        # 柱状图并列排布
        x = np.arange(len(x_list))
        rects1 = ax.bar(x - 0.1, y11_value, alpha=0.8, width=0.3, color='skyblue', label=label_good)
        rects2 = ax.bar(x + 0.1, y12_value, alpha=0.8, width=0.3, color='steelblue', label=label_bad)

        # 添加柱状图的数值标注
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(x_list, rotation=rotation, fontsize=12)
        ax.legend(loc='upper left', fontsize=12)

        # 第二个Y轴
        ax2 = ax.twinx()
        plt.ylim(0, y2)
        ax2.plot(x_list, y2_value, color='darkblue', marker='o', linestyle='-', linewidth=2, label=label)

        for x, y in zip(x_list, y2_value):
            ax2.text(x, y, f'{y * 100:.2f}%', ha='center', va='bottom', fontsize=10, color='darkblue')

        ax2.legend(loc='upper right', fontsize=12)
        ax2.set_title(f'{col_name}: IV = {iv}', fontsize=16)
        ax2.set_ylabel('响应率 %', fontsize=12)
        plt.show()



if __name__ == '__main__':
    df = pd.DataFrame({
        'variable': ['A', 'A', 'A', 'A', 'A'],
        'bin': [1, 2, 3, 4, 5],
        'count': [100, 200, 300, 400, 500],
        'good': [30, 60, 90, 120, 150],
        'bad': [70, 140, 210, 280, 350],
        'badprob': [0.3, 0.3, 0.3, 0.3, 0.3],
        'total_iv': [0.1, 0.1, 0.1, 0.1, 0.1],
        'count_distr': [0.1, 0.2, 0.3, 0.4, 0.5],
        'woe': [0.1, 0.2, 0.3, 0.4, 0.5],
        'bin_iv': [0.01, 0.02, 0.03, 0.04, 0.05]
    })

    checker = BinMonotonicityChecker(df, 'badprob', max_iterations=10, min_merge=2)
    checker.adjust_bins()
    checker.plot_histogram('A', 'Example Variable')