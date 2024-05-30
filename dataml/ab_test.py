

def abtest1():
    import numpy as np

    # 模型营销组
    high_response_group_size = 50000
    marketing_group_size = 40000
    model_marketing_group = np.random.choice(range(high_response_group_size), size=marketing_group_size, replace=False)

    # 对照组1
    control_group_size = 10000
    control_group1 = np.random.choice(model_marketing_group, size=control_group_size, replace=False)

    # 对照组2
    total_customers = 100000  # 假设总体客户数量
    control_group2_size = 5000
    control_group2 = np.random.choice(range(total_customers), size=control_group2_size, replace=False)

    # 打印结果
    print("模型营销组：", model_marketing_group)
    print("对照组1：", control_group1)
    print("对照组2：", control_group2)

def abtest():
    import statsmodels.stats.api as sms

    # 设定期望的效果大小（相对提升比例）
    effect_size = 0.02  # 根据实际情况设定

    # 设定显著性水平和统计功效
    alpha = 0.05
    power = 0.8

    # 计算高响应组样本量
    high_response_sample_size = sms.NormalIndPower().solve_power(effect_size, alpha=alpha, power=power, ratio=1,
                                                                 alternative='larger')

    # 对照组1样本量与高响应组相同
    control_group1_sample_size = high_response_sample_size

    # 对照组2样本量（整体客户群体的样本量）
    total_population_sample_size = sms.NormalIndPower().solve_power(effect_size, alpha=alpha, power=power, ratio=1,
                                                                    alternative='two-sided')

    # 输出样本量
    print(f"高响应组样本量: {high_response_sample_size:.0f}")
    print(f"对照组1样本量: {control_group1_sample_size:.0f}")
    print(f"对照组2样本量: {total_population_sample_size:.0f}")

def abtest3():
    import pandas as pd
    import numpy as np

    # 假设你有一个名为 df 的 DataFrame，包含了客户信息

    # 模型营销组
    high_response_group_size = 50000
    marketing_group_size = 40000
    model_marketing_group = np.random.choice(df.index, size=marketing_group_size, replace=False)

    # 对照组1
    control_group_size = 10000
    control_group1 = np.random.choice(model_marketing_group, size=control_group_size, replace=False)

    # 对照组1取反操作
    control_group1_neg = ~df.index.isin(control_group1)

    # 对照组2
    total_customers = len(df)
    control_group2_size = 5000
    control_group2 = np.random.choice(df.index, size=control_group2_size, replace=False)

    # 通过 DataFrame.loc 获取相应的行
    model_marketing_group_df = df.loc[model_marketing_group]
    control_group1_df = df.loc[control_group1_neg]
    control_group2_df = df.loc[control_group2]

    # 打印结果
    print("模型营销组：", model_marketing_group_df)
    print("对照组1：", control_group1_df)
    print("对照组2：", control_group2_df)



if __name__ == '__main__':
    abtest1()