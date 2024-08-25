## 混合采样
**目标:** 让少数类样本占多数类样本的比例达到一个指定的目标。这个目标是通过**过采样**（增加少数类样本）和**欠采样**（减少多数类样本）共同实现的。

假设：
- $ n_{\text{pos}} $是少数类样本（目标类）的原始数量。
- $ n_{\text{neg}} $是多数类样本的原始数量。
- $ r_{\text{target}} $是你希望最终达到的少数类样本与多数类样本的比例（即 `desired_ratio`）。
- $ r_{\text{over}} $是过采样的比例。
- $ r_{\text{under}} $是欠采样的比例。

### 目标
我们希望最终的少数类样本和多数类样本的数量满足：
$$
\frac{n_{\text{pos}} \times r_{\text{over}}}{n_{\text{neg}} \times r_{\text{under}}} = r_{\text{target}}
$$

### 过采样和欠采样的计算

1. **过采样比例** $ r_{\text{over}} $：
    $$
    r_{\text{over}} = \frac{r_{\text{target}} \times n_{\text{neg}} \times r_{\text{under}}}{n_{\text{pos}}}
    $$

2. **欠采样比例** $ r_{\text{under}} $：
    $$
    r_{\text{under}} = \frac{n_{\text{pos}} \times r_{\text{over}}}{r_{\text{target}} \times n_{\text{neg}}}
    $$

但通常情况下不会同时计算两者，而是先确定一个比例，然后根据这个比例计算另一个。

### 具体步骤：
- 先确定一个欠采样比例 $ r_{\text{under}} $，然后根据目标比例 $ r_{\text{target}} $计算所需的过采样比例 $ r_{\text{over}} $。
- 或者，先确定过采样比例，然后计算所需的欠采样比例。

### 示例代码

```python
def calculate_sampling_ratios(n_pos, n_neg, desired_ratio):
    """
    计算混合采样的过采样和欠采样比例，以达到指定的少数类/多数类比例。
    
    :param n_pos: 原始少数类样本数量
    :param n_neg: 原始多数类样本数量
    :param desired_ratio: 希望达到的少数类/多数类比例
    :return: 过采样比例 (r_over), 欠采样比例 (r_under)
    """
    # 先指定一个欠采样比例 r_under
    r_under = 0.5  # 例如，想将多数类样本减少到一半

    # 根据目标比例和欠采样比例计算过采样比例
    r_over = (desired_ratio * n_neg * r_under) / n_pos

    # 如果过采样比例大于1，则说明欠采样比例过大，可以适当调整欠采样比例
    if r_over > 1:
        r_over = 1
        r_under = (n_pos * r_over) / (desired_ratio * n_neg)

    return r_over, r_under

# 示例使用
n_pos = 100  # 假设少数类样本数量
n_neg = 1000  # 假设多数类样本数量
desired_ratio = 0.5  # 希望达到的少数类/多数类比例

r_over, r_under = calculate_sampling_ratios(n_pos, n_neg, desired_ratio)
print(f"过采样比例: {r_over}, 欠采样比例: {r_under}")
```

### 输出示例
假设少数类样本有100个，多数类样本有1000个，你希望达到的目标比例是0.5（即少数类占多数类的一半），那么代码会计算出适当的过采样比例和欠采样比例。

### 注意点
1. **边界条件**：如果计算出的过采样比例大于1，意味着想生成比多数类还多的少数类样本，这是不合理的。此时需要调整欠采样比例。

2. **平衡取舍**：如果对欠采样比例有很强的偏好（例如不想丢失太多数据），可以先固定欠采样比例，再计算过采样比例。反之亦然。


## 按指定过采样或者降采样样本浓度采样
需要计算适合的过采样和欠采样比例，以便直接传递给现有的函数来执行混合采样。这两个函数 `oversample_dataframe` 和 `under_sample` 中的 `desired_ratio` 参数应该分别表示在过采样或欠采样后少数类样本相对于多数类样本的比例。

### 目标
1. 通过 **过采样** 增加少数类样本，达到一个指定的比例 $ r_{\text{over}} $。
2. 通过 **欠采样** 减少多数类样本，达到一个指定的比例 $ r_{\text{under}} $。

最终要实现的是指定的目标比例 $ r_{\text{target}} $：

$$
\frac{n_{\text{pos}} \times r_{\text{over}}}{n_{\text{neg}} \times r_{\text{under}}} = r_{\text{target}}
$$

### 计算思路
1. 先确定一个欠采样比例 $ r_{\text{under}} $。
2. 根据这个比例和目标比例 $ r_{\text{target}} $，计算过采样比例 $ r_{\text{over}} $。

### 代码实现

```python
def calculate_sampling_ratios(n_pos, n_neg, r_target):
    """
    计算过采样和欠采样的desired_ratio，以实现指定的少数类/多数类目标比例。
    
    :param n_pos: 原始少数类样本数量
    :param n_neg: 原始多数类样本数量
    :param r_target: 希望达到的少数类/多数类目标比例
    :return: 欠采样的desired_ratio (r_under), 过采样的desired_ratio (r_over)
    """
    # 1. 先指定一个欠采样比例
    r_under = 0.5  # 你可以调整这个值，决定欠采样的强度
    
    # 2. 计算过采样比例
    r_over = (r_target * n_neg * r_under) / n_pos

    # 3. 如果计算的过采样比例大于1，则调整欠采样比例
    if r_over > 1:
        r_over = 1
        r_under = (n_pos * r_over) / (r_target * n_neg)

    return r_under, r_over

def mixed_sampling_with_categorical(df, r_target, y_target='target'):
    """
    混合采样，按指定目标比例进行过采样和欠采样。

    :param df: 数据集
    :param r_target: 希望达到的少数类/多数类目标比例
    :param y_target: 目标变量列名
    :return: 经过混合采样后的DataFrame
    """
    # 获取少数类和多数类样本数量
    n_pos = sum(df[y_target] == 1)
    n_neg = sum(df[y_target] == 0)

    # 计算过采样和欠采样的比例
    r_under, r_over = calculate_sampling_ratios(n_pos, n_neg, r_target)

    # 进行欠采样
    df_under = under_sample(df, y_target=y_target, desired_ratio=r_under)

    # 进行过采样
    df_mixed = oversample_dataframe(df_under, y_target=y_target, positive_class_ratio=r_over)

    return df_mixed

# 示例使用
n_pos = 100  # 假设少数类样本数量
n_neg = 1000  # 假设多数类样本数量
desired_ratio = 0.5  # 希望达到的最终少数类/多数类比例

# 调用混合采样
df_mixed = mixed_sampling_with_categorical(df, r_target=desired_ratio, y_target='target')
```

### 关键点
- `r_under` 控制欠采样的强度，影响多数类样本减少的量。
- `r_over` 控制过采样的强度，影响少数类样本增加的量。
