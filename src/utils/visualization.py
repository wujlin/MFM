import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 设置可视化参数
SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

# 设置图形风格
plt.style.use('seaborn-white')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': SMALL_SIZE,
    'axes.titlesize': MEDIUM_SIZE,
    'axes.labelsize': MEDIUM_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'legend.fontsize': SMALL_SIZE,
    'mathtext.fontset': 'stix'
})

# 计算相关系数
correlations = []
p_values = []

# 定义变量映射
sim_emp_pairs = [
    ('o_people', 'H', 'sentiment_high_p', 'High Arousal'),
    ('o_people', 'M', 'sentiment_middle_p', 'Middle Arousal'),
    ('o_people', 'L', 'sentiment_low_p', 'Low Arousal'),
    ('m_media', 'R', 'm_risk_p', 'Mainstream Media Risk'),
    ('w_media', 'R', 'w_risk_p', 'We Media Risk')
]

# 提取经验数据的周数
weeks = sorted(list(empirical_data['m_risk_p'].keys()))

# 计算每对变量的相关系数
for category, state, emp_key, label in sim_emp_pairs:
    # 提取每周对应的模拟数据（使用每周的最后一天）
    sim_data = [history[week*7][category][state] for week in weeks]
    
    # 提取经验数据
    emp_data = [empirical_data[emp_key][week] for week in weeks]
    
    # 计算相关系数
    corr, p_val = stats.pearsonr(sim_data, emp_data)
    correlations.append(corr)
    p_values.append(p_val)

# 创建图形
fig, ax = plt.subplots(figsize=(8.27, 5), dpi=300)

# 设置柱状图位置
x = np.arange(len(sim_emp_pairs))
width = 0.6

# 定义颜色
colors = ['#DF5557', '#EFCD61', '#9EC9E7', '#479692', '#F08D2A']

# 绘制柱状图
bars = ax.bar(x, correlations, width, color=colors)

# 添加显著性标记
for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
    marker_y_offset = 0.05 if corr >= 0 else -0.1  # 根据相关系数正负调整标记位置
    if p_val < 0.001:
        marker = '***'
    elif p_val < 0.01:
        marker = '**'
    elif p_val < 0.05:
        marker = '*'
    else:
        marker = 'ns'
    
    ax.text(i, corr + np.sign(corr) * marker_y_offset, marker, 
            ha='center', va='bottom' if corr >= 0 else 'top', 
            fontsize=SMALL_SIZE, 
            fontfamily='Times New Roman')

# 设置坐标轴
ax.set_ylabel('Correlation Coefficient')
ax.set_xticks(x)
ax.set_xticklabels([label for _, _, _, label in sim_emp_pairs], 
                   rotation=45, ha='right')

# 添加水平线
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 设置y轴范围，留出显示显著性标记的空间
ax.set_ylim(-1.1, 1.1)

# 设置y轴刻度
ax.set_yticks(np.arange(-1.0, 1.1, 0.25))

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加网格线
ax.grid(True, axis='y', alpha=0.3, linestyle=':')

# 调整布局
plt.tight_layout()

plt.show()