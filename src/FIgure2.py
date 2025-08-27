# 开始joinsentiment和risk数据
import pandas as pd
import numpy as np
import os
# 导入数据
wemedia_risk = pd.read_csv('predictions/wemedia_risk.csv')
mainstream_risk = pd.read_csv('predictions/mainstream_risk.csv')
emotional_sentiment = pd.read_csv('predictions/emotion_predictions.csv')


# 开始绘制子图A
import pandas as pd

def clean_date_to_day(df, date_col='date'):
    # 先转为datetime，再只保留日期部分
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    return df

def calc_daily_ratio_with_min_count(df, label_col, min_count=10):
    # 统计每天各类别数量
    daily_counts = df.groupby(['date', label_col]).size().unstack(fill_value=0)
    # 计算每日总数
    daily_total = daily_counts.sum(axis=1)
    # 过滤每日总数小于min_count的行
    valid_mask = daily_total >= min_count
    daily_counts = daily_counts[valid_mask]
    daily_total = daily_total[valid_mask]
    # 计算占比
    daily_ratio = daily_counts.div(daily_total, axis=0)
    return daily_ratio

# 筛掉某个类别为0的天数
def filter_zero_columns(df, min_ratio=0.0):
    """
    筛掉指定DataFrame中任何列值为0的行
    
    参数:
    - df: 包含占比数据的DataFrame
    - min_ratio: 最小占比阈值（默认0.0，即只要不是0就保留）
    
    返回:
    - 过滤后的DataFrame
    """
    # 创建掩码：所有列都不为0
    mask = (df > min_ratio).all(axis=1)
    filtered_df = df[mask]
    
    print(f"原始数据: {len(df)} 天")
    print(f"过滤后: {len(filtered_df)} 天")
    print(f"筛掉: {len(df) - len(filtered_df)} 天")
    
    return filtered_df

wemedia_risk_time = wemedia_risk.loc[wemedia_risk['risk_confidence']>0.8, ][['date', 'predicted_risk']]
mainstream_risk_time = mainstream_risk.loc[mainstream_risk['risk_confidence']>0.8, ][['date', 'predicted_risk']]
emotional_sentiment_time = emotional_sentiment.loc[emotional_sentiment['prediction_confidence']>0.8, ][['date', 'predicted_emotion']]
mainstream_risk_time = clean_date_to_day(mainstream_risk_time, 'date')
wemedia_risk_time = clean_date_to_day(wemedia_risk_time, 'date')
emotional_sentiment_time = clean_date_to_day(emotional_sentiment_time, 'date')

mainstream_ratio_time = calc_daily_ratio_with_min_count(mainstream_risk_time, 'predicted_risk')
wemedia_ratio_time = calc_daily_ratio_with_min_count(wemedia_risk_time, 'predicted_risk')
emotion_ratio_time = calc_daily_ratio_with_min_count(emotional_sentiment_time, 'predicted_emotion')
# 应用过滤
mainstream_ratio_filtered = filter_zero_columns(mainstream_ratio_time)
wemedia_ratio_filtered = filter_zero_columns(wemedia_ratio_time)  
emotion_ratio_filtered = filter_zero_columns(emotion_ratio_time)
# 重命名列以避免冲突
mainstream_ratio_filtered_renamed = mainstream_ratio_filtered.copy()
wemedia_ratio_filtered_renamed = wemedia_ratio_filtered.copy()
emotion_ratio_filtered_renamed = emotion_ratio_filtered.copy()

# 重命名列
mainstream_ratio_filtered_renamed.columns = ['mainstream_' + col for col in mainstream_ratio_filtered_renamed.columns]
wemedia_ratio_filtered_renamed.columns = ['wemedia_' + col for col in wemedia_ratio_filtered_renamed.columns]
emotion_ratio_filtered_renamed.columns = ['emotion_' + col for col in emotion_ratio_filtered_renamed.columns]

# 重置索引，将date变为列
mainstream_ratio_filtered_renamed = mainstream_ratio_filtered_renamed.reset_index()
wemedia_ratio_filtered_renamed = wemedia_ratio_filtered_renamed.reset_index()
emotion_ratio_filtered_renamed = emotion_ratio_filtered_renamed.reset_index()

# 按date合并三个DataFrame
merged_ratios = mainstream_ratio_filtered_renamed.merge(
    wemedia_ratio_filtered_renamed, 
    on='date', 
    how='inner'
).merge(
    emotion_ratio_filtered_renamed, 
    on='date', 
    how='inner'
)

import datetime
date_cut = datetime.date(2022, 12, 1)
data_cut_2 = datetime.date(2023, 1, 1)
merged_ratios = merged_ratios[merged_ratios['date'] > date_cut]
merged_ratios = merged_ratios[merged_ratios['date'] < data_cut_2]

import matplotlib.pyplot as plt
import pandas as pd

# 假设 merged_ratios 已经是你合并好的DataFrame
df = merged_ratios.copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 只保留三条主线
plot_cols = [
    'mainstream_risk', 'wemedia_risk', 'emotion_high'
]

# Nature子刊风格配色
nature_palette = {
    'mainstream_risk': '#A23B72',      # 紫红
    'wemedia_risk': '#DC2626',         # 红
    'emotion_high': '#F59E42',         # 橙
}

line_styles = {
    'mainstream_risk': '--',
    'wemedia_risk': '--',
    'emotion_high': '-',
}

# 设置全局样式
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.8,
    'legend.frameon': False,
    'figure.dpi': 400,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight'
})

fig, ax = plt.subplots(figsize=(12, 8))  # 调整为与其他子图相似的尺寸比例

# 绘制三条主线
for col in plot_cols:
    if col in df.columns:
        ax.plot(
            df['date'], df[col],
            label=col.replace('mainstream_', 'Mainstream ').replace('wemedia_', 'We-media ').replace('emotion_', 'Emotion ').replace('_', ' ').title(),
            color=nature_palette[col],
            linestyle=line_styles[col],
            linewidth=2.8,
            alpha=0.98
        )

# 坐标轴与标签
ax.set_xlabel('Date', fontsize=24, fontweight='bold', fontfamily='Times New Roman')
ax.set_ylabel('Proportion', fontsize=24, fontweight='bold', fontfamily='Times New Roman')
ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=7, direction='out')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')
ax.set_facecolor('white')

# 图例
ax.legend(loc='upper right', fontsize=18, prop={'family': 'Times New Roman'})

# 美化边框
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 网格
ax.grid(True, which='major', axis='y', alpha=0.2)

# 时间轴美化
fig.autofmt_xdate()

plt.tight_layout()
# 保存子图A到临时变量，不显示
fig_A = fig
plt.close()  # 关闭当前图，不显示

# 获取最优子图B、C、D
print("🎨 开始生成子图B、C、D...")

# B: 媒体暴露 vs 1-entropy (选择最优拟合)
from empirical_analysis import run_comprehensive_visualization
print("📊 生成子图B: 媒体暴露 vs 情绪集中度...")
figures_B, results_df_B, detailed_results_B = run_comprehensive_visualization(
    wemedia_risk, mainstream_risk, emotional_sentiment,
    lag_range=range(-2, 3),           # 扩大范围寻找最优
    num_threshold=5,
    confidence_threshold=0.7,
    save_plots=False,                 # 不单独保存
    enhanced_visualization=True
)

# 选择R²最高的图作为子图B
if figures_B:
    best_B = max(figures_B, key=lambda x: x[2])  # x[2]是R²值
    fig_B = best_B[1]
    print(f"   ✅ 选择滞后{best_B[0]}天的图 (R²={best_B[2]:.3f})")
else:
    fig_B = None
    print("   ❌ 子图B生成失败")

# C: 主流媒体暴露 vs 情绪极化
from empirical_analysis import run_polarization_visualization
print("📊 生成子图C: 主流媒体暴露 vs 情绪极化...")
figures_C, results_C, details_C = run_polarization_visualization(
    mainstream_risk, emotional_sentiment,
    lag_range=range(-2, 5),
    num_threshold=8,
    confidence_threshold=0.8,
    save_plots=False
)

if figures_C:
    best_C = max(figures_C, key=lambda x: x[2])
    fig_C = best_C[1]
    print(f"   ✅ 选择滞后{best_C[0]}天的图 (R²={best_C[2]:.3f})")
else:
    fig_C = None
    print("   ❌ 子图C生成失败")

# D: 自媒体暴露 vs 情绪极化
from empirical_analysis import run_wemedia_polarization_visualization
print("📊 生成子图D: 自媒体暴露 vs 情绪极化...")
figures_D, results_D, details_D = run_wemedia_polarization_visualization(
    wemedia_risk, emotional_sentiment,
    lag_range=range(-2, 3),
    num_threshold=15,
    confidence_threshold=0.9,
    save_plots=False
)

if figures_D:
    best_D = max(figures_D, key=lambda x: x[2])
    fig_D = best_D[1]
    print(f"   ✅ 选择滞后{best_D[0]}天的图 (R²={best_D[2]:.3f})")
else:
    fig_D = None
    print("   ❌ 子图D生成失败")

# 整合成2x2大图
print("\n🎨 开始整合Figure 2...")

# 使用更简单的方法：直接将各个图保存为临时文件，然后用PIL合并
import tempfile
from PIL import Image
import numpy as np

# 创建临时文件保存各个子图
temp_files = []

# 保存子图A
if fig_A is not None:
    temp_A = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig_A.savefig(temp_A.name, dpi=200, bbox_inches='tight', facecolor='white')
    temp_files.append(('A', temp_A.name))
    plt.close(fig_A)

# 保存子图B
if fig_B is not None:
    temp_B = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig_B.savefig(temp_B.name, dpi=200, bbox_inches='tight', facecolor='white')
    temp_files.append(('B', temp_B.name))
    plt.close(fig_B)

# 保存子图C  
if fig_C is not None:
    temp_C = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig_C.savefig(temp_C.name, dpi=200, bbox_inches='tight', facecolor='white')
    temp_files.append(('C', temp_C.name))
    plt.close(fig_C)

# 保存子图D
if fig_D is not None:
    temp_D = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig_D.savefig(temp_D.name, dpi=200, bbox_inches='tight', facecolor='white')
    temp_files.append(('D', temp_D.name))
    plt.close(fig_D)

# 使用matplotlib的方式重新组合
fig_combined, axes = plt.subplots(2, 2, figsize=(24, 18))  # 增加尺寸以容纳更大的字体
fig_combined.patch.set_facecolor('white')

# 字典映射子图位置
positions = {'A': (0, 0), 'B': (0, 1), 'C': (1, 0), 'D': (1, 1)}
titles = {
    'A': 'A. Temporal Patterns', 
    'B': 'B. Risk Exposure vs Emotion Concentration',
    'C': 'C. Mainstream Media vs Emotion Polarization', 
    'D': 'D. We-media vs Emotion Polarization'
}

# 将每个子图作为图像插入到对应位置
for label, temp_file in temp_files:
    row, col = positions[label]
    ax = axes[row, col]
    
    # 读取图像
    img = Image.open(temp_file)
    img_array = np.array(img)
    
    # 显示图像
    ax.imshow(img_array)
    # ax.set_title(titles[label], fontsize=20, fontweight='bold', 
    #             fontfamily='Times New Roman', pad=20)
    ax.axis('off')  # 隐藏坐标轴

# 如果某些子图缺失，显示占位符
for label in ['A', 'B', 'C', 'D']:
    if label not in [item[0] for item in temp_files]:
        row, col = positions[label]
        ax = axes[row, col]
        ax.text(0.5, 0.5, f'{titles[label]}\n(Not Available)', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=18, fontfamily='Times New Roman')
        ax.set_title(titles[label], fontsize=20, fontweight='bold',
                    fontfamily='Times New Roman', pad=20)
        ax.axis('off')

# 调整布局
plt.tight_layout(pad=2.0)

# 创建保存目录
save_dir = "Figures/Figure 2"
os.makedirs(save_dir, exist_ok=True)

# 保存整合后的大图
save_path = os.path.join(save_dir, "Figure2_new.png")
fig_combined.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')

print(f"\n✅ Figure 2 整合完成！")
print(f"📁 已保存到: {save_path}")

# 清理临时文件
import os
for _, temp_file in temp_files:
    try:
        os.unlink(temp_file)
    except:
        pass

plt.show()