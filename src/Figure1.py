#!/usr/bin/env python3
"""
为多次训练结果创建发表级可视化图表
支持三个任务：emotion、mainstream、wemedia
自动从保存的结果文件中读取数据，包含训练曲线分析
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy import stats
import json
from pathlib import Path
from typing import Dict, Any

# Nature期刊样式设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.5,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'figure.dpi': 400,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight'
})

# Nature子刊配色方案 - 为三个任务设计
TASK_COLORS = {
    'emotion': {
        'primary': '#2E86AB',      # 深蓝
        'secondary': '#A23B72',    # 深玫红
        'light': '#87CEEB',        # 天蓝
        'name': 'Emotion'
    },
    'mainstream': {
        'primary': '#059669',      # 深绿
        'secondary': '#10B981',    # 中绿
        'light': '#6EE7B7',        # 浅绿
        'name': 'Mainstream'
    },
    'wemedia': {
        'primary': '#DC2626',      # 深红
        'secondary': '#F59E0B',    # 橙黄
        'light': '#FCA5A5',        # 浅红
        'name': 'We-media'
    }
}

# 通用配色
NATURE_COLORS = {
    'dark': '#343a40',         # 深灰
    'grid': '#E2E8F0',         # 浅灰
    'background': '#FAFAFA'    # 背景色
}

def load_task_training_results(task_dirs):
    """从三个任务目录中加载训练结果"""
    
    print("🔍 加载三个任务的训练结果...")
    
    all_results = {}
    
    for task_name, task_dir in task_dirs.items():
        print(f"\n📊 处理任务: {task_name}")
        print(f"   目录: {task_dir}")
        
        task_path = Path(task_dir)
        if not task_path.exists():
            print(f"   ❌ 目录不存在: {task_path}")
            continue
        
        # 尝试读取CSV文件
        csv_path = task_path / "analysis" / "multi_run_results.csv"
        if csv_path.exists():
            print(f"   ✅ 找到CSV结果文件: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"   📊 加载了 {len(df)} 次训练结果")
            
            # 加载训练曲线
            training_curves = load_task_training_curves(task_path)
            
            all_results[task_name] = {
                'results_df': df,
                'training_curves': training_curves
            }
        else:
            # 尝试从JSON报告中提取
            json_path = task_path / "analysis" / "comprehensive_report.json"
            if json_path.exists():
                print(f"   ✅ 找到JSON报告文件: {json_path}")
                with open(json_path, 'r') as f:
                    report = json.load(f)
                
                if 'individual_results' in report:
                    df = pd.DataFrame(report['individual_results'])
                    print(f"   📊 从JSON加载了 {len(df)} 次训练结果")
                    
                    # 加载训练曲线
                    training_curves = load_task_training_curves(task_path)
                    
                    all_results[task_name] = {
                        'results_df': df,
                        'training_curves': training_curves
                    }
                else:
                    print(f"   ❌ JSON文件格式不正确")
            else:
                print(f"   ❌ 未找到结果文件")
    
    return all_results

def load_task_training_curves(task_dir):
    """加载单个任务的训练曲线数据"""
    
    print(f"   📈 加载训练曲线数据...")
    training_curves = []
    
    for run_dir in sorted(task_dir.glob("run_*_seed_*")):
        run_info = {
            'run_id': int(run_dir.name.split('_')[1]),
            'seed': int(run_dir.name.split('_')[-1]),
            'training_data': None,
            'validation_data': None
        }
        
        # 加载训练曲线
        train_file = run_dir / "training_metrics.csv"
        val_file = run_dir / "validation_metrics.csv"
        
        if train_file.exists():
            try:
                train_df = pd.read_csv(train_file)
                run_info['training_data'] = train_df
            except Exception as e:
                print(f"     ❌ 加载训练曲线失败: {e}")
        
        if val_file.exists():
            try:
                val_df = pd.read_csv(val_file)
                run_info['validation_data'] = val_df
            except Exception as e:
                print(f"     ❌ 加载验证曲线失败: {e}")
        
        if run_info['training_data'] is not None or run_info['validation_data'] is not None:
            training_curves.append(run_info)
    
    print(f"     📊 加载了 {len(training_curves)} 个运行的训练曲线")
    return training_curves

def aggregate_training_curves(training_curves):
    """聚合多次运行的训练曲线，计算均值和标准差"""
    
    if not training_curves:
        return None, None
    
    # 聚合训练损失曲线
    train_aggregated = None
    if any(curve['training_data'] is not None for curve in training_curves):
        all_train_data = []
        
        for curve in training_curves:
            if curve['training_data'] is not None:
                df = curve['training_data'].copy()
                df['run_id'] = curve['run_id']
                all_train_data.append(df)
        
        if all_train_data:
            combined_train = pd.concat(all_train_data, ignore_index=True)
            
            # 按epoch聚合
            train_aggregated = combined_train.groupby('epoch').agg({
                'train_loss': ['mean', 'std', 'count'],
                'learning_rate': 'mean'
            }).reset_index()
            
            # 展平列名
            train_aggregated.columns = ['epoch', 'train_loss_mean', 'train_loss_std', 'train_loss_count', 'learning_rate']
            train_aggregated['train_loss_sem'] = train_aggregated['train_loss_std'] / np.sqrt(train_aggregated['train_loss_count'])
    
    # 聚合验证准确率曲线
    val_aggregated = None
    if any(curve['validation_data'] is not None for curve in training_curves):
        all_val_data = []
        
        for curve in training_curves:
            if curve['validation_data'] is not None:
                df = curve['validation_data'].copy()
                df['run_id'] = curve['run_id']
                all_val_data.append(df)
        
        if all_val_data:
            combined_val = pd.concat(all_val_data, ignore_index=True)
            
            # 按epoch聚合
            val_aggregated = combined_val.groupby('epoch').agg({
                'eval_accuracy': ['mean', 'std', 'count'],
                'eval_loss': ['mean', 'std', 'count']
            }).reset_index()
            
            # 展平列名
            val_aggregated.columns = ['epoch', 'eval_accuracy_mean', 'eval_accuracy_std', 'eval_accuracy_count', 
                                    'eval_loss_mean', 'eval_loss_std', 'eval_loss_count']
            val_aggregated['eval_accuracy_sem'] = val_aggregated['eval_accuracy_std'] / np.sqrt(val_aggregated['eval_accuracy_count'])
            val_aggregated['eval_loss_sem'] = val_aggregated['eval_loss_std'] / np.sqrt(val_aggregated['eval_loss_count'])
    
    return train_aggregated, val_aggregated

def create_results_visualization(task_dirs=None):
    """基于三个任务的训练结果创建可视化"""
    
    # 默认任务目录
    if task_dirs is None:
        task_dirs = {
            'emotion': 'multi_run_experiments_emotion',
            'mainstream': 'multi_run_experiments_mainstream', 
            'wemedia': 'multi_run_experiments_wemedia'
        }
    
    # 加载所有任务的数据
    all_results = load_task_training_results(task_dirs)
    
    if not all_results:
        print("❌ 无法加载任何任务的训练结果")
        return None, None
    
    print(f"\n✅ 成功加载 {len(all_results)} 个任务的数据")
    
    # 创建2x3布局的图表（恢复6个子图）
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    
    # A. 测试准确率概率密度分布图
    ax1 = fig.add_subplot(gs[0, 0])
    
    for task_name, task_data in all_results.items():
        if 'test_accuracy' in task_data['results_df'].columns:
            accuracy_data = task_data['results_df']['test_accuracy'] * 100
            
            # 使用KDE绘制概率密度曲线
            kde = stats.gaussian_kde(accuracy_data)
            x_range = np.linspace(accuracy_data.min() - 2, accuracy_data.max() + 2, 100)
            density = kde(x_range)
            
            color_config = TASK_COLORS[task_name]
            ax1.fill_between(x_range, density, alpha=0.3, color=color_config['primary'], 
                           label=f"{color_config['name']}")
            ax1.plot(x_range, density, color=color_config['primary'], linewidth=2)
            
            # 添加均值线
            mean_val = accuracy_data.mean()
            ax1.axvline(mean_val, color=color_config['primary'], linestyle='--', 
                       alpha=0.8, linewidth=1.5)
    
    ax1.set_xlabel('Test Accuracy (%)')
    ax1.set_ylabel('Probability Density')
    # ax1.set_title('a. Test Accuracy Probability Density Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # B. F1-Macro的箱线图+散点图组合
    ax2 = fig.add_subplot(gs[0, 1])
    
    box_data_f1_macro = []
    box_labels = []
    colors = []
    
    for i, (task_name, task_data) in enumerate(all_results.items()):
        if 'test_f1_macro' in task_data['results_df'].columns:
            f1_macro_data = task_data['results_df']['test_f1_macro'] * 100
            box_data_f1_macro.append(f1_macro_data)
            box_labels.append(TASK_COLORS[task_name]['name'])
            colors.append(TASK_COLORS[task_name]['primary'])
    
    if box_data_f1_macro:
        # 创建箱线图
        box_plot = ax2.boxplot(box_data_f1_macro, positions=range(len(box_data_f1_macro)), 
                              patch_artist=True, showfliers=False)
        
        # 设置箱线图颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加散点图
        for i, (data, color) in enumerate(zip(box_data_f1_macro, colors)):
            x_positions = np.random.normal(i, 0.04, len(data))
            ax2.scatter(x_positions, data, alpha=0.6, s=20, 
                       color=color, edgecolors='white', linewidth=0.5)
    
    ax2.set_ylabel('F1-Macro Score (%)')
    # ax2.set_title('b. F1-Macro Distribution & Variability Analysis', fontweight='bold')
    ax2.set_xticks(range(len(box_labels)))
    ax2.set_xticklabels(box_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # C. F1-Weighted的箱线图+散点图组合
    ax3 = fig.add_subplot(gs[0, 2])
    
    box_data_f1_weighted = []
    box_labels_weighted = []
    colors_weighted = []
    
    for i, (task_name, task_data) in enumerate(all_results.items()):
        if 'test_f1_weighted' in task_data['results_df'].columns:
            f1_weighted_data = task_data['results_df']['test_f1_weighted'] * 100
            box_data_f1_weighted.append(f1_weighted_data)
            box_labels_weighted.append(TASK_COLORS[task_name]['name'])
            colors_weighted.append(TASK_COLORS[task_name]['primary'])
    
    if box_data_f1_weighted:
        # 创建箱线图
        box_plot = ax3.boxplot(box_data_f1_weighted, positions=range(len(box_data_f1_weighted)), 
                              patch_artist=True, showfliers=False)
        
        # 设置箱线图颜色
        for patch, color in zip(box_plot['boxes'], colors_weighted):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加散点图
        for i, (data, color) in enumerate(zip(box_data_f1_weighted, colors_weighted)):
            x_positions = np.random.normal(i, 0.04, len(data))
            ax3.scatter(x_positions, data, alpha=0.6, s=20, 
                       color=color, edgecolors='white', linewidth=0.5)
    
    ax3.set_ylabel('F1-Weighted Score (%)')
    # ax3.set_title('c. F1-Weighted Distribution & Variability Analysis', fontweight='bold')
    ax3.set_xticks(range(len(box_labels_weighted)))
    ax3.set_xticklabels(box_labels_weighted, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # D. Test Recall概率密度分布图
    ax4 = fig.add_subplot(gs[1, 0])
    
    for task_name, task_data in all_results.items():
        # 从每个run的evaluation_results.json中提取recall数据
        recall_values = []
        
        # 获取任务目录路径
        task_dir = task_dirs[task_name]
        task_path = Path(task_dir)
        
        if task_path.exists():
            # 遍历所有run目录
            for run_dir in sorted(task_path.glob("run_*_seed_*")):
                eval_file = run_dir / "evaluation_results.json"
                
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                        
                        # 提取recall数据，优先级：macro avg > weighted avg > 整体accuracy
                        if 'classification_report' in eval_data:
                            report = eval_data['classification_report']
                            
                            if 'macro avg' in report and 'recall' in report['macro avg']:
                                recall_val = report['macro avg']['recall'] * 100
                                recall_values.append(recall_val)
                            elif 'weighted avg' in report and 'recall' in report['weighted avg']:
                                recall_val = report['weighted avg']['recall'] * 100
                                recall_values.append(recall_val)
                            elif 'accuracy' in report:
                                # 对于某些任务，accuracy可能等同于recall
                                recall_val = report['accuracy'] * 100
                                recall_values.append(recall_val)
                        
                        # 备选：从test_metrics中获取
                        elif 'test_metrics' in eval_data and 'eval_accuracy' in eval_data['test_metrics']:
                            recall_val = eval_data['test_metrics']['eval_accuracy'] * 100
                            recall_values.append(recall_val)
                            
                    except Exception as e:
                        print(f"     ⚠️ 读取{eval_file}失败: {e}")
                        continue
        
        # 如果收集到了recall数据，绘制概率密度曲线
        if len(recall_values) > 1:  # 至少需要2个数据点才能绘制KDE
            recall_data = np.array(recall_values)
            
            # 使用KDE绘制概率密度曲线
            kde = stats.gaussian_kde(recall_data)
            x_range = np.linspace(recall_data.min() - 2, recall_data.max() + 2, 100)
            density = kde(x_range)
            
            color_config = TASK_COLORS[task_name]
            ax4.fill_between(x_range, density, alpha=0.3, color=color_config['primary'], 
                           label=f"{color_config['name']}")
            ax4.plot(x_range, density, color=color_config['primary'], linewidth=2)
            
            # 添加均值线
            mean_recall = recall_data.mean()
            ax4.axvline(mean_recall, color=color_config['primary'], linestyle='--', 
                       alpha=0.8, linewidth=1.5)
            
            print(f"   ✅ {task_name}: 找到 {len(recall_values)} 个recall数据点，均值={mean_recall:.1f}%")
        
        elif len(recall_values) == 1:
            # 只有一个数据点，绘制垂直线
            color_config = TASK_COLORS[task_name]
            ax4.axvline(recall_values[0], color=color_config['primary'], 
                       linewidth=3, alpha=0.8, label=f"{color_config['name']}")
            print(f"   ⚠️ {task_name}: 只找到 1 个recall数据点: {recall_values[0]:.1f}%")
        
        else:
            print(f"   ❌ {task_name}: 没有找到recall数据")
    
    ax4.set_xlabel('Test Recall (%)')
    ax4.set_ylabel('Probability Density')
    # ax4.set_title('d. Test Recall Probability Density Distribution', fontweight='bold')
    
    # 只有在有图例数据时才显示图例
    handles, labels = ax4.get_legend_handles_labels()
    if handles:
        ax4.legend()
    
    ax4.grid(True, alpha=0.3)
    
    # E. 最终验证准确率对比（Nature风格柱状图，简洁版）
    ax5 = fig.add_subplot(gs[1, 1])
    
    final_val_accs = []
    final_val_acc_labels = []
    final_val_acc_colors = []
    
    for task_name, task_data in all_results.items():
        train_agg, val_agg = aggregate_training_curves(task_data['training_curves'])
        
        if val_agg is not None:
            # 获取最终验证准确率的平均值
            final_acc_mean = val_agg['eval_accuracy_mean'].iloc[-1] * 100
            
            final_val_accs.append(final_acc_mean)
            final_val_acc_labels.append(TASK_COLORS[task_name]['name'])
            final_val_acc_colors.append(TASK_COLORS[task_name]['primary'])
    
    if final_val_accs:
        # Nature风格柱状图设计（简洁版，无误差棒）
        x_pos = np.arange(len(final_val_accs))
        
        # 创建柱状图
        bars = ax5.bar(x_pos, final_val_accs, 
                      color=final_val_acc_colors,
                      alpha=0.85,
                      edgecolor='white', 
                      linewidth=2,
                      width=0.6)
        
        # 添加顶部数值标签（Nature风格）
        for i, (bar, acc) in enumerate(zip(bars, final_val_accs)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', 
                    ha='center', va='bottom', 
                    fontweight='bold', 
                    fontsize=10,
                    color=NATURE_COLORS['dark'])
            
            # 柱子样式
            bar.set_facecolor(final_val_acc_colors[i])
            bar.set_edgecolor('white')
        
        ax5.set_ylabel('Final Test Accuracy (%)', fontweight='600')
        # ax5.set_title('e. Final Performance Comparison', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([label.replace(' ', '\n') for label in final_val_acc_labels], 
                           rotation=0, ha='center', fontweight='500')
        
        # 移除网格线
        ax5.grid(False)
        
        # 设置y轴范围，突出差异
        if len(final_val_accs) > 1:
            y_min = min(final_val_accs) - 2
            y_max = max(final_val_accs) + 4
            ax5.set_ylim(y_min, y_max)
        
        # 移除顶部和右侧的边框
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
    
    # F. 训练稳定性分析（Nature风格柱状图，无网格）
    ax6 = fig.add_subplot(gs[1, 2])
    
    stability_metrics = []
    stability_labels = []
    stability_colors = []
    
    for task_name, task_data in all_results.items():
        train_agg, val_agg = aggregate_training_curves(task_data['training_curves'])
        
        if val_agg is not None:
            # 计算后半期的稳定性（变异系数）
            mid_point = len(val_agg) // 2
            late_period_std = val_agg['eval_accuracy_std'].iloc[mid_point:].mean() * 100
            late_period_mean = val_agg['eval_accuracy_mean'].iloc[mid_point:].mean() * 100
            
            if late_period_mean > 0:
                cv = late_period_std / late_period_mean * 100  # 变异系数百分比
                stability_metrics.append(cv)
                stability_labels.append(TASK_COLORS[task_name]['name'])
                stability_colors.append(TASK_COLORS[task_name]['primary'])
    
    if stability_metrics:
        # Nature风格柱状图设计
        x_pos = np.arange(len(stability_metrics))
        
        bars = ax6.bar(x_pos, stability_metrics,
                      color=stability_colors,
                      alpha=0.85,
                      edgecolor='white',
                      linewidth=2,
                      width=0.6)
        
        # 添加数值标签
        for i, (bar, cv) in enumerate(zip(bars, stability_metrics)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{cv:.2f}%', 
                    ha='center', va='bottom', 
                    fontweight='bold',
                    fontsize=10,
                    color=NATURE_COLORS['dark'])
        
        ax6.set_ylabel('Coefficient of Variation (%)', fontweight='600')
        # ax6.set_title('f. Training Stability Analysis\n(Lower = More Stable)', fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([label.replace(' ', '\n') for label in stability_labels], 
                           rotation=0, ha='center', fontweight='500')
        
        # 移除网格线
        ax6.grid(False)
        
        # 移除顶部和右侧的边框
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        
        # # 添加稳定性等级标注（Nature风格）
        # max_cv = max(stability_metrics)
        # if max_cv < 1:
        #     stability_note = "Excellent Stability"
        #     note_color = '#2ca02c'  # 绿色
        # elif max_cv < 2:
        #     stability_note = "Good Stability"
        #     note_color = '#ff7f0e'  # 橙色
        # else:
        #     stability_note = "Moderate Stability"
        #     note_color = '#d62728'  # 红色
        
        # ax6.text(0.98, 0.95, stability_note, 
        #         transform=ax6.transAxes,
        #         ha='right', va='top', 
        #         fontsize=9, 
        #         fontweight='600',
        #         color=note_color,
        #         bbox=dict(boxstyle='round,pad=0.4', 
        #                  facecolor='white', 
        #                  edgecolor=note_color,
        #                  alpha=0.9,
        #                  linewidth=1.5))
    
    # 去掉大标题
    # plt.suptitle('Multi-Task Training Analysis: Comprehensive Performance Evaluation\n' + 
    #              f'(Independent Runs Across {len(all_results)} Tasks)', 
    #              fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图表
    output_dir = Path("multi_task_analysis")
    output_dir.mkdir(exist_ok=True)
    
    fig_path = output_dir / 'multi_task_performance_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    
    print(f"\n✅ 多任务性能分析图已保存: {fig_path}")
    plt.show()
    
    # 打印详细统计总结
    print("\n" + "="*80)
    print("📊 MULTI-TASK STATISTICAL SUMMARY FOR MANUSCRIPT")
    print("="*80)
    
    for task_name, task_data in all_results.items():
        df = task_data['results_df']
        color_config = TASK_COLORS[task_name]
        
        print(f"\n🎯 {color_config['name']}:")
        print("-" * 40)
        
        # 计算详细统计
        for metric in ['test_accuracy', 'test_f1_macro', 'test_f1_weighted']:
            if metric in df.columns:
                values = df[metric].values * 100
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0
                median_val = np.median(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                metric_name = metric.replace('test_', '').replace('_', '-').title()
                print(f"  {metric_name}: {mean_val:.2f}% ± {std_val:.2f}% [{min_val:.2f}%-{max_val:.2f}%]")
        
        # 添加训练损失统计
        train_agg, _ = aggregate_training_curves(task_data['training_curves'])
        if train_agg is not None:
            final_losses = []
            for curve in task_data['training_curves']:
                if curve['training_data'] is not None:
                    final_loss = curve['training_data']['train_loss'].iloc[-1]
                    final_losses.append(final_loss)
            
            if final_losses:
                final_losses = np.array(final_losses)
                mean_loss = np.mean(final_losses)
                std_loss = np.std(final_losses, ddof=1) if len(final_losses) > 1 else 0
                print(f"  Final-Training-Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        
        print(f"  Sample size: n = {len(df)} runs")
    
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    # 可以手动指定任务目录
    task_directories = {
        'emotion': 'multi_run_experiments_emotion',
        'mainstream': 'multi_run_experiments_mainstream', 
        'wemedia': 'multi_run_experiments_wemedia'
    }
    
    results = create_results_visualization(task_directories) 