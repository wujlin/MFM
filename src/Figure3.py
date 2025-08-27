#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3: Macro-level Analysis - Phase Transition Characteristics
直接调用对比不同相变点.py中的现有函数，组合成2×2布局
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 添加项目路径
sys.path.append('.')
sys.path.append('src')

# 导入现有的对比器类
from src.对比不同相变点 import CriticalPointComparator

def create_figure_3_visualization(data_base_dir: str = "kappa120_scan_new_full_test",
                                 output_dir: str = "Figures/Figure 3",
                                 phi_range: List[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55],
                                 theta_range_abc: List[float] = [0.55],  # 🔧 A、B、C图用单个θ值
                                 theta_range_d: List[float] = [0.40, 0.45, 0.50, 0.55],  # 🔧 D图用多个θ值
                                 kappa: int = 120):
    """
    创建Figure 3的完整可视化，直接调用现有函数
    
    参数:
        data_base_dir: 数据基础目录
        output_dir: 输出目录
        phi_range: phi值范围
        theta_range: theta值范围
        kappa: kappa参数
    """
    print("🎨 开始生成 Figure 3: Macro-level Analysis...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建对比器实例
    comparator = CriticalPointComparator(
        data_base_dir=data_base_dir,
        output_dir=output_dir
    )
    
    print("📊 生成子图A: Order Parameter Comparison...")
    print(f"   使用参数: phi_range={len(phi_range)}个值, theta_range={theta_range_abc}")
    # A. Order Parameter Basic - 使用单个theta值
    success_a = comparator.compare_order_parameters(
        phi_range=phi_range,
        theta_range=theta_range_abc,  # 🔧 单个θ值，简洁清晰
        kappa=kappa,
        save_name="basic_order_parameter"
    )
    
    print("📈 生成子图B: Three State Lines...")
    print(f"   使用参数: phi_range={len(phi_range)}个值, theta_range={theta_range_abc}")
    # B. Three State Lines - 使用单个theta值
    success_b = comparator.compare_three_state_distribution(
        phi_range=phi_range,
        theta_range=theta_range_abc,  # 🔧 单个θ值，简洁清晰
        kappa=kappa,
        save_name="basic_three_state_distribution"
    )
    
    print("📊 生成子图C: Three State Contour...")
    # C. Three State Contour (由上面的函数同时生成contour和lines)
    success_c = success_b  # contour由compare_three_state_distribution生成
    
    print("📊 生成子图D: Jump Amplitude Analysis...")
    print(f"   使用参数: phi_range={len(phi_range)}个值, theta_range={theta_range_d}")
    # D. Jump Amplitude Analysis - 使用多个theta值生成复合图
    success_d = comparator.analyze_jump_amplitude(
        phi_range=phi_range,
        theta_range=theta_range_d,  # 🔧 多个θ值，生成热力图+棒棒图组合
        kappa=kappa,
        save_name="detailed_stability_jump_amplitude"
    )
    
    if all([success_a, success_b, success_c, success_d]):
        print("✅ 所有子图生成成功!")
        
        # 现在组合成2×2布局
        print("🎨 组合子图到Figure 3...")
        combine_subplots_to_figure3(output_dir)
        
        return True
    else:
        print("❌ 部分子图生成失败!")
        print(f"Order Parameter: {'✅' if success_a else '❌'}")
        print(f"Three State Lines: {'✅' if success_b else '❌'}")
        print(f"Three State Contour: {'✅' if success_c else '❌'}")
        print(f"Jump Amplitude: {'✅' if success_d else '❌'}")
        return False

def combine_subplots_to_figure3(source_dir: str):
    """
    将生成的单图组合成Figure 3的2×2布局
    """
    import matplotlib.image as mpimg
    from matplotlib.patheffects import withStroke
    
    # 设置高质量图形参数
    plt.rcParams['figure.dpi'] = 150  # 🔧 降低DPI，减小整体尺寸
    plt.rcParams['savefig.dpi'] = 300  # 保存时使用高DPI
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 32  # 🔧 大幅增加基础字体大小
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rcParams['xtick.major.width'] = 2.5
    plt.rcParams['ytick.major.width'] = 2.5
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.weight'] = 'normal'
    
    # 🔧 创建更合理尺寸的2×2布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 定义子图文件路径
    subplot_files = {
        'A': os.path.join(source_dir, "order_parameter_basic_order_parameter.png"),
        'B': os.path.join(source_dir, "three_state_lines_basic_three_state_distribution_lines.png"),
        'C': os.path.join(source_dir, "three_state_contour_basic_three_state_distribution_contour.png"),
        'D': os.path.join(source_dir, "jump_amplitude_detailed_stability_jump_amplitude.png")
    }
    
    axes = [ax1, ax2, ax3, ax4]
    labels = ['A', 'B', 'C', 'D']
    
    # 🔧 为每个子图设置不同的精确标签位置 - 确保完美对齐
    label_positions = [
        (-0.18, 1.12),  # A: 序参量图，需要更多左偏移避开图例，稍微上移
        (-0.15, 1.12),  # B: 三稳态折线图，标准位置
        (-0.15, 1.12),  # C: 三稳态等高线图，标准位置
        (-0.15, 1.12)   # D: 跳变幅度分析，标准位置
    ]
    
    # 🔧 优化标签字体和样式，减小尺寸避免过大
    label_fontsize = 32  # 🔧 减小标签字体从45到32
    label_style = {
        'fontsize': label_fontsize,
        'fontweight': 'bold',
        'fontfamily': 'serif',
        'color': 'black',
        'ha': 'center',
        'va': 'center',
        'bbox': dict(
            boxstyle="round,pad=0.4",  # 🔧 减小内边距
            facecolor="white", 
            edgecolor="black", 
            alpha=0.95,  # 🔧 稍微减少不透明度
            linewidth=2.0  # 🔧 减小边框粗细
        ),
        # 🔧 减少阴影效果
        'path_effects': [withStroke(linewidth=3, foreground='white')]
    }
    
    # 加载并显示每个子图
    for i, (ax, label) in enumerate(zip(axes, labels)):
        subplot_key = label
        file_path = subplot_files[subplot_key]
        
        if os.path.exists(file_path):
            try:
                # 加载图片
                img = mpimg.imread(file_path)
                
                # 显示图片
                ax.imshow(img)
                ax.axis('off')  # 隐藏坐标轴
                
                # 🔧 删除子图标签 - 用户会手动添加
                # pos_x, pos_y = label_positions[i]
                # ax.text(pos_x, pos_y, label, transform=ax.transAxes, **label_style)
                
                print(f"✅ 子图{label}加载成功: {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"❌ 加载子图{label}失败: {e}")
                ax.text(0.5, 0.5, f'Failed to load\nSubplot {label}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=16, bbox=dict(boxstyle="round", facecolor="lightgray"))
                ax.axis('off')
        else:
            print(f"❌ 子图文件不存在: {file_path}")
            ax.text(0.5, 0.5, f'File not found\nSubplot {label}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, bbox=dict(boxstyle="round", facecolor="lightcoral"))
            ax.axis('off')
    
    # 🔧 删除总标题 - 按用户要求放在正文中
    # fig.suptitle('Figure 3: Macro-level Analysis', fontsize=42, fontweight='bold', y=0.98)
    
    # 🔧 调整布局 - 优化间距以适应更大的标签
    plt.tight_layout()
    plt.subplots_adjust(top=0.89, bottom=0.03, left=0.03, right=0.97, 
                       wspace=0.06, hspace=0.10)  # 🔧 进一步增加间距以容纳更大的标签
    
    # 🔧 保存组合图 - 优化保存参数
    output_path = os.path.join(source_dir, "Figure_3_macro_analysis_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.15,
               facecolor='white', edgecolor='none', format='png',
               metadata={'Title': 'Figure 3: Macro-level Analysis'})  # 🔧 添加元数据
    
    # 同时保存高质量PDF版本
    pdf_path = os.path.join(source_dir, "Figure_3_macro_analysis_combined.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.15,
               facecolor='white', edgecolor='none', format='pdf',
               metadata={'Title': 'Figure 3: Macro-level Analysis'})
    
    # 🔧 显示图像以便检查
    plt.show()
    
    print(f"✅ Figure 3 组合图已保存: {output_path}")
    print(f"✅ Figure 3 PDF版本已保存: {pdf_path}")
    print(f"🎨 图像优化完成:")
    print(f"   - 标签字体大小: {label_fontsize}pt")
    print(f"   - 标题字体大小: 34pt")
    print(f"   - 添加了阴影效果和边框")
    print(f"   - 优化了对齐和间距")
    
    # 恢复全局设置
    plt.rcParams['font.family'] = plt.rcParamsDefault['font.family']
    plt.rcParams['figure.dpi'] = plt.rcParamsDefault['figure.dpi']
    plt.rcParams['savefig.dpi'] = plt.rcParamsDefault['savefig.dpi']
    plt.rcParams['font.size'] = plt.rcParamsDefault['font.size']
    plt.rcParams['text.usetex'] = plt.rcParamsDefault['text.usetex']
    plt.rcParams['font.weight'] = plt.rcParamsDefault['font.weight']

if __name__ == "__main__":
    # 运行示例
    print("🎨 生成 Figure 3: Macro-level Analysis...")
    
    # 参数设置 - 🔧 确保多个θ值以生成完整复合图
    data_base_dir = "kappa120_scan_new_full_test"
    output_dir = "Figures/Figure 3"
    phi_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55]
    theta_range_abc = [0.55]  # 🔧 A、B、C图用单个θ值
    theta_range_d = [0.40, 0.45, 0.50, 0.55]  # 🔧 D图用多个θ值
    kappa = 120
    
    # 生成图像
    success = create_figure_3_visualization(
        data_base_dir=data_base_dir,
        output_dir=output_dir,
        phi_range=phi_range,
        theta_range_abc=theta_range_abc,
        theta_range_d=theta_range_d,
        kappa=kappa
    )
    
    if success:
        print("✅ Figure 3 生成完成！")
    else:
        print("❌ Figure 3 生成失败！")
