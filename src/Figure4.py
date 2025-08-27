#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4: Macro Model Explanation - System Dynamics Analysis
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

def create_figure_4_visualization(data_base_dir: str = "kappa120_scan_new_full_test",
                                 output_dir: str = "Figures/Figure 4",
                                 phi_range: List[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55],
                                 phi_range_second_order: List[float] = [0.01, 0.1, 0.2],  # 🔧 新增：二阶相变点
                                 phi_range_first_order: List[float] = [0.3, 0.4, 0.5, 0.55],  # 🔧 新增：一阶相变点
                                 theta_range: List[float] = [0.55],
                                 kappa: int = 120):
    """
    创建Figure 4的完整可视化，直接调用现有函数
    
    参数:
        data_base_dir: 数据基础目录
        output_dir: 输出目录
        phi_range: phi值范围（用于B、C、D图）
        phi_range_second_order: 二阶相变点phi值（用于A图关联长度）
        phi_range_first_order: 一阶相变点phi值（备用）
        theta_range: theta值范围
        kappa: kappa参数
    """
    print("🎨 开始生成 Figure 4: Macro Model Explanation...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建对比器实例
    comparator = CriticalPointComparator(
        data_base_dir=data_base_dir,
        output_dir=output_dir
    )
    
    # 注释掉关联长度子图A的生成
    print("📏 子图A: Correlation Length Comparison - 已注释掉")
    print(f"   原计划使用二阶相变点: {phi_range_second_order}")
    # A. Correlation Length Basic - 🔧 暂时注释掉
    """
    success_a = comparator.compare_correlation_lengths(
        phi_range=phi_range_second_order,  # 🔧 使用二阶相变点
        theta_range=theta_range,
        kappa=kappa,
        save_name="basic_correlation_length"
    )
    """
    success_a = True  # 临时设置为成功，避免后续逻辑错误
    
    # 注释掉子图B的生成
    print("📺 子图B: Media Influence - 已注释掉")
    # B. Media Influence - 🔧 暂时注释掉
    """
    success_b = comparator.compare_media_influence(
        phi_range=phi_range,
        theta_range=theta_range,
        kappa=kappa,
        save_name="basic_media_influence"
    )
    """
    success_b = True  # 临时设置为成功，避免后续逻辑错误
    
    print("🎯 生成子图C: Detailed Polarization...")
    # C. Detailed Polarization
    success_c = comparator.analyze_detailed_polarization(
        phi_range=phi_range,
        theta_range=theta_range,
        kappa=kappa,
        save_name="basic_detailed_polarization"
    )
    
    print("😐 生成子图D: Medium Arousal Analysis...")
    # D. Medium Arousal Analysis
    success_d = comparator.analyze_medium_arousal(
        phi_range=phi_range,
        theta_range=theta_range,
        kappa=kappa,
        save_name="basic_medium_arousal"
    )
    
    if all([success_a, success_b, success_c, success_d]):
        print("✅ 所有子图生成成功!")
        
        # 现在组合成2×2布局
        print("🎨 组合子图到Figure 4...")
        combine_subplots_to_figure4(output_dir)
        
        return True
    else:
        print("❌ 部分子图生成失败!")
        print(f"Correlation Length: {'✅' if success_a else '❌'}")
        print(f"Media Influence: {'✅' if success_b else '❌'}")
        print(f"Detailed Polarization: {'✅' if success_c else '❌'}")
        print(f"Medium Arousal: {'✅' if success_d else '❌'}")
        return False

def combine_subplots_to_figure4(source_dir: str):
    """
    将生成的单图组合成Figure 4的1×2布局 - 优化版本
    """
    import matplotlib.image as mpimg
    
    # 设置高质量图形参数
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # 🔧 改为1×2布局，增大图形尺寸
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))  # 1行2列，高度适当调整
    
    # 定义子图文件路径
    subplot_files = {
        'C': os.path.join(source_dir, "detailed_polarization_basic_detailed_polarization.png"),
        'D': os.path.join(source_dir, "medium_arousal_basic_medium_arousal.png")
    }
    
    # 调整布局：使用2个子图
    axes = [ax1, ax2]  # 直接使用ax1和ax2
    labels = ['C', 'D']  # 子图标签
    
    # 加载并显示其他子图
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
                
                # 🔧 删除子图标签，用户后续会自己添加
                # ax.text(-0.05, 1.05, label, transform=ax.transAxes, 
                #        fontsize=18, fontweight='bold')
                
                print(f"✅ 子图{label}加载成功: {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"❌ 加载子图{label}失败: {e}")
                ax.text(0.5, 0.5, f'Failed to load\nSubplot {label}', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round", facecolor="lightgray"))
                ax.axis('off')
        else:
            print(f"❌ 子图文件不存在: {file_path}")
            ax.text(0.5, 0.5, f'File not found\nSubplot {label}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle="round", facecolor="lightcoral"))
            ax.axis('off')
    
    # 🔧 删除总标题，让子图更大
    # plt.suptitle('Figure 4: Macro Model Explanation - System Dynamics Analysis', 
    #             fontsize=20, fontweight='bold', y=0.95)
    
    # 🔧 优化布局 - 减少列间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98,  
                       wspace=0.05)  # 只调整列间距，移除行间距参数
    
    # 保存组合图
    output_path = os.path.join(source_dir, "Figure_4_macro_model_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1,  # 🔧 减少padding
               facecolor='white', edgecolor='none')
    
    # plt.close()
    
    print(f"✅ Figure 4 组合图已保存: {output_path}")
    
    # 恢复全局设置
    plt.rcParams['font.family'] = plt.rcParamsDefault['font.family']

if __name__ == "__main__":
    # 运行示例
    print("🎨 生成 Figure 4: Macro Model Explanation...")
    
    # 参数设置
    data_base_dir = "kappa120_scan_new_full_test"
    output_dir = "Figures/Figure 4"
    phi_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55]
    phi_range_second_order = [0.01, 0.1, 0.2]
    phi_range_first_order = [0.3, 0.4, 0.5, 0.55]
    theta_range = [0.55]
    kappa = 120
    
    # 生成图像
    success = create_figure_4_visualization(
        data_base_dir=data_base_dir,
        output_dir=output_dir,
        phi_range=phi_range,
        phi_range_second_order=phi_range_second_order,
        phi_range_first_order=phi_range_first_order,
        theta_range=theta_range,
        kappa=kappa
    )
    
    if success:
        print("✅ Figure 4 生成完成！")
    else:
        print("❌ Figure 4 生成失败！")
