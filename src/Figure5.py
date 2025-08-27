#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5: 微观层面分析 - 理论验证与转换动力学
包含5个子图：
A. 3D气泡极化分析（来自test_3d_bubble_connection_analysis.py），位于左侧
B. 理论验证误差分析（来自test_optimized_theory_validation.py），位于右上角
C. 转换动力学分析（基于φ分组），位于右下角靠左
D. 转换动力学分析（基于r_mainstream分组），位于右下角靠右
E. 极化度分析（基于r_mainstream分组），位于左下角
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import json
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置高质量图形参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 32  # 基础字体大小
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

class Figure5Generator:
    """Figure 5生成器：微观层面分析"""
    
    def __init__(self, data_root_path: str = "micro_analysis", samples_subdir: str = "samples_5"):
        """
        初始化Figure 5生成器
        
        Args:
            data_root_path: 数据根目录路径
            samples_subdir: 样本数据子目录名
        """
        self.data_root = Path(data_root_path)
        self.samples_subdir = samples_subdir
        self.steady_data = []
        self.steady_df = None
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载所有参数扫描数据"""
        print("📊 正在加载微观分析数据...")
        
        # 查找所有参数组合目录
        pattern = self.data_root / self.samples_subdir / "kappa*_phi*_theta*"
        param_dirs = glob.glob(str(pattern))
        
        print(f"🔍 找到 {len(param_dirs)} 个参数组合目录")
        
        success_count = 0
        for param_dir in param_dirs:
            param_path = Path(param_dir)
            dir_name = param_path.name
            
            # 解析参数值
            try:
                parts = dir_name.split('_')
                kappa = float(parts[0].replace('kappa', '')) / 100  # 转换为小数
                phi = float(parts[1].replace('phi', '')) / 1000    # 转换为小数
                theta = float(parts[2].replace('theta', '')) / 1000 # 转换为小数
                
                # 加载稳态数据
                steady_file = param_path / "steady_states.json"
                
                if steady_file.exists():
                    with open(steady_file, 'r', encoding='utf-8') as f:
                        steady_data = json.load(f)
                    
                    # 处理每个记录
                    for record in steady_data:
                        if record.get('success', False) and record.get('converged', False):
                            processed_record = {
                                'kappa': kappa,
                                'phi': phi, 
                                'theta': theta,
                                'param_dir': dir_name,
                                'sample_idx': record.get('sample_idx', 0),
                                'seed': record.get('seed', 0),
                                'r_mainstream': record.get('r_mainstream', 0),
                                
                                # 基础极化状态变量
                                'X_H': record.get('X_H'),
                                'X_M': record.get('X_M'),
                                'X_L': record.get('X_L'),
                                
                                # 理论验证数据
                                'theory_vs_actual_X_H_diff': record.get('theory_vs_actual_X_H_diff'),
                                'theory_vs_actual_X_L_diff': record.get('theory_vs_actual_X_L_diff'),
                                
                                # 转换动力学数据
                                'transition_rate_high_to_low': record.get('transition_rate_high_to_low'),
                                'transition_rate_low_to_high': record.get('transition_rate_low_to_high'),
                                'transition_rate_medium_to_high': record.get('transition_rate_medium_to_high'),
                                'transition_rate_medium_to_low': record.get('transition_rate_medium_to_low'),
                                'transition_rate_high_to_medium': record.get('transition_rate_high_to_medium'),
                                'transition_rate_low_to_medium': record.get('transition_rate_low_to_medium'),
                                
                                # 连接类型数据
                                'mainstream_connected_X_H': record.get('mainstream_connected_X_H'),
                                'mainstream_connected_X_L': record.get('mainstream_connected_X_L'),
                                'wemedia_connected_X_H': record.get('wemedia_connected_X_H'),
                                'wemedia_connected_X_L': record.get('wemedia_connected_X_L'),
                                'mixed_connected_X_H': record.get('mixed_connected_X_H'),
                                'mixed_connected_X_L': record.get('mixed_connected_X_L'),
                                
                                # 收敛特性
                                'iterations': record.get('iterations', 0),
                                'converged': record.get('converged', False),
                            }
                            
                            # 计算极化度
                            if (processed_record['mainstream_connected_X_H'] is not None and 
                                processed_record['mainstream_connected_X_L'] is not None):
                                processed_record['mainstream_polarization'] = (
                                    processed_record['mainstream_connected_X_H'] + 
                                    processed_record['mainstream_connected_X_L']
                                )
                            else:
                                processed_record['mainstream_polarization'] = np.nan
                                
                            if (processed_record['wemedia_connected_X_H'] is not None and 
                                processed_record['wemedia_connected_X_L'] is not None):
                                processed_record['wemedia_polarization'] = (
                                    processed_record['wemedia_connected_X_H'] + 
                                    processed_record['wemedia_connected_X_L']
                                )
                            else:
                                processed_record['wemedia_polarization'] = np.nan
                            
                            self.steady_data.append(processed_record)
                    
                    success_count += 1
                    
            except Exception as e:
                print(f"⚠️ 解析目录 {dir_name} 时出错: {e}")
                continue
        
        # 转换为DataFrame
        if self.steady_data:
            self.steady_df = pd.DataFrame(self.steady_data)
            print(f"✅ 成功加载 {success_count} 个参数组合")
            print(f"📈 总记录数: {len(self.steady_df)}")
            print(f"🔧 φ值范围: {self.steady_df['phi'].min():.3f} - {self.steady_df['phi'].max():.3f}")
            print(f"🔧 θ值范围: {self.steady_df['theta'].min():.3f} - {self.steady_df['theta'].max():.3f}")
            
            # 调试信息：显示几个样本记录的实际内容
            print(f"🔍 实际字段列表: {list(self.steady_df.columns)}")
            
            # 检查理论验证字段是否为空
            theory_h_null = self.steady_df['theory_vs_actual_X_H_diff'].isna().sum()
            theory_l_null = self.steady_df['theory_vs_actual_X_L_diff'].isna().sum()
            print(f"🧪 theory_vs_actual_X_H_diff 空值数量: {theory_h_null}/{len(self.steady_df)}")
            print(f"🧪 theory_vs_actual_X_L_diff 空值数量: {theory_l_null}/{len(self.steady_df)}")
            
            # 打印前几个记录的理论验证字段值
            print("🔍 前5个记录的理论验证字段值:")
            for i in range(min(5, len(self.steady_df))):
                record = self.steady_df.iloc[i]
                print(f"  记录 {i}: theory_vs_actual_X_H_diff = {record['theory_vs_actual_X_H_diff']}, "
                      f"theory_vs_actual_X_L_diff = {record['theory_vs_actual_X_L_diff']}")
            
            # 检查其他字段是否有值
            print("🔍 检查其他关键字段是否有值:")
            key_fields = ['X_H', 'X_M', 'X_L', 'mainstream_connected_X_H', 'mainstream_connected_X_L']
            for field in key_fields:
                if field in self.steady_df.columns:
                    non_null_count = self.steady_df[field].notna().sum()
                    print(f"  {field}: {non_null_count}/{len(self.steady_df)} 非空值")
                else:
                    print(f"  {field}: 字段不存在")
        else:
            print("⚠️ 警告：未加载到有效数据")
    
    def _create_subplot_b_theory_validation(self, ax1, ax2):
        """创建B图：理论验证误差分析 - 位于右上角，左侧箱型图+右侧RMSE山脊图"""
        if self.steady_df is None or len(self.steady_df) == 0:
            ax1.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=24)
            ax2.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=24)
            return
        
        print("🔬 生成B图：理论验证误差分析...")
        
        # === 左侧：箱型图 + 散点图 ===
        valid_data = self.steady_df.dropna(subset=['theory_vs_actual_X_H_diff', 'theory_vs_actual_X_L_diff'])
        phi_values = sorted(valid_data['phi'].unique())
        
        # 准备箱型图数据
        X_H_data_by_phi = []
        X_L_data_by_phi = []
        valid_phi_values = []  # 只保留有数据的phi值
        
        for phi in phi_values:
            phi_data = valid_data[valid_data['phi'] == phi]
            if len(phi_data) > 0:  # 只添加有数据的phi值
                X_H_data_by_phi.append(phi_data['theory_vs_actual_X_H_diff'].values)
                X_L_data_by_phi.append(phi_data['theory_vs_actual_X_L_diff'].values)
                valid_phi_values.append(phi)
        
        # 调试信息
        print(f"Debug - phi_values length: {len(phi_values)}")
        print(f"Debug - valid_phi_values length: {len(valid_phi_values)}")
        print(f"Debug - X_H_data_by_phi length: {len(X_H_data_by_phi)}")
        print(f"Debug - X_L_data_by_phi length: {len(X_L_data_by_phi)}")
        
        if len(X_H_data_by_phi) == 0:
            ax1.text(0.5, 0.5, 'No Valid Data for Box Plot', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=16, color='red')
            return
        
        # 绘制X_H和X_L的组合箱型图 - 使用有效数据
        positions_H = np.arange(len(valid_phi_values)) - 0.2
        positions_L = np.arange(len(valid_phi_values)) + 0.2
        
        # 先绘制散点图（背景层）
        for i, phi_data in enumerate(X_H_data_by_phi):
            if len(phi_data) > 50:
                sample_indices = np.random.choice(len(phi_data), 50, replace=False)
                sampled_data = phi_data[sample_indices]
            else:
                sampled_data = phi_data
            
            x_jitter = np.random.normal(positions_H[i], 0.05, len(sampled_data))
            ax1.scatter(x_jitter, sampled_data, alpha=0.3, s=8, color='#E74C3C', 
                       edgecolors='#C0392B', linewidth=0.2, zorder=1)
        
        for i, phi_data in enumerate(X_L_data_by_phi):
            if len(phi_data) > 50:
                sample_indices = np.random.choice(len(phi_data), 50, replace=False)
                sampled_data = phi_data[sample_indices]
            else:
                sampled_data = phi_data
            
            x_jitter = np.random.normal(positions_L[i], 0.05, len(sampled_data))
            ax1.scatter(x_jitter, sampled_data, alpha=0.3, s=8, color='#3498DB', 
                       edgecolors='#2980B9', linewidth=0.2, zorder=1)
        
        # X_H箱型图
        bp1 = ax1.boxplot(X_H_data_by_phi, positions=positions_H, 
                         patch_artist=True, widths=0.3, showfliers=False,
                         whis=1.5, zorder=2)
        
        # X_L箱型图
        bp2 = ax1.boxplot(X_L_data_by_phi, positions=positions_L, 
                         patch_artist=True, widths=0.3, showfliers=False,
                         whis=1.5, zorder=2)
        
        # 美化箱型图
        for patch in bp1['boxes']:
            patch.set_facecolor('#E74C3C')
            patch.set_alpha(0.7)
            patch.set_edgecolor('#2C3E50')
            patch.set_linewidth(2)
        
        for patch in bp2['boxes']:
            patch.set_facecolor('#3498DB')
            patch.set_alpha(0.7)
            patch.set_edgecolor('#2C3E50')
            patch.set_linewidth(2)
        
        # 统一样式
        for bp in [bp1, bp2]:
            for whisker in bp['whiskers']:
                whisker.set_color('#2C3E50')
                whisker.set_linewidth(2)
            for cap in bp['caps']:
                cap.set_color('#2C3E50')
                cap.set_linewidth(2)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            # 调整离群值样式：更小、更透明
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markersize(3)
                flier.set_alpha(0.4)
                flier.set_markeredgewidth(0.5)
        
        # 设置左侧图轴标签
        ax1.set_xlabel(r'$\phi$', fontsize=18, fontweight='bold')
        ax1.set_ylabel('MF vs Simulation Difference', fontsize=18, fontweight='bold')
        # 🔧 删除子图标题 - 节省空间
        # ax1.set_title('Box Plot Analysis', fontsize=24, fontweight='bold', pad=15)
        
        # 设置x轴刻度
        step = max(1, len(valid_phi_values) // 6)
        ax1.set_xticks(range(len(valid_phi_values)))
        ax1.set_xticklabels([f'{phi:.3f}' if i % step == 0 else '' for i, phi in enumerate(valid_phi_values)], 
                           rotation=45, ha='right', fontsize=14)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', alpha=0.7, label=r'$X_H$ Error'),
            Patch(facecolor='#3498DB', alpha=0.7, label=r'$X_L$ Error')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
        # 美化
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        
        # === 右侧：RMSE山脊图 ===
        # 按(φ, θ)分组计算RMSE
        phi_theta_groups = self.steady_df.groupby(['phi', 'theta'])
        phi_rmse_data = {}
        
        for (phi, theta), group in phi_theta_groups:
            valid_group = group.dropna(subset=['theory_vs_actual_X_H_diff', 'theory_vs_actual_X_L_diff'])
            
            if len(valid_group) > 0:
                # 计算该(φ, θ)组合下的组合RMSE
                X_H_errors = valid_group['theory_vs_actual_X_H_diff'].values
                X_L_errors = valid_group['theory_vs_actual_X_L_diff'].values
                all_errors = np.concatenate([X_H_errors, X_L_errors])
                rmse_combined = np.sqrt(np.mean(all_errors**2))
                
                # 按φ分组收集RMSE
                if phi not in phi_rmse_data:
                    phi_rmse_data[phi] = []
                phi_rmse_data[phi].append(rmse_combined)
        
        # 过滤有效的φ数据
        filtered_phi_data = {}
        for phi in sorted(phi_rmse_data.keys()):
            if len(phi_rmse_data[phi]) >= 3:  # 至少3个θ的RMSE
                filtered_phi_data[phi] = phi_rmse_data[phi]
        
        # 进一步筛选φ值，避免过于密集
        phi_keys = list(filtered_phi_data.keys())
        if len(phi_keys) > 12:  # 如果φ值太多，进行抽样
            step_ridge = max(2, len(phi_keys) // 10)
            selected_indices = list(range(0, len(phi_keys), step_ridge))
            if selected_indices[-1] != len(phi_keys) - 1:
                selected_indices.append(len(phi_keys) - 1)
            
            final_phi_data = {}
            for idx in selected_indices:
                phi = phi_keys[idx]
                final_phi_data[phi] = filtered_phi_data[phi]
            filtered_phi_data = final_phi_data
        
        if len(filtered_phi_data) == 0:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor RMSE ridge plot', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        else:
            # 准备山脊图数据
            phi_list = list(filtered_phi_data.keys())
            all_rmse_values = []
            for rmse_list in filtered_phi_data.values():
                all_rmse_values.extend(rmse_list)
            
            rmse_min, rmse_max = min(all_rmse_values), max(all_rmse_values)
            rmse_range = np.linspace(rmse_min, rmse_max, 100)
            
            # 计算所有密度值和最大密度
            max_density = 0
            all_densities = []
            
            for phi in phi_list:
                rmse_values = np.array(filtered_phi_data[phi])
                
                if len(rmse_values) > 1:
                    kde = gaussian_kde(rmse_values)
                    density = kde(rmse_range)
                    all_densities.append(density)
                    max_density = max(max_density, density.max())
                else:
                    all_densities.append(None)
            
            # 绘制堆叠的山脊图
            base_line_x = 0.1
            max_width = 0.8
            
            # 使用单色映射渐变
            base_color = (0.2, 0.4, 0.8)  # 深蓝色
            phi_normalized = [(phi - min(phi_list)) / (max(phi_list) - min(phi_list)) if len(phi_list) > 1 else 0.5 for phi in phi_list]
            
            for i, phi in enumerate(phi_list):
                if all_densities[i] is not None:
                    density = all_densities[i]
                    normalized_density = density / max_density * max_width
                    
                    # 颜色深度
                    color_intensity = phi_normalized[i]
                    lightness = 0.3 + 0.7 * (1 - color_intensity)
                    
                    fill_color = (
                        base_color[0] * lightness + (1 - lightness),
                        base_color[1] * lightness + (1 - lightness),
                        base_color[2] * lightness + (1 - lightness)
                    )
                    
                    fill_alpha = 0.3 + 0.4 * color_intensity
                    
                    # 绘制山脊图
                    ax2.fill_betweenx(rmse_range, 
                                     base_line_x,
                                     base_line_x + normalized_density,
                                     alpha=fill_alpha, color=fill_color, linewidth=0)
            
            # 绘制基线
            ax2.plot([base_line_x] * len(rmse_range), rmse_range, 
                    color='black', linewidth=1.5, alpha=0.8)
            
            # 设置山脊图属性
            ax2.set_xlim(0, 1.2)
            ax2.set_ylim(rmse_min, rmse_max)
            ax2.set_xticks([base_line_x, base_line_x + max_width/2, base_line_x + max_width])
            ax2.set_xticklabels(['Base', 'Density', 'Max'], fontsize=14)
            
        ax2.set_ylabel('RMSE Value', fontsize=18, fontweight='bold')
        # 🔧 删除子图标题 - 节省空间
        # ax2.set_title('RMSE Ridge Plot', fontsize=24, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        
    def _create_subplot_c_transition_dynamics(self, ax):
        """创建C图：转换动力学分析（基于test_optimized_transition_dynamics.py）- 位于右下角靠左"""
        if self.steady_df is None or len(self.steady_df) == 0:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=24)
            return
        
        print("🔄 生成C图：转换动力学分析...")
        
        # 获取φ的唯一值并排序
        phi_values = sorted(self.steady_df['phi'].unique())
        
        # 准备转换率强度数据（改进版）
        transition_intensity_data_by_phi = []
        
        for phi in phi_values:
            phi_data = self.steady_df[self.steady_df['phi'] == phi]
            
            # 按θ分组，计算每个θ的平均转换率强度
            theta_groups = phi_data.groupby('theta')
            transition_intensities = []
            
            for theta, theta_group in theta_groups:
                # 计算该θ组合下的转换率强度
                valid_group = theta_group.dropna(subset=['transition_rate_high_to_low', 'transition_rate_low_to_high'])
                
                if len(valid_group) > 0:
                    # 计算平均转换率（范围0-100）
                    avg_high_to_low = valid_group['transition_rate_high_to_low'].mean()
                    avg_low_to_high = valid_group['transition_rate_low_to_high'].mean()
                    
                    # 强制归一化到0-100范围（百分比形式）
                    avg_high_to_low = np.clip(avg_high_to_low, 0, 100)
                    avg_low_to_high = np.clip(avg_low_to_high, 0, 100)
                    avg_transition_rate = (avg_high_to_low + avg_low_to_high) / 2
                    transition_intensities.append(avg_transition_rate)
            
            transition_intensity_data_by_phi.append(transition_intensities)
        
        # 过滤空数据
        valid_phi_indices = [i for i, data in enumerate(transition_intensity_data_by_phi) if len(data) > 0]
        valid_phi_values = [phi_values[i] for i in valid_phi_indices]
        valid_intensity_data = [transition_intensity_data_by_phi[i] for i in valid_phi_indices]
        
        if len(valid_intensity_data) > 0:
            # 绘制散点图（背景层）
            for i, phi_data in enumerate(valid_intensity_data):
                if len(phi_data) > 50:
                    sample_indices = np.random.choice(len(phi_data), 50, replace=False)
                    sampled_data = [phi_data[j] for j in sample_indices]
                else:
                    sampled_data = phi_data
                
                x_jitter = np.random.normal(i, 0.12, len(sampled_data))
                ax.scatter(x_jitter, sampled_data, alpha=0.3, s=12, 
                          color='#E74C3C', edgecolors='#C0392B', linewidth=0.3, zorder=1)
            
            # 绘制箱型图（前景层）
            bp = ax.boxplot(valid_intensity_data, positions=range(len(valid_phi_values)), 
                           patch_artist=True, widths=0.4, showfliers=False, zorder=2,
                           whis=1.5)
            
            # 美化箱型图
            for patch in bp['boxes']:
                patch.set_facecolor('#E74C3C')
                patch.set_alpha(0.7)
                patch.set_edgecolor('#2C3E50')
                patch.set_linewidth(2)
            
            for whisker in bp['whiskers']:
                whisker.set_color('#2C3E50')
                whisker.set_linewidth(2)
            for cap in bp['caps']:
                cap.set_color('#2C3E50')
                cap.set_linewidth(2)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
                median.set_linestyle('--')
            # 调整离群值样式：更小、更透明
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markersize(3)
                flier.set_alpha(0.4)
                flier.set_markeredgewidth(0.5)
            
            # 设置轴标签和标题
            ax.set_xlabel(r'$\phi$', fontsize=20, fontweight='bold')
            ax.set_ylabel('Direct Transition Rate', fontsize=20, fontweight='bold')
            # 🔧 删除子图标题 - 节省空间
            # ax.set_title('Transition Dynamics Analysis', fontsize=32, fontweight='bold', pad=20)
            
            # 设置x轴刻度
            step = max(1, len(valid_phi_values) // 8)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_xticks(range(len(valid_phi_values)))
            ax.set_xticklabels([f'{phi:.3f}' if i % step == 0 else '' 
                               for i, phi in enumerate(valid_phi_values)], 
                               rotation=45, ha='right', fontsize=12)
            
            # 美化
            ax.grid(True, alpha=0.3)

        
    def _create_subplot_a_3d_bubble_polarization(self, ax):
        """创建A图：3D气泡极化分析图，使用1-X_M作为特征值 - 位于左侧"""
        if self.steady_df is None or len(self.steady_df) == 0:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=24)
            return
        
        print("🎯 生成A图：3D气泡极化分析（使用1-X_M作为特征值）...")
        
        # 检查必需的字段是否存在
        required_fields = ['r_mainstream', 'X_M']
        missing_fields = [field for field in required_fields if field not in self.steady_df.columns]
        
        if missing_fields:
            available_fields = list(self.steady_df.columns)
            error_msg = f"Missing required fields: {missing_fields}\nAvailable fields: {available_fields}"
            print(f"❌ Error: {error_msg}")
            ax.text(0.5, 0.5, f'Missing Fields:\n{missing_fields}\n\nCheck data structure', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
            return
        
        # 准备数据 - 使用1-X_M作为特征值
        valid_data = self.steady_df.dropna(subset=['r_mainstream', 'X_M'])
        
        if len(valid_data) == 0:
            ax.text(0.5, 0.5, 'No Valid Data for 1-X_M', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16, color='red')
            return
        
        # 计算1-X_M作为极化特征
        valid_data = valid_data.copy()
        valid_data['polarization_feature'] = 1 - valid_data['X_M']
        
        # 智能分层抽样
        sample_threshold = 300
        max_samples = 250
        high_samples, mid_samples, low_samples = 100, 50, 100
        
        if len(valid_data) > sample_threshold:
            feature_quartiles = valid_data['polarization_feature'].quantile([0.25, 0.5, 0.75])
            
            high_value_data = valid_data[valid_data['polarization_feature'] >= feature_quartiles.iloc[2]]
            mid_value_data = valid_data[(valid_data['polarization_feature'] >= feature_quartiles.iloc[0]) & 
                                       (valid_data['polarization_feature'] < feature_quartiles.iloc[2])]
            low_value_data = valid_data[valid_data['polarization_feature'] < feature_quartiles.iloc[0]]
            
            # 抽样比例
            high_ratio = min(0.4, high_samples / len(high_value_data)) if len(high_value_data) > 0 else 0
            mid_ratio = min(0.1, mid_samples / len(mid_value_data)) if len(mid_value_data) > 0 else 0
            low_ratio = min(0.4, low_samples / len(low_value_data)) if len(low_value_data) > 0 else 0
            
            sampled_parts = []
            if len(high_value_data) > 0:
                sampled_parts.append(high_value_data.sample(frac=high_ratio, random_state=42))
            if len(mid_value_data) > 0:
                sampled_parts.append(mid_value_data.sample(frac=mid_ratio, random_state=42))
            if len(low_value_data) > 0:
                sampled_parts.append(low_value_data.sample(frac=low_ratio, random_state=42))
            
            if sampled_parts:
                sampled_data = pd.concat(sampled_parts, ignore_index=True)
            else:
                sampled_data = valid_data.sample(n=min(max_samples, len(valid_data)), random_state=42)
        else:
            sampled_data = valid_data
        
        # 准备3D数据
        x = sampled_data['phi'].values
        y = sampled_data['theta'].values
        z = sampled_data['r_mainstream'].values
        colors = sampled_data['polarization_feature'].values
        
        # 气泡大小基于极化特征
        sizes = 20 + 120 * (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)
        
        # 动态透明度
        if len(sizes) > 0:
            norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10)
            alphas = 0.5 + 0.4 * norm_sizes
        else:
            alphas = 0.7
        
        # 绘制3D散点图
        scatter = ax.scatter(x, y, z, s=sizes, c=colors, cmap='RdYlBu_r', 
                           alpha=alphas, edgecolors='white', linewidth=0.3)
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1, aspect=30)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Polarization', fontsize=12, fontweight='bold')
        
        # 设置坐标轴标签
        ax.set_xlabel(r'$\phi$', fontsize=14, fontweight='bold', labelpad=4)
        ax.set_ylabel(r'$\theta$', fontsize=14, fontweight='bold', labelpad=4)
        ax.set_zlabel(r'$r$', fontsize=14, fontweight='bold', labelpad=4)
        
        # 设置坐标轴范围 - 扩大显示范围，减少空白
        phi_range = self.steady_df['phi'].quantile([0.02, 0.98])  # 从 0.05-0.95 改为 0.02-0.98
        theta_range = self.steady_df['theta'].quantile([0.02, 0.98])
        r_range = self.steady_df['r_mainstream'].quantile([0.02, 0.98])
        
        ax.set_xlim(phi_range.iloc[0], phi_range.iloc[1])
        ax.set_ylim(theta_range.iloc[0], theta_range.iloc[1])
        ax.set_zlim(r_range.iloc[0], r_range.iloc[1])
        
        # 优化刻度标签
        ax.tick_params(axis='x', labelsize=9, pad=1)
        ax.tick_params(axis='y', labelsize=9, pad=1)
        ax.tick_params(axis='z', labelsize=9, pad=1)
        
        # 设置视角 - 调整视角让气泡图更突出
        ax.view_init(elev=25, azim=45)
        
        # 添加网格
        ax.grid(True, alpha=0.25)
        
        # 美化3D坐标轴背景 - 减少背景透明度
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.05)  # 从 0.1 减少到 0.05
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
        
        print(f"- 数据点数: {len(sampled_data)}")
        print(f"- 特征值(1-X_M)范围: {colors.min():.4f} - {colors.max():.4f}")
        print(f"- 平均极化值: {colors.mean():.4f}")
        
    def _create_subplot_d_transition_dynamics_by_r(self, ax):
        """创建D图：基于r_mainstream分组的转换动力学分析 - 位于右下角靠右"""
        if self.steady_df is None or len(self.steady_df) == 0:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=24)
            return
        
        print("🔄 生成D图：基于r_mainstream分组的转换动力学分析...")
        
        # 获取r_mainstream的唯一值并排序
        r_values = sorted(self.steady_df['r_mainstream'].unique())
        
        # 准备转换率强度数据（基于r_mainstream分组）
        transition_intensity_data_by_r = []
        
        for r in r_values:
            r_data = self.steady_df[self.steady_df['r_mainstream'] == r]
            
            # 按φ分组，计算每个φ的平均转换率强度
            phi_groups = r_data.groupby('phi')
            transition_intensities = []
            
            for phi, phi_group in phi_groups:
                # 计算该φ组合下的转换率强度
                valid_group = phi_group.dropna(subset=['transition_rate_high_to_low', 'transition_rate_low_to_high'])
                
                if len(valid_group) > 0:
                    # 计算平均转换率（范围0-100）
                    avg_high_to_low = valid_group['transition_rate_high_to_low'].mean()
                    avg_low_to_high = valid_group['transition_rate_low_to_high'].mean()
                    
                    # 强制归一化到0-100范围（百分比形式）
                    avg_high_to_low = np.clip(avg_high_to_low, 0, 100)
                    avg_low_to_high = np.clip(avg_low_to_high, 0, 100)
                    avg_transition_rate = (avg_high_to_low + avg_low_to_high) / 2
                    transition_intensities.append(avg_transition_rate)
            
            transition_intensity_data_by_r.append(transition_intensities)
        
        # 过滤空数据
        valid_r_indices = [i for i, data in enumerate(transition_intensity_data_by_r) if len(data) > 0]
        valid_r_values = [r_values[i] for i in valid_r_indices]
        valid_intensity_data = [transition_intensity_data_by_r[i] for i in valid_r_indices]
        
        if len(valid_intensity_data) > 0:
            # 绘制散点图（背景层）
            for i, r_data in enumerate(valid_intensity_data):
                if len(r_data) > 50:
                    sample_indices = np.random.choice(len(r_data), 50, replace=False)
                    sampled_data = [r_data[j] for j in sample_indices]
                else:
                    sampled_data = r_data
                
                x_jitter = np.random.normal(i, 0.12, len(sampled_data))
                ax.scatter(x_jitter, sampled_data, alpha=0.3, s=12, 
                          color='#27AE60', edgecolors='#229954', linewidth=0.3, zorder=1)
            
            # 绘制箱型图（前景层）
            bp = ax.boxplot(valid_intensity_data, positions=range(len(valid_r_values)), 
                           patch_artist=True, widths=0.4, showfliers=False, zorder=2,
                           whis=1.5)
            
            # 美化箱型图
            for patch in bp['boxes']:
                patch.set_facecolor('#27AE60')
                patch.set_alpha(0.7)
                patch.set_edgecolor('#2C3E50')
                patch.set_linewidth(2)
            
            for whisker in bp['whiskers']:
                whisker.set_color('#2C3E50')
                whisker.set_linewidth(2)
            for cap in bp['caps']:
                cap.set_color('#2C3E50')
                cap.set_linewidth(2)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
                median.set_linestyle('--')
            # 调整离群值样式：更小、更透明
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markersize(3)
                flier.set_alpha(0.4)
                flier.set_markeredgewidth(0.5)
            
            # 设置轴标签和标题
            ax.set_xlabel(r'$r$', fontsize=20, fontweight='bold')
            ax.set_ylabel('Direct Transition Rate', fontsize=20, fontweight='bold')
            
            # 设置x轴刻度
            step = max(1, len(valid_r_values) // 8)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_xticks(range(len(valid_r_values)))
            ax.set_xticklabels([f'{r:.3f}' if i % step == 0 else '' 
                               for i, r in enumerate(valid_r_values)], 
                               rotation=45, ha='right', fontsize=12)
            
            # 美化
            ax.grid(True, alpha=0.3)
        
    def _create_subplot_e_polarization_analysis(self, ax):
        """创建E图：极化度分析（基于r_mainstream分组）- 位于左下角"""
        if self.steady_df is None or len(self.steady_df) == 0:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=24)
            return
        
        print("🎯 生成E图：极化度分析...")
        
        # 准备数据
        plot_data = self._prepare_polarization_data('r_mainstream')
        r_values = sorted(self.steady_df['r_mainstream'].unique())
        
        # 数据准备
        if not any('X_H_data' in plot_data[r] for r in r_values):
            ax.text(0.5, 0.5, 'No Valid Data for Polarization Analysis', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16, color='red')
            return
        
        # 准备 X_H 数据
        xh_data = [plot_data[r].get('X_H_data', []) for r in r_values]
        valid_xh_data = [data for data in xh_data if len(data) > 0]
        xh_labels = [f'{r_val:.2f}' for r_val in r_values if len(plot_data[r_val].get('X_H_data', [])) > 0]
        
        # 准备 X_L 数据用于计算
        xl_data = [plot_data[r].get('X_L_data', []) for r in r_values]
        valid_xl_data = [data for data in xl_data if len(data) > 0]
        
        # 计算并准备 Polarization (X_H + X_L) 数据
        combined_data = []
        if valid_xh_data and valid_xl_data:
            num_plots = min(len(valid_xh_data), len(valid_xl_data))
            for i in range(num_plots):
                xh_array = np.array(valid_xh_data[i])
                xl_array = np.array(valid_xl_data[i])
                combined = xh_array + xl_array
                combined_data.append(combined)
        
        if not combined_data:
            ax.text(0.5, 0.5, 'Failed to calculate polarization data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16, color='red')
            return
        
        # 开始绘图
        positions = np.arange(len(xh_labels))
        box_width = 0.35
        
        # 绘制函数：用于绘制一组箱型图和散点图
        def draw_boxplot_group(ax, data, positions, width, color, jitter_color):
            bp = ax.boxplot(data, positions=positions, widths=width,
                            patch_artist=True, showfliers=False, zorder=2,
                            whiskerprops=dict(color='black', linewidth=1.2),
                            capprops=dict(color='black', linewidth=1.2),
                            medianprops=dict(color='black', linewidth=1.5))
            
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            for i, d in enumerate(data):
                x_jitter = np.random.normal(positions[i], 0.04, size=len(d))
                ax.scatter(x_jitter, d, alpha=0.5, s=15, color=jitter_color, edgecolors='white', linewidth=0.5, zorder=1)
        
        # 绘制 X_H 分布 (左侧)
        draw_boxplot_group(ax, valid_xh_data, positions - box_width/2, box_width, '#E74C3C', '#C0392B')
        
        # 绘制 Polarization 分布 (右侧)
        draw_boxplot_group(ax, combined_data, positions + box_width/2, box_width, '#9B59B6', '#8E44AD')
        
        # 设置图表样式
        ax.set_xlabel(r'$r$', fontsize=16, fontweight='bold')
        ax.set_ylabel('Polarization Metric Value', fontsize=16, fontweight='bold')
        
        # 设置X轴刻度和标签
        ax.set_xticks(positions)
        display_labels = [label if i % 2 == 0 else '' for i, label in enumerate(xh_labels)]
        ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # 创建图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E74C3C', alpha=0.7, label=r'High Polarization ($X_H$)'),
            Patch(facecolor='#9B59B6', alpha=0.7, label=r'Total Polarization ($X_H + X_L$)')
        ]
        ax.legend(handles=legend_elements, loc='center right', fontsize=8)
        
        # 美化
        ax.grid(True, alpha=0.3)
        
    def _prepare_polarization_data(self, group_by_column):
        """准备极化度数据"""
        plot_data = {}
        
        # 为每个分组准备数据
        for group_val in self.steady_df[group_by_column].unique():
            group_df = self.steady_df[self.steady_df[group_by_column] == group_val]
            plot_data[group_val] = {}
            
            # 原始状态数据
            if all(f in self.steady_df.columns for f in ['X_H', 'X_M', 'X_L']):
                plot_data[group_val].update({
                    'X_H_data': group_df['X_H'].values,
                    'X_M_data': group_df['X_M'].values,
                    'X_L_data': group_df['X_L'].values
                })
        
        return plot_data
        
    def generate_figure5(self, save_path: str = "Figures/Figure 5/Figure5_Micro_Analysis.png"):
        """
        生成Figure 5：微观层面分析
        
        Args:
            save_path: 保存路径
        """
        print("🎨 开始生成Figure 5：微观层面分析...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 布局说明：
        # A图: 3D气泡极化分析，位于左侧，占据(0,0)到(0,2)的位置
        # B图: 理论验证误差分析，位于右上角，由两个子图构成，位置(0,2)和(0,3)
        # C图: 转换动力学分析（基于φ分组），位于右下角靠左，位置(1,2)
        # D图: 转换动力学分析（基于r_mainstream分组），位于右下角靠右，位置(1,3)
        # E图: 极化度分析（基于r_mainstream分组），位于左下角，位置(1,0)到(1,1)
        
        # A图: 3D气泡极化分析，占据左侧一半空间，使其更突出
        ax_a = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=1, projection='3d')

        # B图: 理论验证误差分析，占据右上角，由两个子图构成
        ax_b1 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
        ax_b2 = plt.subplot2grid((2, 4), (0, 3), colspan=1)

        # C图: 转换动力学分析，占据右下角靠左位置
        ax_c = plt.subplot2grid((2, 4), (1, 2), colspan=1)

        # D图: 转换动力学分析（基于r_mainstream分组），占据右下角靠右位置
        ax_d = plt.subplot2grid((2, 4), (1, 3), colspan=1)
        
        # E图: 极化度分析（基于r_mainstream分组），占据左下角位置
        ax_e = plt.subplot2grid((2, 4), (1, 0), colspan=2)
        
        # 生成各个子图（按实际布局顺序）
        # A图：3D气泡极化分析（使用1-X_M作为特征）- 左侧
        self._create_subplot_a_3d_bubble_polarization(ax_a)
        
        # B图：理论验证误差分析 - 右上角
        self._create_subplot_b_theory_validation(ax_b1, ax_b2)
        
        # C图：转换动力学分析（基于φ分组）- 右下角靠左
        self._create_subplot_c_transition_dynamics(ax_c)
        
        # D图：转换动力学分析（基于r_mainstream分组）- 右下角靠右
        self._create_subplot_d_transition_dynamics_by_r(ax_d)
        
        # E图：极化度分析（基于r_mainstream分组）- 左下角
        self._create_subplot_e_polarization_analysis(ax_e)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.95,
                           wspace=0.3, hspace=0.3) # 调整了wspace和hspace
        
        # 保存图像前，确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"✅ Figure 5 已保存到: {save_path}")
        plt.show()
        
        # 恢复全局设置
        plt.rcParams['figure.dpi'] = plt.rcParamsDefault['figure.dpi']
        plt.rcParams['savefig.dpi'] = plt.rcParamsDefault['savefig.dpi']
        plt.rcParams['font.size'] = plt.rcParamsDefault['font.size']

def CREA():
    """主函数：生成Figure 5"""
    print("=== Figure 5 生成器：微观层面分析 ===")
    
    # 🔧 确保输出目录存在
    output_dir = "Figures/Figure 5"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 确保输出目录存在: {output_dir}")
    
    # 创建生成器
    generator = Figure5Generator()
    
    if generator.steady_df is None or len(generator.steady_df) == 0:
        print("⚠️ 未能加载有效数据，请检查数据路径")
        return
    
    # 生成Figure 5 - 使用完整路径
    save_path = os.path.join(output_dir, "Figure5_Micro_Analysis.png")
    generator.generate_figure5(save_path)
    
    print("✅ Figure 5 生成完成！")

if __name__ == "__main__":
    CREA()