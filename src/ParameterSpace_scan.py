import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.signal import medfilt
import time
import pickle
from functools import partial
import argparse
import math
from matplotlib import rcParams
from scipy.optimize import minimize
from scipy.stats import poisson
import sys
from itertools import product
import warnings
import csv
from glob import glob

# Import other modules from the project
from src.model_with_a_minimal_v3 import ThresholdDynamicsModel
from src.detection_utils import detect_jumps_improved
from src.model_v3_fixed import ThresholdDynamicsModelV3Fixed
from src.config import MAX_ITER, CONVERGENCE_TOL, BOUNDARY_FRACTION

# Set detailed logging output
import logging
logging.basicConfig(level=logging.INFO)


# Set English font
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



def solve_steady_state_for_r(model, r_m, threshold_params=None):
    """
    为单个r_m值求解稳态，用于并行计算
    
    参数:
        model: ThresholdDynamicsModel实例或参数字典
        r_m: 移除比例
        threshold_params: 阈值参数(可选，当model为参数字典时需提供)
        
    返回:
        包含稳态解的字典
    """
    # 如果传入的是参数字典而不是模型实例，则创建模型
    if not isinstance(model, ThresholdDynamicsModel):
        network_params = model
        model = ThresholdDynamicsModelV3Fixed(network_params, threshold_params)
    
    try:
        # 求解自洽方程
        result = model.solve_self_consistent(
            removal_ratios={'mainstream': r_m},
            max_iter=MAX_ITER,
            tol=CONVERGENCE_TOL
        )
        
        if not result['converged']:
            raise ValueError(f"稳态在r_m={r_m}处未收敛")
        
        # 返回结果(添加r_m值以便后续处理)
        result['r_m'] = r_m
        return result
    except Exception as e:
        # 不返回默认值，直接抛出异常
        raise ValueError(f"在r_m={r_m}处求解稳态失败: {str(e)}")

def _solve_steady_state_wrapper(params):
    """用于多进程的包装函数"""
    net_params, r, thresh_params = params
    return solve_steady_state_for_r(net_params, r, thresh_params)

def compute_single_point_r_combination(args):
    """
    计算单个(phi, theta, r)组合的稳态，用于展平并行计算
    
    参数:
        args: 包含(phi, theta, r_m, network_params, base_threshold_params)的元组
        
    返回:
        包含计算结果的字典
    """
    phi, theta, r_m, network_params, base_threshold_params = args
    
    try:
        # 构建阈值参数
        if base_threshold_params is None:
            threshold_params = {
                'theta': theta,
                'phi': phi
            }
        else:
            threshold_params = base_threshold_params.copy()
            threshold_params['theta'] = theta
            threshold_params['phi'] = phi
        
        # 创建模型
        model = ThresholdDynamicsModelV3Fixed(network_params, threshold_params)
        
        # 求解自洽方程
        result = model.solve_self_consistent(
            removal_ratios={'mainstream': r_m},
            max_iter=MAX_ITER,
            tol=CONVERGENCE_TOL
        )
        
        if not result['converged']:
            raise ValueError(f"稳态在r_m={r_m}处未收敛")
        
        # 返回结果
        return {
            'phi': phi,
            'theta': theta,
            'r_m': r_m,
            'success': True,
            'X_H': result['X_H'],
            'X_M': result['X_M'],
            'X_L': result['X_L'],
            'p_risk': result['p_risk'],
            'p_risk_m': result['p_risk_m'],
            'p_risk_w': result['p_risk_w']
        }
        
    except Exception as e:
        return {
            'phi': phi,
            'theta': theta,
            'r_m': r_m,
            'success': False,
            'error': str(e)
        }

def compute_single_point_r_combination_with_kappa(args):
    """
    计算单个(phi, theta, kappa, r)组合的稳态，用于展平并行计算
    
    参数:
        args: 包含(phi, theta, r_m, kappa_network_params, base_threshold_params, kappa, init_states)的元组
        
    返回:
        包含计算结果的字典
    """
    # 兼容旧格式（不包含init_states）和新格式（包含init_states）
    if len(args) == 6:
        # 旧格式：没有init_states
        phi, theta, r_m, kappa_network_params, base_threshold_params, kappa = args
        init_states = None
    elif len(args) == 7:
        # 新格式：包含init_states
        phi, theta, r_m, kappa_network_params, base_threshold_params, kappa, init_states = args
    else:
        raise ValueError(f"Invalid argument format: expected 6 or 7 arguments, got {len(args)}")
    
    try:
        # 构建阈值参数
        if base_threshold_params is None:
            threshold_params = {
                'theta': theta,
                'phi': phi
            }
        else:
            threshold_params = base_threshold_params.copy()
            threshold_params['theta'] = theta
            threshold_params['phi'] = phi
        
        # 创建模型
        model = ThresholdDynamicsModelV3Fixed(kappa_network_params, threshold_params)
        
        # 准备solve_self_consistent的参数
        solve_params = {
            'removal_ratios': {'mainstream': r_m},
            'max_iter': MAX_ITER,
            'tol': CONVERGENCE_TOL
        }
        
        # 如果提供了init_states，则传递给求解器
        if init_states is not None:
            solve_params['init_states'] = init_states
        
        # 求解自洽方程
        result = model.solve_self_consistent(**solve_params)
        
        if not result['converged']:
            raise ValueError(f"稳态在r_m={r_m}处未收敛")
        
        # 返回结果
        return {
            'phi': phi,
            'theta': theta,
            'kappa': kappa,
            'r_m': r_m,
            'success': True,
            'X_H': result['X_H'],
            'X_M': result['X_M'],
            'X_L': result['X_L'],
            'p_risk': result['p_risk'],
            'p_risk_m': result['p_risk_m'],
            'p_risk_w': result['p_risk_w']
        }
        
    except Exception as e:
        return {
            'phi': phi,
            'theta': theta,
            'kappa': kappa,
            'r_m': r_m,
            'success': False,
            'error': str(e)
        }

def process_parameter_point_flattened(phi, theta, r_values, network_params, 
                                    base_threshold_params, save_dir, 
                                    abs_jump_threshold, removal_type, 
                                    all_results_dict, init_states=None):
    """
    处理单个参数点，使用展平的预计算结果
    
    参数:
        phi, theta: 参数值
        r_values: r值数组
        network_params: 网络参数
        base_threshold_params: 阈值参数模板
        save_dir: 保存目录
        abs_jump_threshold: 跳变阈值
        removal_type: 移除类型
        all_results_dict: 预计算的所有结果字典 {(phi, theta, r): result}
        init_states: 初始状态字典，默认None
        
    返回:
        处理结果字典
    """
    try:
        # 计算kappa值
        kappa = network_params.get('k_out_mainstream', 60) + network_params.get('k_out_wemedia', 60)
        
        # 创建参数点对应的目录
        kappa_int = int(round(kappa))
        phi_int = int(round(phi * 100))
        theta_int = int(round(theta * 100))
        
        point_dir = os.path.join(save_dir, f"kappa{kappa_int:03d}_phi{phi_int:03d}_theta{theta_int:03d}")
        os.makedirs(point_dir, exist_ok=True)
        
        # 构建阈值参数
        if base_threshold_params is None:
            threshold_params = {
                'theta': theta,
                'phi': phi
            }
        else:
            threshold_params = base_threshold_params.copy()
            threshold_params['theta'] = theta
            threshold_params['phi'] = phi
        
        print(f"  1. Extracting steady states from precomputed results ({len(r_values)} points)...")
        
        # 从预计算结果中提取稳态值
        X_H_values = []
        X_M_values = []
        X_L_values = []
        p_risk_values = []
        p_risk_m_values = []
        p_risk_w_values = []
        
        failed_points = []
        
        # 🔧 修复：添加调试信息和容错机制
        print(f"  🔍 调试：检查all_results_dict键值匹配...")
        print(f"    目标参数：phi={phi}, theta={theta}, kappa={kappa}")
        print(f"    字典中总键数：{len(all_results_dict)}")
        
        # 显示字典中的前几个键作为样本
        sample_keys = list(all_results_dict.keys())[:3]
        for i, sample_key in enumerate(sample_keys):
            print(f"    样本键{i+1}: {sample_key}")
        
        matched_count = 0
        for r in r_values:
            # 🔧 修复浮点数精度问题：使用更强的容错匹配
            result = None
            
            # 第一步：尝试直接键匹配
            key = (phi, theta, kappa, r)
            if key in all_results_dict and all_results_dict[key]['success']:
                result = all_results_dict[key]
                matched_count += 1
            else:
                # 第二步：使用容错匹配寻找最佳匹配
                best_match = None
                best_r_diff = float('inf')
                
                for stored_key, stored_result in all_results_dict.items():
                    stored_phi, stored_theta, stored_kappa, stored_r = stored_key
                    if (abs(stored_phi - phi) < 1e-10 and 
                        abs(stored_theta - theta) < 1e-10 and 
                        abs(stored_kappa - kappa) < 1e-10 and 
                        stored_result['success']):
                        
                        r_diff = abs(stored_r - r)
                        if r_diff < best_r_diff:
                            best_r_diff = r_diff
                            best_match = stored_result
                
                # 如果找到了足够接近的匹配（差异小于1e-6）
                if best_match is not None and best_r_diff < 1e-6:
                    result = best_match
                    matched_count += 1
                else:
                    # 调试输出：显示最接近的键
                    closest_r_diff = float('inf')
                    closest_key = None
                    for stored_key in all_results_dict.keys():
                        stored_phi, stored_theta, stored_kappa, stored_r = stored_key
                        if (abs(stored_phi - phi) < 1e-10 and 
                            abs(stored_theta - theta) < 1e-10 and 
                            abs(stored_kappa - kappa) < 1e-10):
                            r_diff = abs(stored_r - r)
                            if r_diff < closest_r_diff:
                                closest_r_diff = r_diff
                                closest_key = stored_key
                    
                    print(f"    ❌ 未找到r={r:.6f}的匹配键")
                    if closest_key:
                        print(f"       最接近的键: {closest_key}, r差异={closest_r_diff:.2e}")
                    
                failed_points.append(r)
                # 使用NaN填充缺失值
                X_H_values.append(np.nan)
                X_M_values.append(np.nan)
                X_L_values.append(np.nan)
                p_risk_values.append(np.nan)
                p_risk_m_values.append(np.nan)
                p_risk_w_values.append(np.nan)
                continue
            
            # 成功找到匹配结果
            X_H_values.append(result['X_H'])
            X_M_values.append(result['X_M'])
            X_L_values.append(result['X_L'])
            p_risk_values.append(result['p_risk'])
            p_risk_m_values.append(result['p_risk_m'])
            p_risk_w_values.append(result['p_risk_w'])
        
        print(f"  📊 键值匹配结果：成功匹配 {matched_count}/{len(r_values)} 个r值")
        
        if failed_points:
            print(f"  ⚠️ Warning: {len(failed_points)} r values failed to find matches")
            print(f"      Failed r values: {failed_points[:5]}..." if len(failed_points) > 5 else f"      Failed r values: {failed_points}")
        else:
            print(f"  ✅ 所有r值都成功匹配到预计算结果")
        
        # 序参量跳变分析
        print(f"  2. Analyzing order parameter jumps...")
        
        # 移除NaN值进行跳变分析
        valid_mask = ~np.isnan(X_H_values)
        if np.sum(valid_mask) < 5:
            return {
                'status': 'failed',
                'phi': phi,
                'theta': theta,
                'kappa': kappa,
                'error': 'Insufficient valid data points for analysis'
            }
        
        valid_r_values = np.array(r_values)[valid_mask]
        valid_X_H = np.array(X_H_values)[valid_mask]
        
        # 使用改进的跳变检测
        jumps, r_jumps, jump_detected, z_score = detect_jumps_improved(valid_r_values, valid_X_H)
        
        # 提取跳变信息
        if jumps and r_jumps:
            max_jump_position = r_jumps[0]  # 第一个（最大的）跳变位置
            max_jump_size = jumps[0]        # 第一个（最大的）跳变大小
        else:
            max_jump_position = None
            max_jump_size = 0.0
        
        if jump_detected:
            print(f"  Significant jump detected: r_c = {max_jump_position:.4f}, jump size = {max_jump_size:.4f}, z-score = {z_score:.2f}")
        elif max_jump_position is not None:
            print(f"  Jump detected but not significant: r_c = {max_jump_position:.4f}, jump size = {max_jump_size:.4f}, z-score = {z_score:.2f}")
        else:
            print(f"  No jump detected")
        
        # 关联长度计算 - 物理上需要扰动分析，必须重新计算
        print(f"  3. Computing correlation length (requires perturbation analysis)...")
        
        # 创建模型实例
        model = ThresholdDynamicsModelV3Fixed(network_params, threshold_params)
        
        try:
            # 并行计算关联长度（物理上需要扰动分析，不能基于预计算稳态值）
            corr_result = model.calculate_correlation_length_robust(
                r_values, 
                removal_type=removal_type, 
                n_processes=max(1, mp.cpu_count() - 1),  # 自动选择合理的进程数
                external_threshold_params=threshold_params,
                state_name='X_H',
                enable_diagnostics=False,
                enable_power_law=True,
                init_states=init_states  # 传递初始状态
            )
            
            # 处理返回值
            if len(corr_result) >= 4:
                corr_lengths, critical_r, raw_data, power_law_results = corr_result
                print(f"   Successfully retrieved correlation length data")
            else:
                raise ValueError(f"Correlation length calculation returned abnormal result")
                
        except RuntimeError as e:
            if "计算结果长度不匹配" in str(e):
                print(f"   ⚠️ 部分r值计算失败，尝试使用容错处理...")
                # 过滤掉容易失败的高r值，重新计算
                r_filtered = r_values[r_values <= 0.9]  # 移除r>0.9的点
                if len(r_filtered) < 10:
                    # 如果过滤后点数太少，抛出原始异常
                    raise e
                
                print(f"   使用过滤后的r值范围: [{r_filtered[0]:.3f}, {r_filtered[-1]:.3f}] ({len(r_filtered)}个点)")
                
                corr_result = model.calculate_correlation_length_robust(
                    r_filtered, 
                    removal_type=removal_type, 
                    n_processes=max(1, mp.cpu_count() - 1),
                    external_threshold_params=threshold_params,
                    state_name='X_H',
                    enable_diagnostics=False,
                    enable_power_law=True,
                    init_states=init_states
                )
                
                if len(corr_result) >= 4:
                    corr_lengths_filtered, critical_r, raw_data, power_law_results = corr_result
                    print(f"   ✅ 容错计算成功，使用{len(r_filtered)}个有效点")
                    
                    # 扩展结果到原始r_values长度，缺失值用NaN填充
                    corr_lengths = np.full(len(r_values), np.nan)
                    for i, r in enumerate(r_values):
                        if r in r_filtered:
                            filtered_idx = np.where(r_filtered == r)[0][0]
                            corr_lengths[i] = corr_lengths_filtered[filtered_idx]
                    
                else:
                    raise ValueError(f"容错处理后仍然失败")
            else:
                # 其他类型的RuntimeError，直接抛出
                raise e
        
        # Quality assessment
        quality_info = assess_correlation_quality_v3(
            r_values, corr_lengths, critical_r, power_law_results
        )
        
        max_corr = np.max(corr_lengths)
        max_corr_idx = np.argmax(corr_lengths)
        max_corr_position = r_values[max_corr_idx]
        
        # 从幂律结果提取ν值
        nu_value = None
        if power_law_results and 'correlation_length_scaling' in power_law_results:
            scaling = power_law_results['correlation_length_scaling']
            if 'nu' in scaling:
                nu_value = scaling['nu']
        
        print(f"  Quality assessment: {quality_info['overall_score']}/{quality_info['max_score']} ({quality_info['quality_level']})")
        
        # 4. 确定相变类型
        transition_type = determine_transition_type_v3(
            jump_detected, quality_info, max_jump_position, critical_r,
            X_H_values=X_H_values, r_values=r_values, power_law_results=power_law_results
        )
        
        # 5. 保存结果
        point_result = {
            'phi': phi,
            'theta': theta,
            'r_c': critical_r if transition_type in ['second_order', 'possible_second_order'] else max_jump_position,
            'max_corr': max_corr,
            'max_corr_position': max_corr_position,
            'critical_r': critical_r,
            'nu': nu_value,
            'has_jump': jump_detected,
            'jump_position': max_jump_position if jump_detected else None,
            'jump_size': max_jump_size if jump_detected else 0,
            'transition_type': transition_type,
            'quality_info': quality_info,
            'correlation_lengths': corr_lengths,
            'r_values': r_values,
            'X_H_values': X_H_values,
            'X_M_values': X_M_values,
            'X_L_values': X_L_values,
            'p_risk_values': p_risk_values,
            'p_risk_m_values': p_risk_m_values,
            'p_risk_w_values': p_risk_w_values,
            'power_law_results': power_law_results,
            'failed_points': failed_points
        }
        
        # 保存结果
        result_file = os.path.join(point_dir, 'result.pkl')
        with open(result_file, 'wb') as f:
            pickle.dump(point_result, f)
        
        # 生成可视化
        generate_visualization_v3(point_result, point_dir, phi)
        
        # 修复格式化错误：添加对None值的处理
        r_c_value = point_result['r_c']
        r_c_str = f"{r_c_value:.4f}" if r_c_value is not None else "None"
        print(f"  ✅ Analysis completed: {transition_type}, r_c={r_c_str}")
        
        return {
            'status': 'success',
            'phi': phi,
            'theta': theta,
            'result': point_result
        }
        
    except Exception as e:
        print(f"  ❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'phi': phi,
            'theta': theta,
            'error': str(e)
        }

def process_parameter_point_flattened_with_kappa(phi, theta, kappa, r_values, kappa_network_params, 
                                                base_threshold_params, save_dir, 
                                                abs_jump_threshold, removal_type, 
                                                all_results_dict, init_states=None):
    """
    处理单个参数点，使用展平的预计算结果 - 支持kappa参数
    
    参数:
        phi, theta, kappa: 参数值
        r_values: r值数组
        kappa_network_params: 网络参数（已根据kappa调整）
        base_threshold_params: 阈值参数模板
        save_dir: 保存目录
        abs_jump_threshold: 跳变阈值
        removal_type: 移除类型
        all_results_dict: 预计算的所有结果字典 {(phi, theta, kappa, r): result}
        init_states: 初始状态字典，可选
        
    返回:
        处理结果字典
    """
    try:
        # 创建参数点对应的目录
        kappa_int = int(round(kappa))
        phi_int = int(round(phi * 100))
        theta_int = int(round(theta * 100))
        
        point_dir = os.path.join(save_dir, f"kappa{kappa_int:03d}_phi{phi_int:03d}_theta{theta_int:03d}")
        os.makedirs(point_dir, exist_ok=True)
        
        # 构建阈值参数
        if base_threshold_params is None:
            threshold_params = {
                'theta': theta,
                'phi': phi
            }
        else:
            threshold_params = base_threshold_params.copy()
            threshold_params['theta'] = theta
            threshold_params['phi'] = phi
        
        print(f"  1. Extracting steady states from precomputed results ({len(r_values)} points)...")
        
        # 从预计算结果中提取稳态值
        X_H_values = []
        X_M_values = []
        X_L_values = []
        p_risk_values = []
        p_risk_m_values = []
        p_risk_w_values = []
        
        failed_points = []
        
        # 🔧 修复：添加调试信息和容错机制
        print(f"  🔍 调试：检查all_results_dict键值匹配...")
        print(f"    目标参数：phi={phi}, theta={theta}, kappa={kappa}")
        print(f"    字典中总键数：{len(all_results_dict)}")
        
        # 显示字典中的前几个键作为样本
        sample_keys = list(all_results_dict.keys())[:3]
        for i, sample_key in enumerate(sample_keys):
            print(f"    样本键{i+1}: {sample_key}")
        
        matched_count = 0
        for r in r_values:
            # 🔧 修复浮点数精度问题：使用更强的容错匹配
            result = None
            
            # 第一步：尝试直接键匹配
            key = (phi, theta, kappa, r)
            if key in all_results_dict and all_results_dict[key]['success']:
                result = all_results_dict[key]
                matched_count += 1
            else:
                # 第二步：使用容错匹配寻找最佳匹配
                best_match = None
                best_r_diff = float('inf')
                
                for stored_key, stored_result in all_results_dict.items():
                    stored_phi, stored_theta, stored_kappa, stored_r = stored_key
                    if (abs(stored_phi - phi) < 1e-10 and 
                        abs(stored_theta - theta) < 1e-10 and 
                        abs(stored_kappa - kappa) < 1e-10 and 
                        stored_result['success']):
                        
                        r_diff = abs(stored_r - r)
                        if r_diff < best_r_diff:
                            best_r_diff = r_diff
                            best_match = stored_result
                
                # 如果找到了足够接近的匹配（差异小于1e-6）
                if best_match is not None and best_r_diff < 1e-6:
                    result = best_match
                    matched_count += 1
                else:
                    # 调试输出：显示最接近的键
                    closest_r_diff = float('inf')
                    closest_key = None
                    for stored_key in all_results_dict.keys():
                        stored_phi, stored_theta, stored_kappa, stored_r = stored_key
                        if (abs(stored_phi - phi) < 1e-10 and 
                            abs(stored_theta - theta) < 1e-10 and 
                            abs(stored_kappa - kappa) < 1e-10):
                            r_diff = abs(stored_r - r)
                            if r_diff < closest_r_diff:
                                closest_r_diff = r_diff
                                closest_key = stored_key
                    
                    print(f"    ❌ 未找到r={r:.6f}的匹配键")
                    if closest_key:
                        print(f"       最接近的键: {closest_key}, r差异={closest_r_diff:.2e}")
                    
                failed_points.append(r)
                # 使用NaN填充缺失值
                X_H_values.append(np.nan)
                X_M_values.append(np.nan)
                X_L_values.append(np.nan)
                p_risk_values.append(np.nan)
                p_risk_m_values.append(np.nan)
                p_risk_w_values.append(np.nan)
                continue
            
            # 成功找到匹配结果
            X_H_values.append(result['X_H'])
            X_M_values.append(result['X_M'])
            X_L_values.append(result['X_L'])
            p_risk_values.append(result['p_risk'])
            p_risk_m_values.append(result['p_risk_m'])
            p_risk_w_values.append(result['p_risk_w'])
        
        print(f"  📊 键值匹配结果：成功匹配 {matched_count}/{len(r_values)} 个r值")
        
        if failed_points:
            print(f"  ⚠️ Warning: {len(failed_points)} r values failed to find matches")
            print(f"      Failed r values: {failed_points[:5]}..." if len(failed_points) > 5 else f"      Failed r values: {failed_points}")
        else:
            print(f"  ✅ 所有r值都成功匹配到预计算结果")
        
        # 序参量跳变分析
        print(f"  2. Analyzing order parameter jumps...")
        
        # 移除NaN值进行跳变分析
        valid_mask = ~np.isnan(X_H_values)
        if np.sum(valid_mask) < 5:
            return {
                'status': 'failed',
                'phi': phi,
                'theta': theta,
                'kappa': kappa,
                'error': 'Insufficient valid data points for analysis'
            }
        
        valid_r_values = np.array(r_values)[valid_mask]
        valid_X_H = np.array(X_H_values)[valid_mask]
        
        # 使用改进的跳变检测
        jumps, r_jumps, jump_detected, z_score = detect_jumps_improved(valid_r_values, valid_X_H)
        
        # 提取跳变信息
        if jumps and r_jumps:
            max_jump_position = r_jumps[0]  # 第一个（最大的）跳变位置
            max_jump_size = jumps[0]        # 第一个（最大的）跳变大小
        else:
            max_jump_position = None
            max_jump_size = 0.0
        
        if jump_detected:
            print(f"  Significant jump detected: r_c = {max_jump_position:.4f}, jump size = {max_jump_size:.4f}, z-score = {z_score:.2f}")
        elif max_jump_position is not None:
            print(f"  Jump detected but not significant: r_c = {max_jump_position:.4f}, jump size = {max_jump_size:.4f}, z-score = {z_score:.2f}")
        else:
            print(f"  No jump detected")
        
        # 关联长度计算 - 物理上需要扰动分析，必须重新计算
        # 注释掉第二层关联长度计算，只保留第一层稳态计算
        print(f"  3. Computing correlation length (requires perturbation analysis)...")
        print(f"  ⚠️  关联长度计算已注释掉，只进行第一层稳态计算")
        
        # 创建模型实例
        model = ThresholdDynamicsModelV3Fixed(kappa_network_params, threshold_params)
        
        # 注释掉关联长度计算部分
        """
        try:
            # 并行计算关联长度（物理上需要扰动分析，不能基于预计算稳态值）
            corr_result = model.calculate_correlation_length_robust(
                r_values, 
                removal_type=removal_type, 
                n_processes=max(1, mp.cpu_count() - 1),  # 自动选择合理的进程数
                external_threshold_params=threshold_params,
                state_name='X_H',
                enable_diagnostics=False,
                enable_power_law=True,
                init_states=init_states  # 传递初始状态
            )
            
            # 处理返回值
            if len(corr_result) >= 4:
                corr_lengths, critical_r, raw_data, power_law_results = corr_result
                print(f"   Successfully retrieved correlation length data")
            else:
                raise ValueError(f"Correlation length calculation returned abnormal result")
                
        except RuntimeError as e:
            if "计算结果长度不匹配" in str(e):
                print(f"   ⚠️ 部分r值计算失败，尝试使用容错处理...")
                # 过滤掉容易失败的高r值，重新计算
                r_filtered = r_values[r_values <= 0.9]  # 移除r>0.9的点
                if len(r_filtered) < 10:
                    # 如果过滤后点数太少，抛出原始异常
                    raise e
                
                print(f"   使用过滤后的r值范围: [{r_filtered[0]:.3f}, {r_filtered[-1]:.3f}] ({len(r_filtered)}个点)")
                
                corr_result = model.calculate_correlation_length_robust(
                    r_filtered, 
                    removal_type=removal_type, 
                    n_processes=max(1, mp.cpu_count() - 1),
                    external_threshold_params=threshold_params,
                    state_name='X_H',
                    enable_diagnostics=False,
                    enable_power_law=True,
                    init_states=init_states
                )
                
                if len(corr_result) >= 4:
                    corr_lengths_filtered, critical_r, raw_data, power_law_results = corr_result
                    print(f"   ✅ 容错计算成功，使用{len(r_filtered)}个有效点")
                    
                    # 扩展结果到原始r_values长度，缺失值用NaN填充
                    corr_lengths = np.full(len(r_values), np.nan)
                    for i, r in enumerate(r_values):
                        if r in r_filtered:
                            filtered_idx = np.where(r_filtered == r)[0][0]
                            corr_lengths[i] = corr_lengths_filtered[filtered_idx]
                    
                else:
                    raise ValueError(f"容错处理后仍然失败")
            else:
                # 其他类型的RuntimeError，直接抛出
                raise e
        
        # Quality assessment（移到if语句外面）
        quality_info = assess_correlation_quality_v3(
            r_values, corr_lengths, critical_r, power_law_results
        )
        
        max_corr = np.max(corr_lengths)
        max_corr_idx = np.argmax(corr_lengths)
        max_corr_position = r_values[max_corr_idx]
        
        # 从幂律结果提取ν值
        nu_value = None
        if power_law_results and 'correlation_length_scaling' in power_law_results:
            scaling = power_law_results['correlation_length_scaling']
            if 'nu' in scaling:
                nu_value = scaling['nu']
        
        print(f"  Quality assessment: {quality_info['overall_score']}/{quality_info['max_score']} ({quality_info['quality_level']})")
        """
        
        # 简化版本：只进行第一层稳态计算，不计算关联长度
        corr_lengths = np.full(len(r_values), np.nan)  # 填充NaN
        critical_r = None
        power_law_results = None
        quality_info = {'overall_score': 0, 'max_score': 10, 'quality_level': 'no_data'}
        max_corr = np.nan
        max_corr_position = np.nan
        nu_value = None
        
        print(f"  ✅ 第一层稳态计算完成，关联长度计算已跳过")
        
        # 4. 确定相变类型（简化版本，只基于跳变检测）
        # 由于关联长度计算被注释掉，相变类型判断也相应简化
        if jump_detected:
            transition_type = 'first_order'  # 有跳变就是一级相变
        else:
            transition_type = 'continuous'   # 无跳变就是连续相变
        
        print(f"  简化相变类型判断: {transition_type}")
        
        # 注释掉原来的复杂相变类型判断
        """
        transition_type = determine_transition_type_v3(
            jump_detected, quality_info, max_jump_position, critical_r,
            X_H_values=X_H_values, r_values=r_values, power_law_results=power_law_results
        )
        """
        
        # 5. 保存结果（简化版本，关联长度相关值为NaN/None）
        point_result = {
            'phi': phi,
            'theta': theta,
            'kappa': kappa,  # 添加kappa信息
            'r_c': max_jump_position if jump_detected else None,  # 简化：只基于跳变位置
            'max_corr': max_corr,  # NaN
            'max_corr_position': max_corr_position,  # NaN
            'critical_r': critical_r,  # None
            'nu': nu_value,  # None
            'has_jump': jump_detected,
            'jump_position': max_jump_position if jump_detected else None,
            'jump_size': max_jump_size if jump_detected else 0,
            'transition_type': transition_type,
            'quality_info': quality_info,  # 简化版本
            'correlation_lengths': corr_lengths,  # 全为NaN
            'r_values': r_values,
            'X_H_values': X_H_values,
            'X_M_values': X_M_values,
            'X_L_values': X_L_values,
            'p_risk_values': p_risk_values,
            'p_risk_m_values': p_risk_m_values,
            'p_risk_w_values': p_risk_w_values,
            'power_law_results': power_law_results,  # None
            'failed_points': failed_points
        }
        
        # 保存结果
        result_file = os.path.join(point_dir, 'result.pkl')
        with open(result_file, 'wb') as f:
            pickle.dump(point_result, f)
        
        # 生成可视化
        generate_visualization_v3_with_kappa(point_result, point_dir, phi, kappa)
        
        # 修复格式化错误：添加对None值的处理
        r_c_value = point_result['r_c']
        r_c_str = f"{r_c_value:.4f}" if r_c_value is not None else "None"
        print(f"  ✅ Analysis completed: {transition_type}, r_c={r_c_str}")
        
        return {
            'status': 'success',
            'phi': phi,
            'theta': theta,
            'kappa': kappa,
            'result': point_result
        }
        
    except Exception as e:
        print(f"  ❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'phi': phi,
            'theta': theta,
            'kappa': kappa,
            'error': str(e)
        }

def quick_scan_critical_points_v3_optimized(phi_range, theta_range, removal_range, kappa_range=None,
                                  network_params=None, base_threshold_params=None, n_processes=None, 
                                  save_dir=None, abs_jump_threshold=0.2, trivial_threshold=0.1, 
                                  skip_existing=True, removal_type='mainstream',
                                           parallel_param_points=True, init_states=None,
                                           power_law_params=None):
    """
    V3版本的临界点扫描函数 - 优化版本，支持稳态结果缓存和幂律分布参数
    
    新增功能：
    - 稳态结果自动缓存和加载
    - 避免重复计算已有的稳态值
    - 支持中断后继续计算
    - 支持幂律分布参数配置
    
    参数:
        phi_range: phi值范围 [start, end, step] 或 phi值列表
        theta_range: theta值范围 [start, end, step] 或 theta值列表
        removal_range: 移除比例范围 [start, end, step]
        kappa_range: kappa值范围 [start, end, step] 或 kappa值列表，默认None使用network_params中的值
        network_params: 网络参数(可选)
        base_threshold_params: 基础阈值参数(可选)
        n_processes: 并行进程数(可选)
        save_dir: 结果保存目录(可选)
        abs_jump_threshold: 绝对跳跃阈值，默认0.2
        trivial_threshold: 微小跳跃阈值，默认0.1
        skip_existing: 是否跳过已有结果，默认True
        removal_type: 移除类型，默认'mainstream'
        parallel_param_points: 是否使用并行计算，默认True
        init_states: 初始状态字典，默认None
        power_law_params: 幂律分布参数字典，包含gamma_pref, k_min_pref等(可选)
    
    返回:
        详细的扫描结果字典
    """
    print(f"V3 优化版本扫描 - 支持稳态结果缓存和幂律分布参数...")
    
    # 设置保存目录
    if save_dir is None:
        import tempfile
        save_dir = tempfile.mkdtemp(prefix=f"v3_optimized_scan_")
    else:
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"Results will be saved in: {save_dir}")
    
    if network_params is None:
        raise ValueError("Must provide network parameters")
    
    # 处理幂律分布参数
    if power_law_params is not None:
        # 验证幂律分布参数
        gamma_pref = power_law_params.get('gamma_pref')
        k_min_pref = power_law_params.get('k_min_pref', 1)
        max_k = power_law_params.get('max_k', network_params.get('max_k', 200))
        
        if gamma_pref is not None:
            if not isinstance(gamma_pref, (int, float)) or gamma_pref <= 0:
                raise ValueError(f"幂律指数 gamma_pref 必须为正数，当前值: {gamma_pref}")
            if gamma_pref <= 1:
                raise ValueError(f"幂律指数 gamma_pref 必须大于1以确保分布可归一化，当前值: {gamma_pref}")
            
            if not isinstance(k_min_pref, int) or k_min_pref < 1:
                raise ValueError(f"幂律分布下界 k_min_pref 必须为正整数，当前值: {k_min_pref}")
            
            if not isinstance(max_k, int) or max_k <= k_min_pref:
                raise ValueError(f"幂律分布上界 max_k 必须为正整数且大于下界，当前值: {max_k}")
            
            # 更新网络参数
            network_params = network_params.copy()
            network_params['gamma_pref'] = gamma_pref
            network_params['k_min_pref'] = k_min_pref
            network_params['max_k'] = max_k
            
            print(f"配置幂律分布参数: γ={gamma_pref}, k_min={k_min_pref}, k_max={max_k}")
        else:
            print("未配置幂律分布参数，使用默认分布")
    else:
        print("未提供幂律分布参数，使用默认分布")
    
    # 生成参数值数组（复用原有逻辑）
    if isinstance(phi_range, list) or isinstance(phi_range, tuple):
        if len(phi_range) == 3 and all(isinstance(x, (int, float)) for x in phi_range):
            phi_start, phi_end, phi_step = phi_range
            phi_values = np.arange(phi_start, phi_end + phi_step/2, phi_step)
        else:
            phi_values = np.array(phi_range)
    else:
        phi_values = np.array([phi_range])
    
    if isinstance(theta_range, list) or isinstance(theta_range, tuple):
        if len(theta_range) == 3 and all(isinstance(x, (int, float)) for x in theta_range):
            theta_start, theta_end, theta_step = theta_range
            theta_values = np.arange(theta_start, theta_end + theta_step/2, theta_step)
        else:
            theta_values = np.array(theta_range)
    else:
        theta_values = np.array([theta_range])
    
    if kappa_range is None:
        default_kappa = network_params.get('k_out_mainstream', 60) + network_params.get('k_out_wemedia', 60)
        kappa_values = np.array([default_kappa])
        print(f"Using default kappa: {default_kappa}")
    else:
        if isinstance(kappa_range, list) or isinstance(kappa_range, tuple):
            if len(kappa_range) == 3 and all(isinstance(x, (int, float)) for x in kappa_range):
                kappa_start, kappa_end, kappa_step = kappa_range
                kappa_values = np.arange(kappa_start, kappa_end + kappa_step/2, kappa_step)
            else:
                kappa_values = np.array(kappa_range)
        else:
            kappa_values = np.array([kappa_range])
        print(f"Scanning kappa values: {kappa_values}")
    
    r_start, r_end, r_step = removal_range
    r_values = np.arange(r_start, r_end + r_step/2, r_step)
    
    # 创建结果存储
    results = {
        'phi_values': phi_values,
        'theta_values': theta_values,
        'kappa_values': kappa_values,
        'r_values': r_values,
        'critical_points': [],
        'init_states': init_states,
        'power_law_params': power_law_params
    }
    
    # 配置进程数
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    n_processes = min(n_processes, 112)
    print(f"Using {n_processes} processes for analysis")
    
    # 生成所有参数点（预先过滤约束条件）
    param_points = []
    invalid_points = []
    
    for phi in phi_values:
        for theta in theta_values:
            for kappa in kappa_values:
                # 检查约束条件：theta > phi
                if theta > phi:
                    param_points.append((phi, theta, kappa))
                else:
                    invalid_points.append((phi, theta, kappa))
    
    print(f"Total parameter points to process: {len(param_points)}")
    if invalid_points:
        print(f"⚠️  过滤掉 {len(invalid_points)} 个不符合约束的点 (theta <= phi)")
        print(f"   有效参数点: {len(param_points)}")
        print(f"   无效参数点示例: {invalid_points[:3]}...")
    
    # 检查是否有现有结果
    if skip_existing:
        existing_results = []
        for phi, theta, kappa in param_points:
            result_file = os.path.join(save_dir, f"phi_{phi:.3f}_theta_{theta:.3f}_kappa_{kappa:.0f}_result.pkl")
            if os.path.exists(result_file):
                existing_results.append((phi, theta, kappa))
        
        if existing_results:
            print(f"Found {len(existing_results)} existing results, skipping...")
            param_points = [(phi, theta, kappa) for phi, theta, kappa in param_points 
                          if (phi, theta, kappa) not in existing_results]
            print(f"Remaining points to process: {len(param_points)}")
    
    if not param_points:
        print("No new parameter points to process")
        return results
    
    # 检查稳态结果缓存
    steady_state_file = os.path.join(save_dir, "steady_state_cache.pkl")
    existing_steady_results = {}
    
    if os.path.exists(steady_state_file):
        try:
            with open(steady_state_file, 'rb') as f:
                existing_steady_results = pickle.load(f)
            print(f"📖 加载稳态结果缓存: {len(existing_steady_results)} 个结果")
        except Exception as e:
            print(f"❌ 加载稳态结果缓存失败: {str(e)}")
            existing_steady_results = {}
    else:
        print(f"📝 未找到稳态结果缓存文件")
    
    # 第一步：计算缺失的稳态值
    print(f"\n第一步：计算缺失的稳态值...")
    
    flattened_tasks = []
    for phi, theta, kappa in param_points:
        kappa_network_params = network_params.copy()
        kappa_network_params['k_out_mainstream'] = kappa // 2
        kappa_network_params['k_out_wemedia'] = kappa // 2
        
        for r_m in r_values:
            key = (phi, theta, kappa, r_m)
            
            # 检查是否已经计算过
            if key in existing_steady_results and existing_steady_results[key]['success']:
                continue
            
            task = (phi, theta, r_m, kappa_network_params, base_threshold_params, kappa, init_states)
            flattened_tasks.append(task)
    
    total_steady_tasks = len(param_points) * len(r_values)
    print(f"需要计算的稳态点: {len(flattened_tasks)}/{total_steady_tasks}")
    if total_steady_tasks > 0:
        print(f"缓存命中率: {(total_steady_tasks - len(flattened_tasks)) / total_steady_tasks * 100:.1f}%")
    
    # 并行计算缺失的稳态值
    new_steady_results = {}
    all_steady_results = []  # 初始化变量，防止UnboundLocalError
    
    if len(flattened_tasks) > 0:
        if parallel_param_points and len(flattened_tasks) > 1:
            print(f"Using {n_processes} processes to calculate steady state values...")
            
            with mp.Pool(processes=n_processes) as pool:
                all_steady_results = list(tqdm(
                    pool.map(compute_single_point_r_combination_with_kappa, flattened_tasks),
                    total=len(flattened_tasks),
                    desc="Calculating steady state values"
                ))
        else:
            print("Using serial mode to calculate steady state values...")
            all_steady_results = []
            for task in tqdm(flattened_tasks, desc="Calculating steady state values"):
                result = compute_single_point_r_combination_with_kappa(task)
                all_steady_results.append(result)
    
    # 整理新计算的结果
    for result in all_steady_results:
        key = (result['phi'], result['theta'], result['kappa'], result['r_m'])
        new_steady_results[key] = result
    
    # 合并所有稳态结果
    all_steady_results_dict = existing_steady_results.copy()
    all_steady_results_dict.update(new_steady_results)
    
    # 🔧 保存更新后的稳态结果
    if len(new_steady_results) > 0:
        try:
            with open(steady_state_file, 'wb') as f:
                pickle.dump(all_steady_results_dict, f)
            print(f"✅ 稳态结果已保存到: {steady_state_file}")
            print(f"   总计 {len(all_steady_results_dict)} 个稳态结果（新增 {len(new_steady_results)} 个）")
        except Exception as e:
            print(f"❌ 保存稳态结果失败: {str(e)}")
    
    # 统计成功率
    success_count = sum(1 for result in all_steady_results_dict.values() if result['success'])
    print(f"稳态计算总成功率: {success_count}/{len(all_steady_results_dict)} ({success_count/len(all_steady_results_dict)*100:.1f}%)")
    
    # 第二步：处理每个参数点（计算跳变和关联长度）
    print(f"\n第二步：串行处理各参数点的分析...")
    
    point_results = []
    failed_cases = 0
    
    for i, (phi, theta, kappa) in enumerate(param_points):
        print(f"\nProcessing parameter point {i+1}/{len(param_points)}: (κ, φ, θ) = ({kappa}, {phi:.2f}, {theta:.2f})")
        
        kappa_network_params = network_params.copy()
        kappa_network_params['k_out_mainstream'] = kappa // 2
        kappa_network_params['k_out_wemedia'] = kappa // 2
        
        result = process_parameter_point_flattened_with_kappa(
            phi, theta, kappa, r_values, kappa_network_params, 
            base_threshold_params, save_dir, 
            abs_jump_threshold, removal_type, 
            all_steady_results_dict, init_states
        )
        point_results.append(result)
    
    # 处理结果
    for result in point_results:
        if result['status'] == 'success':
            results['critical_points'].append(result['result'])
        else:
            failed_cases += 1

    # 输出摘要
    print("\n===== 优化版本扫描结果摘要 =====")
    print(f"总参数点: {len(param_points)}")
    print(f"成功案例: {len(results['critical_points'])}")
    print(f"失败案例: {failed_cases}")
    print(f"成功率: {len(results['critical_points'])/len(param_points)*100:.1f}%")
    
    # 保存最终结果
    results_file = os.path.join(save_dir, "scan_results.pkl")
    try:
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"✅ 扫描结果已保存到: {results_file}")
    except Exception as e:
        print(f"❌ 保存扫描结果失败: {str(e)}")
    
    return results

def assess_correlation_quality_v3(r_values, correlation_lengths, critical_r, power_law_results=None):
    """
    V3版本的质量评估，更严格的标准
    """
    r_array = np.array(r_values)
    corr_array = np.array(correlation_lengths)
    
    quality_info = {
        'overall_score': 0,
        'max_score': 10,
        'issues': [],
        'strengths': []
    }
    
    # 1. 峰值显著性
    max_idx = np.argmax(corr_array)
    peak_value = corr_array[max_idx]
    
    if len(corr_array) > 10:
        boundary_indices = list(range(min(5, len(corr_array)//4))) + list(range(-min(5, len(corr_array)//4), 0))
        background_values = corr_array[boundary_indices]
        
        if len(background_values) > 0:
            background_mean = np.mean(background_values)
            background_std = np.std(background_values)
            
            if background_std > 0:
                z_score = (peak_value - background_mean) / background_std
                
                if z_score > 4.0:
                    quality_info['overall_score'] += 3
                    quality_info['strengths'].append(f"Highly significant peak: {z_score:.1f}σ")
                elif z_score > 3.0:
                    quality_info['overall_score'] += 2
                    quality_info['strengths'].append(f"Significant peak: {z_score:.1f}σ")
                elif z_score > 2.0:
                    quality_info['overall_score'] += 1
                    quality_info['strengths'].append(f"Moderate significance: {z_score:.1f}σ")
                else:
                    quality_info['issues'].append(f"Weak statistical significance: {z_score:.1f}σ")
    
    # 2. 边界检查
    if critical_r is not None and (critical_r <= 0.08 or critical_r >= 0.92):
        quality_info['issues'].append(f"Boundary artifact: r_c={critical_r:.4f}")
    elif critical_r is not None:
        quality_info['overall_score'] += 2
        quality_info['strengths'].append("No boundary artifacts")
    else:
        # critical_r为None的情况
        quality_info['issues'].append("Critical point is None")
    
    # 3. 幂律质量
    if power_law_results and 'correlation_length_scaling' in power_law_results:
        scaling = power_law_results['correlation_length_scaling']
        if 'nu' in scaling and 'r_squared' in scaling:
            nu = scaling['nu']
            r_squared = scaling['r_squared']
            
            # 安全处理None值
            if (r_squared is not None and nu is not None and 
                r_squared >= 0.8 and 0.3 <= nu <= 2.0):
                quality_info['overall_score'] += 3
                quality_info['strengths'].append(f"Excellent power law: ν={nu:.3f}, R²={r_squared:.3f}")
            elif r_squared is not None and nu is not None and r_squared >= 0.7:
                quality_info['overall_score'] += 2
                quality_info['strengths'].append(f"Good power law: ν={nu:.3f}, R²={r_squared:.3f}")
            elif r_squared is not None:
                quality_info['issues'].append(f"Poor power law fit: R²={r_squared:.3f}")
            else:
                quality_info['issues'].append("Power law analysis failed (None values)")
    
    # 4. 平滑度
    if len(corr_array) >= 3:
        second_deriv = np.diff(corr_array, n=2)
        smoothness = np.sqrt(np.mean(second_deriv**2))
        
        if smoothness < 0.5:
            quality_info['overall_score'] += 2
            quality_info['strengths'].append(f"Excellent smoothness: {smoothness:.3f}")
        elif smoothness < 1.0:
            quality_info['overall_score'] += 1
        else:
            quality_info['issues'].append(f"Poor smoothness: {smoothness:.3f}")
    
    # 综合评估
    if quality_info['overall_score'] >= 8:
        quality_info['quality_level'] = "excellent"
    elif quality_info['overall_score'] >= 6:
        quality_info['quality_level'] = "good"
    elif quality_info['overall_score'] >= 4:
        quality_info['quality_level'] = "moderate"
    else:
        quality_info['quality_level'] = "poor"
    
    return quality_info

def determine_transition_type_v3(has_jump, quality_info, jump_position, critical_r, 
                                X_H_values=None, r_values=None, power_law_results=None):
    """
    V3版本的相变类型判断，基于物理特征的严格标准
    
    判断逻辑：
    1. 有跳变 + 临界点前平坦  = mixed_order (二阶相变特征 + 一阶跳跃)
    2. 有跳变 + 临界点前上升趋势 = first_order (纯一阶相变)
    3. 无跳变 + 高质量幂律 = second_order (纯二阶相变)
    4. 其他情况 = continuous 或根据具体情况判断
    """
    quality_level = quality_info.get('quality_level', 'poor')
    
    # 检查幂律分析质量 - 使用非常严格的标准
    has_good_power_law = False
    power_law_quality = "none"
    
    if power_law_results and 'correlation_length_scaling' in power_law_results:
        scaling = power_law_results['correlation_length_scaling']
        
        # 首先检查是否有错误
        if 'error' in scaling:
            power_law_quality = "failed"
            print(f"  幂律分析失败: {scaling['error']}")
        elif 'nu' in scaling and 'r_squared' in scaling and scaling['nu'] is not None and scaling['r_squared'] is not None:
            r_squared = scaling['r_squared']
            nu = scaling['nu']
            fit_points = scaling.get('fit_points', 0)
            has_quality_issue = scaling.get('quality_issue', False)
            
            print(f"  幂律分析结果: ν={nu:.3f}, R²={r_squared:.3f}, 拟合点数={fit_points}")
            
            # 简化的幂律标准 - 清晰的0.8阈值
            if (r_squared >= 0.80 and 0.3 <= nu <= 2.5 and fit_points >= 4 and not has_quality_issue):
                has_good_power_law = True
                power_law_quality = "good"
                print(f"  ✅ 检测到高质量幂律行为，支持二阶相变")
            else:
                power_law_quality = "poor"
                print(f"  ❌ 幂律质量不足，不支持二阶相变")
        else:
            power_law_quality = "incomplete"
            print(f"  ⚠️ 幂律分析结果不完整")
    
    if has_jump:
        # 有跳变的情况：需要分析临界点前的行为
        if X_H_values is not None and r_values is not None and critical_r is not None:
            # 分析临界点前的趋势
            r_array = np.array(r_values)
            X_H_array = np.array(X_H_values)
            
            # 找到临界点前的区域（临界点前20%的数据）
            pre_critical_mask = r_array < critical_r
            if np.sum(pre_critical_mask) >= 3:
                pre_critical_r = r_array[pre_critical_mask]
                pre_critical_X_H = X_H_array[pre_critical_mask]
                
                # 计算临界点前最后一段的斜率（使用最后30%的数据）
                n_recent = max(3, int(len(pre_critical_r) * 0.3))
                recent_r = pre_critical_r[-n_recent:]
                recent_X_H = pre_critical_X_H[-n_recent:]
                
                # 线性拟合计算斜率
                if len(recent_r) >= 3:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(recent_r, recent_X_H)
                    
                    print(f"  临界点前趋势分析: 斜率={slope:.4f}, R²={r_value**2:.3f}")
                    
                    # 修正判断逻辑：first_order在临界点前有上升，mixed_order在临界点前平坦
                    if abs(slope) < 0.05 and r_value**2 > 0.3:  # 平坦趋势（小斜率）
                        if has_good_power_law:
                            print(f"  📊 判断: 平坦趋势 + 高质量幂律 → mixed_order")
                            return 'mixed_order'  # 有跳变 + 平坦趋势 + 好的幂律 = mixed order
                        else:
                            print(f"  📊 判断: 平坦趋势，幂律质量不足 → first_order")
                            return 'first_order'  # 平坦但幂律质量不够，仍是一阶
                    elif slope > 0.1 and r_value**2 > 0.5:  # 明显上升趋势
                        print(f"  📊 判断: 上升趋势 + 跳变 → first_order")
                        return 'first_order'  # 有跳变 + 上升趋势 = 纯一阶相变
                    else:  # 下降或不显著趋势
                        print(f"  📊 判断: 不明确趋势 → first_order")
                        return 'first_order'  # 默认为一阶相变
        
        # 无法分析趋势时的fallback
        if has_good_power_law:
            return 'mixed_order'
        else:
            return 'first_order'
    
    else:
        # 无跳变的情况：主要看幂律质量
        if has_good_power_law:
            return 'second_order'  # 无跳变 + 高质量幂律 = 二阶相变
        elif quality_level in ['excellent', 'good']:
            return 'possible_second_order'  # 质量还可以但幂律不够好
        else:
            return 'continuous'  # 质量差，可能是连续变化

def generate_visualization_v3(result, save_dir, phi=None):
    """
    V3版本的可视化，更清晰的展示
    """
    try:
        theta = result['theta']
        phi = result.get('phi', phi)  # 优先使用结果中的phi值
        r_values = result['r_values']
        X_H_values = result['X_H_values']
        X_M_values = result['X_M_values']
        X_L_values = result['X_L_values']
        correlation_lengths = result['correlation_lengths']
        critical_r = result['r_c']
        transition_type = result['transition_type']
        quality_info = result.get('quality_info', {})
        power_law_results = result.get('power_law_results')
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 稳态图
        ax1 = axes[0, 0]
        ax1.plot(r_values, X_H_values, 'bo-', markersize=4, linewidth=2, label='X_H')
        
        # 安全处理critical_r为None的情况
        if critical_r is not None:
            ax1.axvline(x=critical_r, color='red', linestyle='--', linewidth=2, 
                       label=f'Critical r_c={critical_r:.4f}')
        else:
            # 如果critical_r为None，不显示垂直线，但在图例中标注
            ax1.plot([], [], color='red', linestyle='--', linewidth=2, label='Critical r_c=None')
        
        if result.get('has_jump') and result.get('jump_position'):
            jump_pos = result['jump_position']
            ax1.axvline(x=jump_pos, color='orange', linestyle=':', linewidth=2,
                       label=f'Jump at r={jump_pos:.4f}')
        
        ax1.set_xlabel('Removal Ratio (r)')
        ax1.set_ylabel('High Arousal Proportion (X_H)')
        ax1.set_title(f'Steady State (φ={phi:.2f}, θ={theta:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 关联长度图
        ax2 = axes[0, 1]
        ax2.plot(r_values, correlation_lengths, 'co-', markersize=4, linewidth=2)
        
        # 安全处理critical_r为None的情况
        if critical_r is not None:
            ax2.axvline(x=critical_r, color='red', linestyle='--', linewidth=2)
        
        max_idx = np.argmax(correlation_lengths)
        max_corr = correlation_lengths[max_idx]
        max_r = r_values[max_idx]
        ax2.plot(max_r, max_corr, 'ro', markersize=8, 
                label=f'Max: ξ={max_corr:.2f} at r={max_r:.4f}')
        
        ax2.set_xlabel('Removal Ratio (r)')
        ax2.set_ylabel('Correlation Length (ξ)')
        ax2.set_title('Correlation Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 幂律分析
        ax3 = axes[1, 0]
        if power_law_results and 'correlation_length_scaling' in power_law_results and critical_r is not None:
            scaling = power_law_results['correlation_length_scaling']
            if 'nu' in scaling and 'r_squared' in scaling:
                nu = scaling['nu']
                r_squared = scaling['r_squared']
                
                # 在临界点附近的数据
                dr = np.abs(r_values - critical_r)
                mask = (dr > 0.001) & (correlation_lengths > 0.01)
                
                if np.sum(mask) > 0:
                    ax3.loglog(dr[mask], correlation_lengths[mask], 'o', 
                              color='blue', markersize=6)
                    
                    # 拟合线
                    dr_fit = np.logspace(np.log10(min(dr[mask])), 
                                        np.log10(max(dr[mask])), 100)
                    intercept = scaling.get('intercept', 0)
                    corr_fit = 10**(intercept - nu * np.log10(dr_fit))
                    ax3.loglog(dr_fit, corr_fit, 'r-', linewidth=2, 
                              label=f'ξ ~ |r-r_c|^(-{nu:.3f})\nR² = {r_squared:.3f}')
                
                ax3.set_xlabel('|r - r_c|')
                ax3.set_ylabel('Correlation Length (ξ)')
                ax3.set_title('Power Law Analysis')
                ax3.legend()
                ax3.grid(True, alpha=0.3, which='both')
        else:
            # 如果没有幂律分析或critical_r为None，显示提示信息
            reason = 'No power law analysis'
            if critical_r is None:
                reason = 'No power law analysis\n(critical_r is None)'
            ax3.text(0.5, 0.5, reason, 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12)
        
        # 4. 质量信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 安全处理critical_r为None的情况
        critical_r_str = f"{critical_r:.4f}" if critical_r is not None else "None"
        
        quality_text = f"""Transition Type: {transition_type.upper()}
Quality Level: {quality_info.get('quality_level', 'unknown')}
Score: {quality_info.get('overall_score', 0)}/{quality_info.get('max_score', 10)}
Critical Point: r_c = {critical_r_str}

Strengths:
"""
        for strength in quality_info.get('strengths', [])[:3]:
            quality_text += f"• {strength}\n"
        
        quality_text += "\nIssues:\n"
        for issue in quality_info.get('issues', [])[:3]:
            quality_text += f"• {issue}\n"
        
        # Set color
        if transition_type == 'second_order':
            bg_color = 'lightgreen'
        elif transition_type in ['first_order', 'mixed_order']:
            bg_color = 'lightblue'
        else:
            bg_color = 'lightcoral'
        
        ax4.text(0.05, 0.95, quality_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))
        
        plt.suptitle(f'V3 Analysis: φ={phi:.2f}, θ={theta:.2f} - {transition_type.upper()}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'v3_analysis_phi{phi:.2f}_theta{theta:.2f}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Visualization charts saved")
        
    except Exception as e:
        print(f"    Error generating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_visualization_v3_with_kappa(result, save_dir, phi, kappa):
    """
    V3版本的可视化，更清晰的展示
    """
    try:
        theta = result['theta']
        phi = result.get('phi', phi)  # 优先使用结果中的phi值
        r_values = result['r_values']
        X_H_values = result['X_H_values']
        X_M_values = result['X_M_values']
        X_L_values = result['X_L_values']
        correlation_lengths = result['correlation_lengths']
        critical_r = result['r_c']
        transition_type = result['transition_type']
        quality_info = result.get('quality_info', {})
        power_law_results = result.get('power_law_results')
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 稳态图
        ax1 = axes[0, 0]
        ax1.plot(r_values, X_H_values, 'bo-', markersize=4, linewidth=2, label='X_H')
        
        # 安全处理critical_r为None的情况
        if critical_r is not None:
            ax1.axvline(x=critical_r, color='red', linestyle='--', linewidth=2, 
                       label=f'Critical r_c={critical_r:.4f}')
        else:
            # 如果critical_r为None，不显示垂直线，但在图例中标注
            ax1.plot([], [], color='red', linestyle='--', linewidth=2, label='Critical r_c=None')
        
        if result.get('has_jump') and result.get('jump_position'):
            jump_pos = result['jump_position']
            ax1.axvline(x=jump_pos, color='orange', linestyle=':', linewidth=2,
                       label=f'Jump at r={jump_pos:.4f}')
        
        ax1.set_xlabel('Removal Ratio (r)')
        ax1.set_ylabel('High Arousal Proportion (X_H)')
        ax1.set_title(f'Steady State (φ={phi:.2f}, θ={theta:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 关联长度图
        ax2 = axes[0, 1]
        ax2.plot(r_values, correlation_lengths, 'co-', markersize=4, linewidth=2)
        
        # 安全处理critical_r为None的情况
        if critical_r is not None:
            ax2.axvline(x=critical_r, color='red', linestyle='--', linewidth=2)
        
        max_idx = np.argmax(correlation_lengths)
        max_corr = correlation_lengths[max_idx]
        max_r = r_values[max_idx]
        ax2.plot(max_r, max_corr, 'ro', markersize=8, 
                label=f'Max: ξ={max_corr:.2f} at r={max_r:.4f}')
        
        ax2.set_xlabel('Removal Ratio (r)')
        ax2.set_ylabel('Correlation Length (ξ)')
        ax2.set_title('Correlation Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 幂律分析
        ax3 = axes[1, 0]
        if power_law_results and 'correlation_length_scaling' in power_law_results and critical_r is not None:
            scaling = power_law_results['correlation_length_scaling']
            if 'nu' in scaling and 'r_squared' in scaling:
                nu = scaling['nu']
                r_squared = scaling['r_squared']
                
                # 在临界点附近的数据
                dr = np.abs(r_values - critical_r)
                mask = (dr > 0.001) & (correlation_lengths > 0.01)
                
                if np.sum(mask) > 0:
                    ax3.loglog(dr[mask], correlation_lengths[mask], 'o', 
                              color='blue', markersize=6)
                    
                    # 拟合线
                    dr_fit = np.logspace(np.log10(min(dr[mask])), 
                                        np.log10(max(dr[mask])), 100)
                    intercept = scaling.get('intercept', 0)
                    corr_fit = 10**(intercept - nu * np.log10(dr_fit))
                    ax3.loglog(dr_fit, corr_fit, 'r-', linewidth=2, 
                              label=f'ξ ~ |r-r_c|^(-{nu:.3f})\nR² = {r_squared:.3f}')
                
                ax3.set_xlabel('|r - r_c|')
                ax3.set_ylabel('Correlation Length (ξ)')
                ax3.set_title('Power Law Analysis')
                ax3.legend()
                ax3.grid(True, alpha=0.3, which='both')
        else:
            # 如果没有幂律分析或critical_r为None，显示提示信息
            reason = 'No power law analysis'
            if critical_r is None:
                reason = 'No power law analysis\n(critical_r is None)'
            ax3.text(0.5, 0.5, reason, 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12)
        
        # 4. 质量信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 安全处理critical_r为None的情况
        critical_r_str = f"{critical_r:.4f}" if critical_r is not None else "None"
        
        quality_text = f"""Transition Type: {transition_type.upper()}
Quality Level: {quality_info.get('quality_level', 'unknown')}
Score: {quality_info.get('overall_score', 0)}/{quality_info.get('max_score', 10)}
Critical Point: r_c = {critical_r_str}

Strengths:
"""
        for strength in quality_info.get('strengths', [])[:3]:
            quality_text += f"• {strength}\n"
        
        quality_text += "\nIssues:\n"
        for issue in quality_info.get('issues', [])[:3]:
            quality_text += f"• {issue}\n"
        
        # Set color
        if transition_type == 'second_order':
            bg_color = 'lightgreen'
        elif transition_type in ['first_order', 'mixed_order']:
            bg_color = 'lightblue'
        else:
            bg_color = 'lightcoral'
        
        ax4.text(0.05, 0.95, quality_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))
        
        plt.suptitle(f'V3 Analysis: φ={phi:.2f}, θ={theta:.2f} - {transition_type.upper()}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'v3_analysis_phi{phi:.2f}_theta{theta:.2f}_kappa{kappa}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Visualization charts saved")
        
    except Exception as e:
        print(f"    Error generating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def load_scan_results(results_file):
    """
    Load scan results file
    
    Parameters:
        results_file: Path to results file
        
    Returns:
        Scan results dictionary or None
    """
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Failed to load results file: {str(e)}")
        return None

def load_case_data_from_dir(base_dir, phi, theta, kappa=120):
    """
    Load data for specific parameter point from directory
    
    Parameters:
        base_dir: Base directory path
        phi: phi value
        theta: theta value
        kappa: kappa value (k_out_mainstream + k_out_wemedia), default 120
        
    Returns:
        Data dictionary or None
    """
    # Use integer naming to find directory
    kappa_int = int(round(kappa))
    phi_int = int(round(phi * 100))  # Convert 0.55 to 55
    theta_int = int(round(theta * 100))  # Convert 0.37 to 37
    point_dir = os.path.join(base_dir, f"kappa{kappa_int:03d}_phi{phi_int:03d}_theta{theta_int:03d}")
    result_file = os.path.join(point_dir, 'result.pkl')
    
    if not os.path.exists(result_file):
        # Try old phi_theta naming (backward compatibility)
        point_dir_old = os.path.join(base_dir, f"phi{phi_int:03d}_theta{theta_int:03d}")
        result_file_old = os.path.join(point_dir_old, 'result.pkl')
        
        if os.path.exists(result_file_old):
            result_file = result_file_old
        else:
            # Try even older float naming
            point_dir_float = os.path.join(base_dir, f"phi{phi:.2f}_theta{theta:.2f}")
            result_file_float = os.path.join(point_dir_float, 'result.pkl')
            
            if os.path.exists(result_file_float):
                result_file = result_file_float
            else:
                print(f"Results file does not exist: {result_file}")
                return None
    
    try:
        with open(result_file, 'rb') as f:
            result = pickle.load(f)
        
        # Ensure data integrity
        required_keys = ['r_values', 'X_H_values', 'correlation_lengths']
        if not all(key in result for key in required_keys):
            print(f"Results file data incomplete: {result_file}")
            return None
        
        return result
    except Exception as e:
        print(f"Failed to load results file: {result_file} - {str(e)}")
        return None

def create_comparison_plots(results, save_dir):
    """
    Create comparison plots
    
    Parameters:
        results: Scan results dictionary
        save_dir: Save directory
    """
    if 'critical_points' not in results or not results['critical_points']:
        print("No available data for visualization")
        return
    
    comparison_dir = os.path.join(save_dir, "comparison_plots")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Group by transition type
    transition_types = {}
    for point in results['critical_points']:
        t_type = point.get('transition_type', 'unknown')
        if t_type not in transition_types:
            transition_types[t_type] = []
        transition_types[t_type].append(point)
    
    # Create comparison plots for each transition type
    for t_type, type_data in transition_types.items():
        if len(type_data) < 2:
            continue
            
        # Order parameter comparison
        create_order_parameter_comparison(type_data, comparison_dir, t_type)
        
        # Correlation length comparison
        create_correlation_length_comparison(type_data, comparison_dir, t_type)
    
    # Create global comparison plots
    create_global_phase_diagram(results['critical_points'], comparison_dir)
    
    # Create parameter trend plots
    create_parameter_trends(results['critical_points'], comparison_dir)

def create_order_parameter_comparison(type_data, save_dir, transition_type):
    """Create order parameter comparison plot"""
    plt.figure(figsize=(12, 8))
    
    cmap = plt.get_cmap('tab10')
    
    for i, data in enumerate(type_data):
        phi = data['phi']
        theta = data['theta']
        r_values = data['r_values']
        X_H_values = data['X_H_values']
        critical_r = data.get('r_c')
        
        color = cmap(i % 10)
        
        # 完全移除label参数，避免自动创建图例
        plt.plot(r_values, X_H_values, 'o-', color=color, 
                markersize=4, linewidth=2)
        
        if critical_r is not None:
            plt.axvline(x=critical_r, color=color, linestyle='--', alpha=0.7)
    
    plt.xlabel('Removal Ratio (r)')
    plt.ylabel('High Arousal Proportion (X_H)')
    plt.title(f'Order Parameter Comparison - {transition_type.replace("_", " ").title()}')
    # 移除图例显示，因为case数量太多
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"order_parameter_{transition_type}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()
    
    print(f"Order parameter comparison plot saved: {filename}")

def create_correlation_length_comparison(type_data, save_dir, transition_type):
    """Create correlation length comparison plot"""
    plt.figure(figsize=(12, 8))
    
    cmap = plt.get_cmap('tab10')
    
    for i, data in enumerate(type_data):
        phi = data['phi']
        theta = data['theta']
        r_values = data['r_values']
        correlation_lengths = data['correlation_lengths']
        critical_r = data.get('r_c')
        nu = data.get('nu')
        
        color = cmap(i % 10)
        
        # 完全移除label参数，避免自动创建图例
        plt.plot(r_values, correlation_lengths, 'o-', color=color, 
                markersize=4, linewidth=2)
        
        if critical_r is not None:
            plt.axvline(x=critical_r, color=color, linestyle='--', alpha=0.7)
    
    plt.xlabel('Removal Ratio (r)')
    plt.ylabel('Correlation Length (ξ)')
    plt.title(f'Correlation Length Comparison - {transition_type.replace("_", " ").title()}')
    # 移除图例显示，因为case数量太多
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"correlation_length_{transition_type}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()
    
    print(f"Correlation length comparison plot saved: {filename}")

def create_global_phase_diagram(all_data, save_dir):
    """Create global phase diagram with kappa support"""
    phi_values = []
    theta_values = []
    kappa_values = []
    critical_rs = []
    transition_types = []
    
    for data in all_data:
        phi_values.append(data['phi'])
        theta_values.append(data['theta'])
        kappa_values.append(data.get('kappa', 120))  # 默认值120
        # 安全处理None值：将None转换为np.nan
        r_c = data.get('r_c', None)
        if r_c is None:
            critical_rs.append(np.nan)
        else:
            critical_rs.append(r_c)
        transition_types.append(data.get('transition_type', 'unknown'))
    
    # 检查是否有多个kappa值
    unique_kappas = list(set(kappa_values))
    has_multiple_kappas = len(unique_kappas) > 1
    
    if has_multiple_kappas:
        # 创建包含kappa维度的图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 按kappa分组的相变类型分布
        ax1 = axes[0, 0]
        type_colors = {
            'second_order': 'red',
            'first_order': 'blue', 
            'mixed_order': 'purple',
            'possible_second_order': 'orange',
            'continuous': 'gray',
            'unknown': 'black'
        }
        
        for kappa in sorted(unique_kappas):
            kappa_indices = [i for i, k in enumerate(kappa_values) if k == kappa]
            if kappa_indices:
                phi_subset = [phi_values[i] for i in kappa_indices]
                theta_subset = [theta_values[i] for i in kappa_indices]
                types_subset = [transition_types[i] for i in kappa_indices]
                
                for t_type in set(types_subset):
                    type_indices = [i for i, t in enumerate(types_subset) if t == t_type]
                    if type_indices:
                        phi_type = [phi_subset[i] for i in type_indices]
                        theta_type = [theta_subset[i] for i in type_indices]
                        
                        ax1.scatter(phi_type, theta_type, 
                                   c=type_colors.get(t_type, 'black'),
                                   marker='o' if kappa == min(unique_kappas) else '^',
                                   s=50, alpha=0.7,
                                   label=f'κ={kappa}, {t_type}' if len(phi_type) > 0 else None)
        
        ax1.set_xlabel('φ (Low Arousal Threshold)')
        ax1.set_ylabel('θ (High Arousal Threshold)')
        ax1.set_title('Phase Diagram by Kappa - Transition Types')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 临界点分布（按kappa分组）
        ax2 = axes[0, 1]
        valid_indices = []
        for i, r_c in enumerate(critical_rs):
            try:
                if r_c is not None and not np.isnan(r_c):
                    valid_indices.append(i)
            except (TypeError, ValueError):
                continue
        
        if valid_indices:
            phi_valid = [phi_values[i] for i in valid_indices]
            theta_valid = [theta_values[i] for i in valid_indices]
            kappa_valid = [kappa_values[i] for i in valid_indices]
            r_c_valid = [critical_rs[i] for i in valid_indices]
            
            # 为不同kappa使用不同标记
            kappa_markers = {k: marker for k, marker in zip(sorted(unique_kappas), ['o', '^', 's', 'D', 'v'])}
            
            for kappa in sorted(unique_kappas):
                kappa_indices = [i for i, k in enumerate(kappa_valid) if k == kappa]
                if kappa_indices:
                    phi_kappa = [phi_valid[i] for i in kappa_indices]
                    theta_kappa = [theta_valid[i] for i in kappa_indices]
                    r_c_kappa = [r_c_valid[i] for i in kappa_indices]
                    
                    scatter = ax2.scatter(phi_kappa, theta_kappa, c=r_c_kappa,
                                        marker=kappa_markers.get(kappa, 'o'),
                                        cmap='viridis', s=50, alpha=0.7,
                                        label=f'κ={kappa}')
            
            # 添加颜色条
            plt.colorbar(scatter, ax=ax2, label='Critical Point (r_c)')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No valid critical points found', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        ax2.set_xlabel('φ (Low Arousal Threshold)')
        ax2.set_ylabel('θ (High Arousal Threshold)')
        ax2.set_title('Phase Diagram by Kappa - Critical Points')
        ax2.grid(True, alpha=0.3)
        
        # 3. Kappa vs Critical Point
        ax3 = axes[1, 0]
        if valid_indices:
            kappa_valid = [kappa_values[i] for i in valid_indices]
            r_c_valid = [critical_rs[i] for i in valid_indices]
            
            ax3.scatter(kappa_valid, r_c_valid, alpha=0.7)
            ax3.set_xlabel('κ (Connectivity)')
            ax3.set_ylabel('Critical Point (r_c)')
            ax3.set_title('Critical Point vs Connectivity')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No valid critical points', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Critical Point vs Connectivity (No Data)')
        
        # 4. 统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 创建统计表格
        stats_text = "Statistics by Kappa:\n\n"
        for kappa in sorted(unique_kappas):
            kappa_indices = [i for i, k in enumerate(kappa_values) if k == kappa]
            kappa_types = [transition_types[i] for i in kappa_indices]
            type_counts = {}
            for t in kappa_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            
            stats_text += f"κ = {kappa} ({len(kappa_indices)} points):\n"
            for t_type, count in type_counts.items():
                stats_text += f"  {t_type}: {count}\n"
            stats_text += "\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Global Phase Diagram with Kappa Analysis', fontsize=16, fontweight='bold')
        
    else:
        # 单一kappa的情况，使用原来的布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Transition type distribution
        type_colors = {
            'second_order': 'red',
            'first_order': 'blue', 
            'mixed_order': 'purple',
            'possible_second_order': 'orange',
            'continuous': 'gray',
            'unknown': 'black'
        }
        
        for t_type in set(transition_types):
            indices = [i for i, t in enumerate(transition_types) if t == t_type]
            if indices:
                phi_subset = [phi_values[i] for i in indices]
                theta_subset = [theta_values[i] for i in indices]
                ax1.scatter(phi_subset, theta_subset, 
                           c=type_colors.get(t_type, 'black'), 
                           label=t_type, s=50, alpha=0.7)
        
        ax1.set_xlabel('φ (Low Arousal Threshold)')
        ax1.set_ylabel('θ (High Arousal Threshold)')
        ax1.set_title(f'Phase Diagram - Transition Types (κ={unique_kappas[0]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Critical point distribution - 安全处理NaN值
        valid_indices = []
        for i, r_c in enumerate(critical_rs):
            try:
                if r_c is not None and not np.isnan(r_c):
                    valid_indices.append(i)
            except (TypeError, ValueError):
                continue
        
        if valid_indices:
            phi_valid = [phi_values[i] for i in valid_indices]
            theta_valid = [theta_values[i] for i in valid_indices]
            r_c_valid = [critical_rs[i] for i in valid_indices]
            
            scatter = ax2.scatter(phi_valid, theta_valid, c=r_c_valid, 
                                cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(scatter, ax=ax2, label='Critical Point (r_c)')
        else:
            ax2.text(0.5, 0.5, 'No valid critical points found', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        ax2.set_xlabel('φ (Low Arousal Threshold)')
        ax2.set_ylabel('θ (High Arousal Threshold)')
        ax2.set_title(f'Phase Diagram - Critical Points (κ={unique_kappas[0]})')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_phase_diagram.png"), dpi=300)
    plt.close()
    
    print("Global phase diagram saved: global_phase_diagram.png")

def create_parameter_trends(all_data, save_dir):
    """Create parameter trends plot"""
    phi_values = [data['phi'] for data in all_data]
    theta_values = [data['theta'] for data in all_data]
    
    # 安全处理None值
    r_c_values = []
    nu_values = []
    
    for data in all_data:
        # 处理r_c值
        r_c = data.get('r_c', None)
        if r_c is None:
            r_c_values.append(np.nan)
        else:
            r_c_values.append(r_c)
        
        # 处理nu值
        nu = data.get('nu', None)
        if nu is None:
            nu_values.append(np.nan)
        else:
            nu_values.append(nu)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # r_c vs phi - 安全处理NaN值
    valid_indices = []
    for i, r_c in enumerate(r_c_values):
        try:
            if r_c is not None and not np.isnan(r_c):
                valid_indices.append(i)
        except (TypeError, ValueError):
            continue
    
    if valid_indices:
        phi_valid = [phi_values[i] for i in valid_indices]
        r_c_valid = [r_c_values[i] for i in valid_indices]
        axes[0,0].plot(phi_valid, r_c_valid, 'bo-')
        axes[0,0].set_xlabel('φ')
        axes[0,0].set_ylabel('Critical Point (r_c)')
        axes[0,0].set_title('Critical Point vs φ')
        axes[0,0].grid(True, alpha=0.3)
    else:
        axes[0,0].text(0.5, 0.5, 'No valid critical points', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Critical Point vs φ (No Data)')
    
    # ν vs phi - 安全处理NaN值
    valid_indices = []
    for i, nu in enumerate(nu_values):
        try:
            if nu is not None and not np.isnan(nu):
                valid_indices.append(i)
        except (TypeError, ValueError):
            continue
    
    if valid_indices:
        phi_valid = [phi_values[i] for i in valid_indices]
        nu_valid = [nu_values[i] for i in valid_indices]
        axes[0,1].plot(phi_valid, nu_valid, 'ro-')
        axes[0,1].set_xlabel('φ')
        axes[0,1].set_ylabel('Critical Exponent (ν)')
        axes[0,1].set_title('Critical Exponent vs φ')
        axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].text(0.5, 0.5, 'No valid critical exponents', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Critical Exponent vs φ (No Data)')
    
    # Transition type statistics
    transition_counts = {}
    for data in all_data:
        t_type = data.get('transition_type', 'unknown')
        transition_counts[t_type] = transition_counts.get(t_type, 0) + 1
    
    types = list(transition_counts.keys())
    counts = list(transition_counts.values())
    axes[1,0].bar(types, counts)
    axes[1,0].set_xlabel('Transition Type')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Transition Type Distribution')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Clear last subplot
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "parameter_trends.png"), dpi=300)
    plt.close()
    
    print("Parameter trends plot saved: parameter_trends.png")

def analyze_scan_results(results_file, create_plots=True):
    """
    Analyze scan results and generate visualization
    
    Parameters:
        results_file: Path to results file
        create_plots: Whether to create plots
    """
    print(f"Analyzing scan results: {results_file}")
    
    # Load results
    results = load_scan_results(results_file)
    if results is None:
        return
    
    # Output basic statistics
    critical_points = results.get('critical_points', [])
    print(f"Total parameter points: {len(critical_points)}")
    
    # Transition type statistics
    type_counts = {}
    for point in critical_points:
        t_type = point.get('transition_type', 'unknown')
        type_counts[t_type] = type_counts.get(t_type, 0) + 1
    
    print("\nTransition type statistics:")
    for t_type, count in type_counts.items():
        print(f"  {t_type}: {count}")
    
    # Valid data statistics
    valid_r_c = sum(1 for p in critical_points if p.get('r_c') is not None and not np.isnan(p.get('r_c', np.nan)))
    valid_nu = sum(1 for p in critical_points if p.get('nu') is not None and not np.isnan(p.get('nu', np.nan)))
    
    print(f"\nValid critical points: {valid_r_c}/{len(critical_points)}")
    print(f"Valid critical exponent points: {valid_nu}/{len(critical_points)}")
    
    # Create visualization
    if create_plots and critical_points:
        save_dir_from_file = os.path.dirname(results_file)
        create_comparison_plots(results, save_dir_from_file)
        print(f"\nVisualization charts saved to: {os.path.join(save_dir_from_file, 'comparison_plots')}")
        
    return results

def run_scan(phi_range, theta_range, r_range=(0.01, 0.99, 0.01), 
             kappa_range=None, save_dir=None, n_processes=None, skip_existing=True, 
             parallel_param_points=True, analyze_results=True, power_law_params=None,
             network_params=None, threshold_params=None):
    """
    在Jupyter中运行扫描的便捷函数 - 支持kappa扫描和幂律分布参数（优化版本）
    
    Parameters:
        phi_range: phi范围，可以是[start, end, step]或值列表
        theta_range: theta范围，可以是[start, end, step]或值列表  
        r_range: r范围，默认(0.01, 0.99, 0.01)
        kappa_range: kappa范围，可以是[start, end, step]或值列表，默认None使用120
        save_dir: 保存目录，默认None会自动创建
        n_processes: 进程数，默认None会自动设置
        skip_existing: 是否跳过已有结果，默认True
        parallel_param_points: 是否并行处理参数点，默认True
        analyze_results: 是否自动进行可视化分析，默认True
        power_law_params: 幂律分布参数字典，包含gamma_pref, k_min_pref等(可选)
        network_params: 网络参数字典，如果不提供则使用默认值
        threshold_params: 阈值参数字典，如果不提供则使用默认值
        
    Returns:
        扫描结果字典
        
    Note:
        使用优化版本，支持稳态结果缓存，避免重复计算
        kappa = k_out_mainstream + k_out_wemedia，会平均分配
        例如：kappa=100 → k_out_mainstream=50, k_out_wemedia=50
        支持幂律分布参数配置，如果不提供power_law_params则使用默认泊松分布
    """
    # 网络参数（支持用户自定义，否则使用默认值）
    if network_params is None:
        network_params = {
            'n_mainstream': 1000,
            'n_wemedia': 1000,
            'n_public': 5000,
            'k_out_mainstream': 60,  # 默认值，会根据kappa调整
            'k_out_wemedia': 60,     # 默认值，会根据kappa调整
            'k_out_public': 10,
            'max_k': 200,            # 度分布上界
            'use_original_like_dist': False  # 不使用原始分布
        }
    
    # 阈值参数（支持用户自定义，否则使用默认值）
    if threshold_params is None:
        threshold_params = {
            'theta': 0.55,
            'phi': 0.2
        }
    
    print("🚀 开始V3优化版本相变点扫描...")
    print("✨ 新功能: 稳态结果自动缓存，支持中断后继续计算")
    print("✨ 新功能: 支持幂律分布参数配置")
    print(f"phi范围: {phi_range}")
    print(f"theta范围: {theta_range}")
    print(f"r范围: {r_range}")
    
    if kappa_range is not None:
        print(f"kappa范围: {kappa_range}")
        print("📝 kappa分配说明: kappa会平均分配给主流媒体和自媒体")
        print("   例如: kappa=100 → k_out_mainstream=50, k_out_wemedia=50")
    else:
        default_kappa = network_params['k_out_mainstream'] + network_params['k_out_wemedia']
        print(f"使用默认kappa: {default_kappa}")
    
    # 处理幂律分布参数
    if power_law_params is not None:
        print(f"幂律分布参数: {power_law_params}")
        gamma_pref = power_law_params.get('gamma_pref')
        k_min_pref = power_law_params.get('k_min_pref', 1)
        max_k = power_law_params.get('max_k', 200)
        
        if gamma_pref is not None:
            print(f"  使用幂律分布: γ={gamma_pref}, k_min={k_min_pref}, k_max={max_k}")
        else:
            print("  未配置幂律指数，使用默认分布")
    else:
        print("使用默认泊松分布")
    
    # 🔧 使用优化版本函数
    results = quick_scan_critical_points_v3_optimized(
        phi_range=phi_range,
        theta_range=theta_range,
        removal_range=r_range,
        kappa_range=kappa_range,
        network_params=network_params,
        base_threshold_params=threshold_params,
        power_law_params=power_law_params,  # 新增幂律分布参数
        n_processes=n_processes,
        save_dir=save_dir,
        skip_existing=skip_existing,
        parallel_param_points=parallel_param_points
    )
    
    print("\n✅ V3优化版本扫描完成！")
    
    # 输出结果摘要
    if 'critical_points' in results:
        type_counts = {}
        kappa_counts = {}
        
        for point in results['critical_points']:
            t_type = point.get('transition_type', 'unknown')
            type_counts[t_type] = type_counts.get(t_type, 0) + 1
            
            kappa = point.get('kappa', 'unknown')
            kappa_counts[kappa] = kappa_counts.get(kappa, 0) + 1
        
        print("\n📊 相变类型统计:")
        for t_type, count in type_counts.items():
            print(f"  {t_type}: {count}")
        
        # 如果有多个kappa值，显示kappa分布
        if len(kappa_counts) > 1:
            print("\n📊 Kappa分布统计:")
            for kappa, count in sorted(kappa_counts.items()):
                if kappa != 'unknown':
                    print(f"  κ={kappa}: {count} 个参数点 (k_out_mainstream={kappa/2}, k_out_wemedia={kappa/2})")
            
            # 按kappa分组的相变类型统计
            print("\n📊 按Kappa分组的相变类型:")
            kappa_values = sorted([k for k in kappa_counts.keys() if k != 'unknown'])
            for kappa in kappa_values:
                kappa_points = [p for p in results['critical_points'] if p.get('kappa') == kappa]
                kappa_type_counts = {}
                for point in kappa_points:
                    t_type = point.get('transition_type', 'unknown')
                    kappa_type_counts[t_type] = kappa_type_counts.get(t_type, 0) + 1
                
                print(f"  κ={kappa} (k_out={kappa/2}+{kappa/2}):")
                for t_type, count in kappa_type_counts.items():
                    print(f"    {t_type}: {count}")
        
        # 显示分布类型信息
        if power_law_params is not None:
            gamma_pref = power_law_params.get('gamma_pref')
            if gamma_pref is not None:
                print(f"\n📊 分布类型: 幂律分布 (γ={gamma_pref})")
            else:
                print(f"\n📊 分布类型: 泊松分布")
        else:
            print(f"\n📊 分布类型: 泊松分布")
        
        # 自动进行可视化分析
        if analyze_results:
            save_dir_final = save_dir if save_dir else "."
            # 根据是否有kappa扫描选择结果文件名
            if kappa_range is not None:
                results_file = os.path.join(save_dir_final, "scan_results.pkl")
            else:
                results_file = os.path.join(save_dir_final, "scan_results.pkl")
                
            if os.path.exists(results_file):
                print(f"\n🎨 自动进行可视化分析...")
                analyze_scan_results(results_file, create_plots=True)
    
    return results
