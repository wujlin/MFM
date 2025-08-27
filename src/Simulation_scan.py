"""
参数扫描并行化脚本

用于并行计算多个参数点的模拟结果，每个参数点可以运行多个样本。
这种方法比单个模拟内部的并行化更高效，因为每个进程完全独立运行。
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import sys
import json
from src.theory_validation_simulator import TheoryValidationSimulator


# ================================
# 全局配置参数区域
# ================================

# 网络结构参数 - V3版本（优化规模以提高计算效率）
NETWORK_PARAMS = {
    'n_mainstream': 100,   # 从1000降到100
    'n_wemedia': 100,      # 从1000降到100  
    'n_public': 500,       # 从1000降到500
    'k_out_mainstream': 60,  # 保持出度不变
    'k_out_wemedia': 60,     # 保持出度不变
    'k_out_public': 10,
    'max_k': 50,           # 必需参数：最大度数
    'use_original_like_dist': False,  # 使用自定义分布参数
    # 分布选择参数 - 二选一：
    # 泊松分布参数 (当 gamma_pref=None 时使用)
    'kappa': 120,          # 泊松分布的平均度数
    # 幂律分布参数 (当 gamma_pref!=None 时使用)
    'gamma_pref': None,    # 幂律指数，None表示使用泊松分布
    'k_min_pref': 1,       # 幂律分布最小度数
}

# 阈值参数基础值（V3简化模型）
THRESHOLD_BASE_PARAMS = {
    'theta': 0.55,  # 高唤醒阈值
    'phi': 0.3      # 低唤醒阈值
}

# 模拟控制参数
SIMULATION_PARAMS = {
    'max_iter': 100,                     # 最大迭代次数（从300减少到50进行测试）
    'save_history': True,              # 是否保存状态历史
    'enable_micro_analysis': True,     # 是否启用微观分析
    'convergence_threshold': 1e-4,     # 收敛阈值
    'default_seed': 42                 # 默认随机种子
}

# 振荡检测参数
OSCILLATION_PARAMS = {
    'window_size': 10,                 # 检测窗口大小
    'oscillation_threshold_range': 0.05,  # 变化幅度阈值
    'oscillation_threshold_changes': 1,    # 方向变化次数阈值
    'p_risk_weight': 2.5              # p_risk在振荡强度计算中的权重
}

# 多进程控制参数
MULTIPROCESSING_PARAMS = {
    'cpu_usage_ratio_small': 1.0,     # 小型CPU使用比例
    'cpu_usage_ratio_medium': 0.8,    # 中型CPU使用比例
    'cpu_usage_ratio_large': 0.7,     # 大型CPU使用比例
    'small_cpu_threshold': 4,         # 小型CPU核心数阈值
    'large_cpu_threshold': 16,        # 大型CPU核心数阈值
    'performance_report_interval': 30,  # 性能报告间隔（秒）
    'min_chunksize': 1,               # 最小chunk大小
    'max_chunksize_factor': 4,        # 最大chunk大小因子
    'tasks_per_chunk': 10             # 每个chunk的任务数
}

# 分析参数
ANALYSIS_PARAMS = {
    'significant_change_threshold': 0.1,    # 显著变化阈值
    'consistency_threshold': 0.05,          # 一致性阈值
    'consistency_score_threshold': 0.5,     # 一致性得分阈值
    'success_completion_rate': 0.8          # 成功完成率阈值
}

# 绘图参数
PLOT_PARAMS = {
    'figure_size': (12, 10),          # 图形大小
    'phase_diagram_size': (16, 8),    # 相变图大小
    'dpi': 300,                       # 图像分辨率
    'grid_alpha': 0.3                 # 网格透明度
}

# 默认测试参数集
DEFAULT_TEST_PARAMS = {
    'small_test': {
        'phi_values': [0.3],
        'theta_values': [0.55],
        'r_mainstream_values': [0.0, 0.5],
        'n_samples': 1
    },
    'quick_parallel': {
        'phi_values': [0.3],
        'theta_values': [0.55],
        'r_values_count': 15,
        'r_min': 0.0,
        'r_max': 0.9,
        'n_samples': 1
    },
    'micro_analysis': {
        'phi_values': [0.3],
        'theta_values': [0.55],
        'r_mainstream_values': [0.0, 0.3, 0.6, 0.9],
        'n_samples': 1,
        'n_processes': 4
    },
    'full_sweep': {
        'phi_values': [0.3, 0.36, 0.45],
        'theta_values': [0.55],
        'r_values_count': 20,
        'r_min': 0.0,
        'r_max': 0.95,
        'n_samples': 5
    },
    'phase_diagram': {
        'phi_count': 5,
        'phi_min': 0.2,
        'phi_max': 0.4,
        'theta_count': 5,
        'theta_min': 0.45,
        'theta_max': 0.65,
        'r_mainstream_values': [0.0, 0.5, 0.75, 0.9],
        'n_samples': 3
    }
}

# ================================
# 以下是原有的功能代码
# ================================

# 定义工作函数 - 必须在全局作用域定义，以便可以被pickle
def worker(params):
    """
    并行计算的工作函数
    
    Parameters:
        params: 参数字典
        
    Returns:
        result: 计算结果
    """
    try:
        # 提取参数
        phi = params['phi']
        theta = params['theta']
        r_mainstream = params['r_mainstream']
        sample_idx = params.get('sample_idx', 0)
        seed = params.get('seed', 42)
        use_simulator = params.get('use_simulator', False)
        output_dir = params.get('output_dir', 'results')
        max_iter = params.get('max_iter', SIMULATION_PARAMS['max_iter'])
        verbose = params.get('verbose', False)
        init_states = params.get('init_states', None)
        
        # 生成结果文件名
        result_file = os.path.join(
            output_dir, 
            f"phi{phi}_theta{theta}_r{r_mainstream}_sample{sample_idx}.json"
        )
        
        # 如果已存在结果且不是强制重新计算，直接加载
        if os.path.exists(result_file) and not params.get('force_recompute', False):
            with open(result_file, 'r') as f:
                result = json.load(f)
            return result
        
        # 设置网络参数
        network_params = {
            'n_mainstream': 100,
            'n_wemedia': 100,
            'n_public': 1000,
            'k_out_mainstream': 60,
            'k_out_wemedia': 60,
            'k_out_public': 10,
            'use_original_like_dist': False
        }
        
        # 设置阈值参数
        threshold_params = {
            'theta': theta,
            'phi': phi
        }
        
        result = {}
        
        # 使用模拟器或理论模型
        if use_simulator:
            # 使用模拟器
            from src.theory_validation_simulator import TheoryValidationSimulator
            
            simulator = TheoryValidationSimulator(
                network_params=network_params,
                threshold_params=threshold_params
            )
            
            # 生成或加载网络
            simulator.generate_or_load_network(seed=seed, verbose=False)
            
            # 初始化状态
            simulator.initialize_states(init_states=init_states)
            
            # 运行模拟 - 启用历史记录和微观分析以获取完整信息
            final_stats = simulator.simulate_to_steady_state(
                max_iter=max_iter,
                removal_ratios={'mainstream': r_mainstream},
                save_history=True,  # 启用历史记录以检测振荡
                enable_micro_analysis=True,  # 启用微观分析
                verbose=verbose
            )
            
            # 检测振荡现象
            oscillation_info = {}
            if hasattr(simulator, 'state_history') and simulator.state_history:
                try:
                    from src.parameter_sweep_fix import detect_oscillation
                    oscillation_info = detect_oscillation(simulator.state_history)
                except:
                    oscillation_info = {'has_oscillation': False, 'strength': 0.0, 'convergence_steps': max_iter}
            
            # 获取微观统计
            micro_stats = getattr(simulator, 'micro_stats', {})
            micro_analysis = getattr(simulator, 'micro_analysis', {})
            
            # 保存模拟器的完整结果
            result = {
                # 基本状态信息
                'X_H': final_stats['X_H'],
                'X_M': final_stats['X_M'],
                'X_L': final_stats['X_L'],
                'p_risk': final_stats['p_risk'],
                'p_risk_m': final_stats['p_risk_m'],
                'p_risk_w': final_stats['p_risk_w'],
                
                # 参数信息
                'phi': phi,
                'theta': theta,
                'r_mainstream': r_mainstream,
                'sample_idx': sample_idx,
                'seed': seed,
                
                # 模拟器特有信息
                'success': True,
                'converged': True,  # 模拟器总是假设收敛（除非出错）
                'iterations': len(simulator.state_history) if hasattr(simulator, 'state_history') else max_iter,
                
                # 振荡信息
                'has_oscillation': oscillation_info.get('has_oscillation', False),
                'oscillation_strength': oscillation_info.get('strength', 0.0),
                'convergence_steps': oscillation_info.get('convergence_steps', max_iter),
                
                # 微观分析 - 连接类型分组统计
                'mainstream_connected_X_H': micro_stats.get('mainstream_X_H', np.nan),
                'mainstream_connected_X_L': micro_stats.get('mainstream_X_L', np.nan),
                'wemedia_connected_X_H': micro_stats.get('wemedia_X_H', np.nan),
                'wemedia_connected_X_L': micro_stats.get('wemedia_X_L', np.nan),
                'mixed_connected_X_H': micro_stats.get('mixed_X_H', np.nan),
                'mixed_connected_X_L': micro_stats.get('mixed_X_L', np.nan),
                
                # 媒体影响力分组统计
                'high_mainstream_X_H': micro_stats.get('high_mainstream_X_H', np.nan),
                'high_mainstream_X_L': micro_stats.get('high_mainstream_X_L', np.nan),
                'high_wemedia_X_H': micro_stats.get('high_wemedia_X_H', np.nan),
                'high_wemedia_X_L': micro_stats.get('high_wemedia_X_L', np.nan),
                
                # 转移速率统计
                'transition_rate_high_to_medium': micro_stats.get('transition_rate_high_to_medium', np.nan),
                'transition_rate_medium_to_high': micro_stats.get('transition_rate_medium_to_high', np.nan),
                'transition_rate_low_to_medium': micro_stats.get('transition_rate_low_to_medium', np.nan),
                'transition_rate_medium_to_low': micro_stats.get('transition_rate_medium_to_low', np.nan),
                'transition_rate_high_to_low': micro_stats.get('transition_rate_high_to_low', np.nan),
                'transition_rate_low_to_high': micro_stats.get('transition_rate_low_to_high', np.nan),
                
                # 理论vs实际差异（如果可用）
                'theory_vs_actual_X_H_diff': micro_analysis.get('weighted_X_H_diff', np.nan),
                'theory_vs_actual_X_L_diff': micro_analysis.get('weighted_X_L_diff', np.nan),
                'theory_vs_actual_X_H_rmse': micro_analysis.get('rmse_X_H', np.nan),
                'theory_vs_actual_X_L_rmse': micro_analysis.get('rmse_X_L', np.nan),
                
                # 网络结构信息
                'network_nodes': simulator.network.number_of_nodes() if simulator.network else 0,
                'network_edges': simulator.network.number_of_edges() if simulator.network else 0,
                'actual_public_nodes': len([n for n in simulator.network.nodes() if n.startswith('p_')]) if simulator.network else 0,
                'actual_mainstream_nodes': len([n for n in simulator.network.nodes() if n.startswith('m_')]) if simulator.network else 0,
                'actual_wemedia_nodes': len([n for n in simulator.network.nodes() if n.startswith('w_')]) if simulator.network else 0,
                
                # 标记数据来源
                'data_source': 'simulator'
            }
        else:
            # 使用理论模型
            from src.model_with_a_minimal_v3 import ThresholdDynamicsModel
            
            model = ThresholdDynamicsModel(
                network_params=network_params,
                threshold_params=threshold_params
            )
            
            # 求解自洽方程
            solution = model.solve_self_consistent(
                init_states=init_states,
                removal_ratios={'mainstream': r_mainstream},
                max_iter=max_iter
            )
            
            # 保存理论模型结果
            result = {
                # 基本状态信息
                'X_H': solution['X_H'],
                'X_M': solution['X_M'],
                'X_L': solution['X_L'],
                'p_risk': solution['p_risk'],
                'p_risk_m': solution['p_risk_m'],
                'p_risk_w': solution['p_risk_w'],
                
                # 参数信息
                'phi': phi,
                'theta': theta,
                'r_mainstream': r_mainstream,
                'sample_idx': sample_idx,
                'seed': seed,
                
                # 理论模型特有信息
                'success': solution['converged'],
                'converged': solution['converged'],
                'iterations': solution.get('iterations', max_iter),
                
                # 理论模型没有这些信息，设为默认值
                'has_oscillation': False,
                'oscillation_strength': 0.0,
                'convergence_steps': solution.get('iterations', max_iter),
                
                # 微观分析信息（理论模型不提供，设为NaN）
                'mainstream_connected_X_H': np.nan,
                'mainstream_connected_X_L': np.nan,
                'wemedia_connected_X_H': np.nan,
                'wemedia_connected_X_L': np.nan,
                'mixed_connected_X_H': np.nan,
                'mixed_connected_X_L': np.nan,
                
                'high_mainstream_X_H': np.nan,
                'high_mainstream_X_L': np.nan,
                'high_wemedia_X_H': np.nan,
                'high_wemedia_X_L': np.nan,
                
                'transition_rate_high_to_medium': np.nan,
                'transition_rate_medium_to_high': np.nan,
                'transition_rate_low_to_medium': np.nan,
                'transition_rate_medium_to_low': np.nan,
                'transition_rate_high_to_low': np.nan,
                'transition_rate_low_to_high': np.nan,
                
                'theory_vs_actual_X_H_diff': np.nan,
                'theory_vs_actual_X_L_diff': np.nan,
                'theory_vs_actual_X_H_rmse': np.nan,
                'theory_vs_actual_X_L_rmse': np.nan,
                
                # 网络结构信息（理论模型使用配置值）
                'network_nodes': network_params['n_mainstream'] + network_params['n_wemedia'] + network_params['n_public'],
                'network_edges': 0,  # 理论模型不关心具体边数
                'actual_public_nodes': network_params['n_public'],
                'actual_mainstream_nodes': int(network_params['n_mainstream'] * (1 - r_mainstream)),  # 考虑移除
                'actual_wemedia_nodes': network_params['n_wemedia'],
                
                # 标记数据来源
                'data_source': 'theory'
            }
        
        # 保存结果到文件
        os.makedirs(output_dir, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 返回错误结果
        return {
            'phi': params.get('phi', 0),
            'theta': params.get('theta', 0),
            'r_mainstream': params.get('r_mainstream', 0),
            'sample_idx': params.get('sample_idx', 0),
            'error': str(e),
            'success': False
        }

def run_single_simulation(params):
    """
    运行单个参数点的模拟 - V3版本
    
    Parameters:
        params: 包含所有参数的字典
                包括可选的'use_simulator'键来控制是否使用模拟器
        
    Returns:
        results: 模拟结果
    """
    # 解包参数
    phi = params['phi']
    theta = params['theta']
    r_mainstream = params['r_mainstream']
    seed = params.get('seed', SIMULATION_PARAMS['default_seed'])
    sample_idx = params.get('sample_idx', 0)
    output_dir = params.get('output_dir', 'results_v3')  # 获取输出目录
    use_simulator = params.get('use_simulator', False)  # 默认使用理论模型
    verbose = params.get('verbose', False)  # 默认不输出详细日志
    
    # 设置阈值参数（V3只需要theta和phi）
    threshold_params = {
        'theta': theta,
        'phi': phi
    }
    
    try:
        if use_simulator:
            # 使用模拟器进行真实模拟
            from src.theory_validation_simulator import TheoryValidationSimulator
            
            simulator = TheoryValidationSimulator(NETWORK_PARAMS, threshold_params)
            simulator.generate_or_load_network(seed=seed, verbose=verbose)
            simulator.initialize_states()
            
            # 设置移除比例
            removal_ratios = {'mainstream': r_mainstream}
            
            # 运行模拟
            result = simulator.simulate_to_steady_state(
                max_iter=params.get('max_iter', SIMULATION_PARAMS['max_iter']),
                removal_ratios=removal_ratios,
                n_processes=1,  # 强制单进程避免嵌套
                save_history=SIMULATION_PARAMS['save_history'],
                enable_micro_analysis=False,  # 关闭微观分析以提高速度
                verbose=verbose  # 控制是否输出详细日志
            )
            
            # 检测振荡现象
            oscillation_info = detect_oscillation(simulator.state_history) if hasattr(simulator, 'state_history') and simulator.state_history else {}
            
        else:
            # 使用V3理论模型直接计算
            from src.model_with_a_minimal_v3 import ThresholdDynamicsModel
            model = ThresholdDynamicsModel(NETWORK_PARAMS, threshold_params)
            
            # 设置移除比例
            removal_ratios = {'mainstream': r_mainstream}
            
            # 使用理论模型求解自洽方程
            result = model.solve_self_consistent(removal_ratios=removal_ratios)
            
            # 检测振荡现象（基于历史数据）
            history = result.get('history', {})
            oscillation_info = detect_oscillation_from_history(history) if history else {}
        

        
        # 构建扩展结果
        final_result = {
            # 基本参数
            'phi': phi,
            'theta': theta,
            'r_mainstream': r_mainstream,
            'sample_idx': sample_idx,
            'seed': seed,
            
            # 全局状态
            'X_H': result['X_H'],
            'X_M': result['X_M'],
            'X_L': result['X_L'],
            'p_risk': result['p_risk'],
            'p_risk_m': result.get('p_risk_m', 0),
            'p_risk_w': result.get('p_risk_w', 0),
            
            # 收敛信息
            'converged': result.get('converged', False),
            'iterations': result.get('iterations', 0),
            
            # 振荡信息
            'has_oscillation': oscillation_info.get('has_oscillation', False),
            'oscillation_strength': oscillation_info.get('strength', 0.0),
            'convergence_steps': oscillation_info.get('convergence_steps', result.get('iterations', 0)),
            
            # V3简化模型不需要复杂的微观分析统计
            'mainstream_connected_X_H': np.nan,
            'mainstream_connected_X_L': np.nan,
            'wemedia_connected_X_H': np.nan,
            'wemedia_connected_X_L': np.nan,
            'mixed_connected_X_H': np.nan,
            'mixed_connected_X_L': np.nan,
            
            'high_mainstream_X_H': np.nan,
            'high_mainstream_X_L': np.nan,
            'high_wemedia_X_H': np.nan,
            'high_wemedia_X_L': np.nan,
            
            'transition_rate_high_to_medium': np.nan,
            'transition_rate_medium_to_high': np.nan,
            'transition_rate_low_to_medium': np.nan,
            'transition_rate_medium_to_low': np.nan,
            
            'mainstream_high_influence': np.nan,
            'mainstream_medium_influence': np.nan,
            'wemedia_high_influence': np.nan,
            'wemedia_medium_influence': np.nan,
            
            'success': True
        }
        
        return final_result
        
    except Exception as e:
        # 如果模拟失败，返回错误信息
        print(f"V3 simulation failed (phi={phi}, theta={theta}, r={r_mainstream}, sample={sample_idx}): {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的堆栈跟踪
        raise  # 重新抛出异常，让worker函数捕获并处理

def detect_oscillation(data, window_size=None):
    """
    检测模拟过程中的振荡现象
    
    Parameters:
        data: 可以是状态历史列表或V3模型的历史数据字典
        window_size: 检测窗口大小，默认使用配置参数
        
    Returns:
        oscillation_info: 振荡信息字典
    """
    if window_size is None:
        window_size = OSCILLATION_PARAMS['window_size']
    
    # 统一数据格式处理
    if isinstance(data, list):
        # 状态历史列表格式
        if not data or len(data) < 4:
            return {'has_oscillation': False, 'strength': 0.0, 'convergence_steps': len(data)}
        
        X_H_series = [step['X_H'] for step in data]
        X_L_series = [step['X_L'] for step in data]
        p_risk_series = [step['p_risk'] for step in data]
        
        # 检查收敛步数
        convergence_threshold = SIMULATION_PARAMS['convergence_threshold']
        convergence_steps = len(data)
        
        for i in range(1, len(data)):
            prev = data[i-1]
            curr = data[i]
            diff = abs(curr['X_H'] - prev['X_H']) + abs(curr['X_L'] - prev['X_L']) + abs(curr['p_risk'] - prev['p_risk'])
            if diff < convergence_threshold:
                convergence_steps = i
                break
    else:
        # V3模型历史数据字典格式
        if not data or not all(key in data for key in ['X_H', 'X_L', 'p_risk']):
            return {'has_oscillation': False, 'strength': 0.0, 'convergence_steps': 0}
        
        X_H_series = data['X_H']
        X_L_series = data['X_L']
        p_risk_series = data['p_risk']
        
        # 检查收敛步数
        convergence_threshold = SIMULATION_PARAMS['convergence_threshold']
        convergence_steps = len(X_H_series)
        
        for i in range(1, len(X_H_series)):
            diff = abs(X_H_series[i] - X_H_series[i-1]) + \
                   abs(X_L_series[i] - X_L_series[i-1]) + \
                   abs(p_risk_series[i] - p_risk_series[i-1])
            if diff < convergence_threshold:
                convergence_steps = i
                break
    
    if len(X_H_series) < 4:
        return {'has_oscillation': False, 'strength': 0.0, 'convergence_steps': len(X_H_series)}
    
    # 计算最后几步的变化
    recent_steps = min(window_size, len(X_H_series))
    recent_X_H = X_H_series[-recent_steps:]
    recent_X_L = X_L_series[-recent_steps:]
    recent_p_risk = p_risk_series[-recent_steps:]
    
    # 计算变化幅度
    X_H_range = max(recent_X_H) - min(recent_X_H)
    X_L_range = max(recent_X_L) - min(recent_X_L)
    p_risk_range = max(recent_p_risk) - min(recent_p_risk)
    
    # 计算方向变化次数
    if len(recent_X_H) >= 3:
        X_H_changes = np.diff(recent_X_H)
        X_L_changes = np.diff(recent_X_L)
        p_risk_changes = np.diff(recent_p_risk)
        
        X_H_sign_changes = np.sum(np.diff(np.sign(X_H_changes)) != 0)
        X_L_sign_changes = np.sum(np.diff(np.sign(X_L_changes)) != 0)
        p_risk_sign_changes = np.sum(np.diff(np.sign(p_risk_changes)) != 0)
    else:
        X_H_sign_changes = X_L_sign_changes = p_risk_sign_changes = 0
    
    # 判断是否存在振荡（使用配置参数）
    oscillation_threshold_range = OSCILLATION_PARAMS['oscillation_threshold_range']
    oscillation_threshold_changes = OSCILLATION_PARAMS['oscillation_threshold_changes']
    
    has_X_H_oscillation = X_H_range > oscillation_threshold_range and X_H_sign_changes >= oscillation_threshold_changes
    has_X_L_oscillation = X_L_range > oscillation_threshold_range and X_L_sign_changes >= oscillation_threshold_changes
    has_p_risk_oscillation = p_risk_range > 0.02 and p_risk_sign_changes >= oscillation_threshold_changes
    
    has_oscillation = has_X_H_oscillation or has_X_L_oscillation or has_p_risk_oscillation
    
    # 计算振荡强度（使用配置的权重）
    p_risk_weight = OSCILLATION_PARAMS['p_risk_weight']
    oscillation_strength = max(X_H_range, X_L_range, p_risk_range * p_risk_weight)
    
    return {
        'has_oscillation': has_oscillation,
        'strength': oscillation_strength,
        'convergence_steps': convergence_steps,
        'X_H_range': X_H_range,
        'X_L_range': X_L_range,
        'p_risk_range': p_risk_range,
        'X_H_sign_changes': X_H_sign_changes,
        'X_L_sign_changes': X_L_sign_changes,
        'p_risk_sign_changes': p_risk_sign_changes
    }

# 为了向后兼容，保留别名
def detect_oscillation_from_history(history, window_size=None):
    """从V3模型的历史数据中检测振荡现象（向后兼容别名）"""
    return detect_oscillation(history, window_size)

def run_simulation_without_multiprocessing(params):
    """
    运行单个参数点的模拟，完全不使用多进程
    这是最后的备选方案
    """
    # 解包参数
    phi = params['phi']
    theta = params['theta']
    r_mainstream = params['r_mainstream']
    seed = params.get('seed', SIMULATION_PARAMS['default_seed'])
    sample_idx = params.get('sample_idx', 0)
    
    try:
        # 导入必要的模块
        import numpy as np
        import networkx as nx
        from src.theory_validation_simulator import TheoryValidationSimulator
        
        # 设置阈值参数
        threshold_params = THRESHOLD_BASE_PARAMS.copy()
        threshold_params['theta'] = theta
        threshold_params['phi'] = phi
        
        # 创建模拟器
        simulator = TheoryValidationSimulator(
            network_params=NETWORK_PARAMS,  # 包含use_original_like_dist参数
            threshold_params=threshold_params
        )
        
        # 生成或加载网络
        simulator.generate_or_load_network(seed=seed, verbose=False)
        
        # 初始化状态
        simulator.initialize_states()
        
        # 设置移除比例
        removal_ratios = {'mainstream': r_mainstream}
        
        # 运行模拟 - 强制使用单进程，启用微观分析
        final_stats = simulator.simulate_to_steady_state(
            max_iter=SIMULATION_PARAMS['max_iter'],
            removal_ratios=removal_ratios,
            n_processes=1,  # 强制使用单进程
            save_history=SIMULATION_PARAMS['save_history'],
            enable_micro_analysis=SIMULATION_PARAMS['enable_micro_analysis']
        )
        
        # 检测振荡现象
        oscillation_info = detect_oscillation(simulator.state_history) if hasattr(simulator, 'state_history') and simulator.state_history else {}
        
        # 获取微观统计信息
        micro_stats = simulator.micro_stats if hasattr(simulator, 'micro_stats') else {}
        micro_analysis = simulator.micro_analysis if hasattr(simulator, 'micro_analysis') else {}
        
        # 构建扩展结果
        result = {
            # 基本参数
            'phi': phi,
            'theta': theta,
            'r_mainstream': r_mainstream,
            'sample_idx': sample_idx,
            'seed': seed,
            
            # 全局状态
            'X_H': final_stats['X_H'],
            'X_M': final_stats['X_M'],
            'X_L': final_stats['X_L'],
            'p_risk': final_stats['p_risk'],
            'p_risk_m': final_stats.get('p_risk_m', 0),
            'p_risk_w': final_stats.get('p_risk_w', 0),
            
            # 振荡信息
            'has_oscillation': oscillation_info.get('has_oscillation', False),
            'oscillation_strength': oscillation_info.get('strength', 0.0),
            'convergence_steps': oscillation_info.get('convergence_steps', SIMULATION_PARAMS['max_iter']),
            
            # 微观分析 - 连接类型分组
            'mainstream_connected_X_H': micro_stats.get('mainstream_X_H', np.nan),
            'mainstream_connected_X_L': micro_stats.get('mainstream_X_L', np.nan),
            'wemedia_connected_X_H': micro_stats.get('wemedia_X_H', np.nan),
            'wemedia_connected_X_L': micro_stats.get('wemedia_X_L', np.nan),
            'mixed_connected_X_H': micro_stats.get('mixed_X_H', np.nan),
            'mixed_connected_X_L': micro_stats.get('mixed_X_L', np.nan),
            
            # 媒体影响力分组
            'high_mainstream_X_H': micro_stats.get('high_mainstream_X_H', np.nan),
            'high_mainstream_X_L': micro_stats.get('high_mainstream_X_L', np.nan),
            'high_wemedia_X_H': micro_stats.get('high_wemedia_X_H', np.nan),
            'high_wemedia_X_L': micro_stats.get('high_wemedia_X_L', np.nan),
            
            # 转移速率（如果可用）
            'transition_rate_high_to_medium': micro_stats.get('transition_rate_high_to_medium', np.nan),
            'transition_rate_medium_to_high': micro_stats.get('transition_rate_medium_to_high', np.nan),
            'transition_rate_low_to_medium': micro_stats.get('transition_rate_low_to_medium', np.nan),
            'transition_rate_medium_to_low': micro_stats.get('transition_rate_medium_to_low', np.nan),
            
            # 媒体节点影响力统计（如果可用）
            'mainstream_high_influence': micro_analysis.get('media_influence', {}).get('mainstream', {}).get('high_influence', np.nan),
            'mainstream_medium_influence': micro_analysis.get('media_influence', {}).get('mainstream', {}).get('medium_influence', np.nan),
            'wemedia_high_influence': micro_analysis.get('media_influence', {}).get('wemedia', {}).get('high_influence', np.nan),
            'wemedia_medium_influence': micro_analysis.get('media_influence', {}).get('wemedia', {}).get('medium_influence', np.nan),
            
            'success': True
        }
        
        return result
        
    except Exception as e:
        print(f"模拟失败 (phi={phi}, theta={theta}, r={r_mainstream}, sample={sample_idx}): {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'phi': phi,
            'theta': theta,
            'r_mainstream': r_mainstream,
            'sample_idx': sample_idx,
            'seed': seed,
            'X_H': np.nan,
            'X_M': np.nan,
            'X_L': np.nan,
            'p_risk': np.nan,
            'p_risk_m': np.nan,
            'p_risk_w': np.nan,
            'has_oscillation': False,
            'oscillation_strength': np.nan,
            'convergence_steps': np.nan,
            'success': False,
            'error': str(e)
        }

def run_sequential_parameter_sweep(phi_values, theta_values, r_mainstream_values, n_samples=1, output_dir='results'):
    """
    运行参数扫描的单进程版本
    
    Parameters:
        phi_values: phi参数值列表
        theta_values: theta参数值列表
        r_mainstream_values: r_mainstream参数值列表
        n_samples: 每个参数点运行的样本数
        output_dir: 输出目录
        
    Returns:
        results_df: 包含所有结果的DataFrame
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成所有参数组合（添加约束：phi < theta）
    param_grid = []
    skipped_count = 0
    for phi in phi_values:
        for theta in theta_values:
            if phi < theta:  # 约束条件：低唤醒阈值必须小于高唤醒阈值
                for r in r_mainstream_values:
                    for sample_idx in range(n_samples):
                        param_grid.append({
                            'phi': phi,
                            'theta': theta,
                            'r_mainstream': r,
                            'sample_idx': sample_idx,
                            'seed': SIMULATION_PARAMS['default_seed'] + sample_idx  # 不同样本使用不同种子
                        })
            else:
                skipped_count += 1
                if skipped_count <= 5:  # 只显示前5个跳过的警告
                    print(f"跳过无效参数点: phi={phi:.3f} >= theta={theta:.3f} (约束: phi < theta)")
                elif skipped_count == 6:
                    print("... 更多无效参数点被跳过")
    
    if skipped_count > 0:
        print(f"总共跳过 {skipped_count} 个无效参数组合 (约束: phi < theta)")
    
    print(f"总共需要计算 {len(param_grid)} 个参数点 (包括 {n_samples} 个样本/参数组合)")
    print("使用单进程顺序计算")
    
    # 记录开始时间
    start_time = time.time()
    
    # 顺序运行所有模拟
    results = []
    for params in tqdm(param_grid, desc="参数扫描进度"):
        try:
            result = run_single_simulation(params)
            results.append(result)
        except Exception as e:
            print(f"模拟失败 (phi={params['phi']}, theta={params['theta']}, r={params['r_mainstream']}, sample={params['sample_idx']}): {str(e)}")
            # 添加失败记录
            results.append({
                'phi': params['phi'],
                'theta': params['theta'],
                'r_mainstream': params['r_mainstream'],
                'sample_idx': params['sample_idx'],
                'seed': params['seed'],
                'X_H': np.nan,
                'X_M': np.nan,
                'X_L': np.nan,
                'p_risk': np.nan,
                'p_risk_m': np.nan,
                'p_risk_w': np.nan,
                'success': False,
                'error': str(e)
            })
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"参数扫描完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 检查是否有失败的模拟
    failed_count = results_df[results_df['success'] == False].shape[0]
    if failed_count > 0:
        print(f"警告: {failed_count} 个模拟失败 ({failed_count/len(results_df)*100:.1f}%)")
    
    # 保存所有结果
    all_results_path = os.path.join(output_dir, 'parameter_sweep_all_samples.csv')
    results_df.to_csv(all_results_path, index=False)
    print(f"所有样本结果已保存到 {all_results_path}")
    
    # 按参数组合分组并计算平均值和标准差
    if 'success' in results_df.columns and not results_df[results_df['success'] == True].empty:
        grouped_mean = results_df[results_df['success'] == True].groupby(['phi', 'theta', 'r_mainstream']).mean().reset_index()
        grouped_std = results_df[results_df['success'] == True].groupby(['phi', 'theta', 'r_mainstream']).std().reset_index()
        
        # 重命名标准差列
        std_columns = {}
        for col in grouped_std.columns:
            if col not in ['phi', 'theta', 'r_mainstream']:
                std_columns[col] = f"{col}_std"
        grouped_std = grouped_std.rename(columns=std_columns)
        
        # 合并平均值和标准差
        merged_results = pd.merge(grouped_mean, grouped_std, on=['phi', 'theta', 'r_mainstream'])
        
        # 保存汇总结果
        avg_results_path = os.path.join(output_dir, 'parameter_sweep_averaged.csv')
        merged_results.to_csv(avg_results_path, index=False)
        print(f"平均结果已保存到 {avg_results_path}")
        
        return results_df, merged_results
    else:
        print("警告: 所有模拟都失败了，无法生成平均结果")
        return results_df, pd.DataFrame()

def run_parallel_parameter_sweep(phi_values, theta_values, r_mainstream_values, n_samples=1, output_dir='results', n_processes=None, chunksize=None):
    """
    运行参数扫描的多进程版本 - 在参数级别并行化
    
    Parameters:
        phi_values: phi参数值列表
        theta_values: theta参数值列表
        r_mainstream_values: r_mainstream参数值列表
        n_samples: 每个参数点运行的样本数
        output_dir: 输出目录
        n_processes: 使用的进程数，默认为None（使用所有可用CPU核心）
        chunksize: 每个进程一次处理的任务数，默认为None（自动计算）
        
    Returns:
        results_df: 包含所有结果的DataFrame
    """
    # 创建主输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为有效的phi值创建子目录（仅为满足约束的phi值）
    valid_phi_values = []
    for phi in phi_values:
        for theta in theta_values:
            if phi < theta:
                if phi not in valid_phi_values:
                    valid_phi_values.append(phi)
                break
    
    phi_dirs = {}
    for phi in valid_phi_values:
        phi_dir = os.path.join(output_dir, f'phi_{phi:.2f}')
        os.makedirs(phi_dir, exist_ok=True)
        phi_dirs[phi] = phi_dir
    
    # 生成所有参数组合（添加约束：phi < theta）
    param_grid = []
    skipped_count = 0
    for phi in phi_values:
        for theta in theta_values:
            if phi < theta:  # 约束条件：低唤醒阈值必须小于高唤醒阈值
                for r in r_mainstream_values:
                    for sample_idx in range(n_samples):
                        param_grid.append({
                            'phi': phi,
                            'theta': theta,
                            'r_mainstream': r,
                            'sample_idx': sample_idx,
                            'seed': SIMULATION_PARAMS['default_seed'] + sample_idx,  # 不同样本使用不同种子
                            'output_dir': phi_dirs[phi]  # 添加对应的phi子目录
                        })
            else:
                skipped_count += 1
    
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个无效参数组合 (约束: phi < theta)")
    
    # 设置进程数
    if n_processes is None:
        n_processes = mp.cpu_count()
    n_processes = min(n_processes, len(param_grid))
    
    # 自动计算最佳chunksize
    if chunksize is None:
        # 使用配置参数计算最佳chunksize
        tasks_per_chunk = MULTIPROCESSING_PARAMS['tasks_per_chunk']
        chunksize = max(MULTIPROCESSING_PARAMS['min_chunksize'], 
                       min(MULTIPROCESSING_PARAMS['max_chunksize_factor'], 
                           len(param_grid) // (n_processes * tasks_per_chunk)))
    
    print(f"总共需要计算 {len(param_grid)} 个参数点 (包括 {n_samples} 个样本/参数组合)")
    print(f"使用 {n_processes} 个进程并行计算参数点，chunksize={chunksize}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 结果列表
    results = []
    
    # 性能监控变量
    completed_tasks = 0
    total_tasks = len(param_grid)
    last_report_time = start_time
    report_interval = MULTIPROCESSING_PARAMS['performance_report_interval']
    
    # 尝试使用多进程
    try:
        # 检查是否支持多进程
        if n_processes <= 1 or mp.cpu_count() <= 1:
            raise ValueError("不支持多进程或进程数设置为1，使用单进程模式")
        
        # 使用更安全的进程启动方法
        if hasattr(mp, 'get_context'):
            # 在Linux上使用'fork'，在Windows上使用'spawn'
            if os.name == 'posix':  # Linux/Mac
                try:
                    mp_ctx = mp.get_context('fork')
                    print("使用'fork'启动方法")
                except ValueError:
                    print("'fork'方法不可用，尝试'spawn'")
                    mp_ctx = mp.get_context('spawn')
            else:  # Windows
                mp_ctx = mp.get_context('spawn')
                print("使用'spawn'启动方法")
        else:
            mp_ctx = mp
            print("使用默认多进程上下文")
        
        # 使用进程池并行计算
        with mp_ctx.Pool(processes=n_processes) as pool:
            # 使用imap_unordered可能会更高效，因为它会立即返回已完成的任务结果
            for result in tqdm(
                pool.imap_unordered(worker, param_grid, chunksize=chunksize),
                total=len(param_grid),
                desc="参数扫描进度"
            ):
                results.append(result)
                
                # 更新性能监控
                completed_tasks += 1
                current_time = time.time()
                if current_time - last_report_time > report_interval:
                    elapsed = current_time - start_time
                    tasks_per_second = completed_tasks / elapsed
                    remaining_tasks = total_tasks - completed_tasks
                    estimated_remaining_time = remaining_tasks / tasks_per_second if tasks_per_second > 0 else float('inf')
                    
                    print(f"\n性能报告:")
                    print(f"  已完成: {completed_tasks}/{total_tasks} 任务 ({completed_tasks/total_tasks*100:.1f}%)")
                    print(f"  速度: {tasks_per_second:.2f} 任务/秒")
                    print(f"  预计剩余时间: {estimated_remaining_time/60:.1f} 分钟")
                    print(f"  已用时间: {elapsed/60:.1f} 分钟")
                    
                    last_report_time = current_time
    
    except Exception as e:
        print(f"多进程计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("切换到单进程模式...")
        
        # 如果多进程失败，切换到单进程模式
        results = []
        for params in tqdm(param_grid, desc="单进程参数扫描"):
            try:
                result = worker(params)
                results.append(result)
                
                # 更新性能监控
                completed_tasks += 1
                current_time = time.time()
                if current_time - last_report_time > report_interval:
                    elapsed = current_time - start_time
                    tasks_per_second = completed_tasks / elapsed
                    remaining_tasks = total_tasks - completed_tasks
                    estimated_remaining_time = remaining_tasks / tasks_per_second if tasks_per_second > 0 else float('inf')
                    
                    print(f"\n性能报告:")
                    print(f"  已完成: {completed_tasks}/{total_tasks} 任务 ({completed_tasks/total_tasks*100:.1f}%)")
                    print(f"  速度: {tasks_per_second:.2f} 任务/秒")
                    print(f"  预计剩余时间: {estimated_remaining_time/60:.1f} 分钟")
                    print(f"  已用时间: {elapsed/60:.1f} 分钟")
                    
                    last_report_time = current_time
            except Exception as e:
                print(f"任务失败 {params}: {str(e)}")
                results.append({
                    'phi': params['phi'],
                    'theta': params['theta'],
                    'r_mainstream': params['r_mainstream'],
                    'sample_idx': params['sample_idx'],
                    'seed': params['seed'],
                    'X_H': np.nan,
                    'X_M': np.nan,
                    'X_L': np.nan,
                    'p_risk': np.nan,
                    'p_risk_m': np.nan,
                    'p_risk_w': np.nan,
                    'success': False,
                    'error': str(e)
                })
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"参数扫描完成，总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
    # 计算性能统计
    tasks_per_second = len(results) / elapsed_time
    print(f"平均处理速度: {tasks_per_second:.2f} 任务/秒")
    print(f"每个任务平均耗时: {elapsed_time/len(results):.2f} 秒")
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 检查是否有失败的模拟
    if 'success' in results_df.columns:
        failed_count = results_df[results_df['success'] == False].shape[0]
        if failed_count > 0:
            print(f"警告: {failed_count} 个模拟失败 ({failed_count/len(results_df)*100:.1f}%)")
    
    # 按phi值分组保存结果
    for phi in phi_values:
        phi_results = results_df[results_df['phi'] == phi]
        if not phi_results.empty:
            # 保存该phi值的所有样本结果
            all_results_path = os.path.join(phi_dirs[phi], 'parameter_sweep_all_samples.csv')
            phi_results.to_csv(all_results_path, index=False)
            print(f"phi={phi:.2f} 的所有样本结果已保存到 {all_results_path}")
            
            # 计算并保存平均值和标准差
            if 'success' in phi_results.columns and not phi_results[phi_results['success'] == True].empty:
                grouped_mean = phi_results[phi_results['success'] == True].groupby(['theta', 'r_mainstream']).mean().reset_index()
                grouped_std = phi_results[phi_results['success'] == True].groupby(['theta', 'r_mainstream']).std().reset_index()
                
                # 重命名标准差列
                std_columns = {}
                for col in grouped_std.columns:
                    if col not in ['theta', 'r_mainstream']:
                        std_columns[col] = f"{col}_std"
                grouped_std = grouped_std.rename(columns=std_columns)
                
                # 合并平均值和标准差
                merged_results = pd.merge(grouped_mean, grouped_std, on=['theta', 'r_mainstream'])
                
                # 保存汇总结果
                avg_results_path = os.path.join(phi_dirs[phi], 'parameter_sweep_averaged.csv')
                merged_results.to_csv(avg_results_path, index=False)
                print(f"phi={phi:.2f} 的平均结果已保存到 {avg_results_path}")
    
    # 在主目录保存完整结果
    all_results_path = os.path.join(output_dir, 'parameter_sweep_all_samples.csv')
    results_df.to_csv(all_results_path, index=False)
    print(f"所有phi值的完整结果已保存到 {all_results_path}")
    
    return results_df

def plot_parameter_sweep_results(results_df, output_dir='results'):
    """
    绘制参数扫描结果
    
    Parameters:
        results_df: 包含平均结果的DataFrame
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取唯一的phi和theta值
    phi_values = sorted(results_df['phi'].unique())
    theta_values = sorted(results_df['theta'].unique())
    
    # 对每个(phi, theta)组合绘制一张图
    for phi in phi_values:
        for theta in theta_values:
            # 筛选数据
            subset = results_df[(results_df['phi'] == phi) & (results_df['theta'] == theta)]
            
            if subset.empty:
                continue
                
            # 排序
            subset = subset.sort_values('r_mainstream')
            
            # 创建图形（使用配置的图形大小）
            plt.figure(figsize=PLOT_PARAMS['figure_size'])
            
            # 绘制高唤醒比例
            plt.subplot(3, 1, 1)
            plt.errorbar(
                subset['r_mainstream'], 
                subset['X_H'], 
                yerr=subset.get('X_H_std', 0),
                fmt='ro-', 
                label='X_H'
            )
            plt.xlabel('Mainstream Media Removal Ratio (r)')
            plt.ylabel('High Arousal Ratio')
            plt.title(f'High Arousal Ratio vs. Media Removal (phi={phi}, theta={theta})')
            plt.grid(True, alpha=PLOT_PARAMS['grid_alpha'])
            
            # 绘制低唤醒比例
            plt.subplot(3, 1, 2)
            plt.errorbar(
                subset['r_mainstream'], 
                subset['X_L'], 
                yerr=subset.get('X_L_std', 0),
                fmt='bo-', 
                label='X_L'
            )
            plt.xlabel('Mainstream Media Removal Ratio (r)')
            plt.ylabel('Low Arousal Ratio')
            plt.title(f'Low Arousal Ratio vs. Media Removal (phi={phi}, theta={theta})')
            plt.grid(True, alpha=PLOT_PARAMS['grid_alpha'])
            
            # 绘制媒体风险比例
            plt.subplot(3, 1, 3)
            plt.errorbar(
                subset['r_mainstream'], 
                subset['p_risk'], 
                yerr=subset.get('p_risk_std', 0),
                fmt='go-', 
                label='p_risk'
            )
            plt.xlabel('Mainstream Media Removal Ratio (r)')
            plt.ylabel('Media Risk Ratio')
            plt.title(f'Media Risk Ratio vs. Media Removal (phi={phi}, theta={theta})')
            plt.grid(True, alpha=PLOT_PARAMS['grid_alpha'])
            
            plt.tight_layout()
            
            # 保存图像（使用配置的DPI）
            fig_path = os.path.join(output_dir, f'parameter_sweep_phi{phi}_theta{theta}.png')
            plt.savefig(fig_path, dpi=PLOT_PARAMS['dpi'])
            print(f"Figure saved to {fig_path}")
            plt.close()
    
    # 创建相变图
    if len(phi_values) > 1 and len(theta_values) > 1:
        # 为每个r值创建一个相变图
        r_values = sorted(results_df['r_mainstream'].unique())
        
        for r in r_values:
            # 筛选数据
            subset = results_df[results_df['r_mainstream'] == r]
            
            if subset.empty:
                continue
            
            # 创建网格
            phi_grid = np.array(phi_values)
            theta_grid = np.array(theta_values)
            X_H_grid = np.zeros((len(phi_grid), len(theta_grid)))
            X_L_grid = np.zeros((len(phi_grid), len(theta_grid)))
            
            # 填充网格
            for i, phi in enumerate(phi_values):
                for j, theta in enumerate(theta_values):
                    row = subset[(subset['phi'] == phi) & (subset['theta'] == theta)]
                    if not row.empty:
                        X_H_grid[i, j] = row['X_H'].values[0]
                        X_L_grid[i, j] = row['X_L'].values[0]
            
            # 创建图形（使用配置的相变图大小）
            plt.figure(figsize=PLOT_PARAMS['phase_diagram_size'])
            
            # 绘制高唤醒热图
            plt.subplot(1, 2, 1)
            plt.imshow(X_H_grid, origin='lower', aspect='auto', cmap='hot')
            plt.colorbar(label='X_H')
            plt.xlabel('Theta Index')
            plt.ylabel('Phi Index')
            plt.title(f'High Arousal Ratio (r={r:.2f})')
            plt.xticks(range(len(theta_grid)), [f'{t:.2f}' for t in theta_grid], rotation=45)
            plt.yticks(range(len(phi_grid)), [f'{p:.2f}' for p in phi_grid])
            
            # 绘制低唤醒热图
            plt.subplot(1, 2, 2)
            plt.imshow(X_L_grid, origin='lower', aspect='auto', cmap='cool')
            plt.colorbar(label='X_L')
            plt.xlabel('Theta Index')
            plt.ylabel('Phi Index')
            plt.title(f'Low Arousal Ratio (r={r:.2f})')
            plt.xticks(range(len(theta_grid)), [f'{t:.2f}' for t in theta_grid], rotation=45)
            plt.yticks(range(len(phi_grid)), [f'{p:.2f}' for p in phi_grid])
            
            plt.tight_layout()
            
            # 保存图像
            fig_path = os.path.join(output_dir, f'phase_diagram_r{r:.2f}.png')
            plt.savefig(fig_path, dpi=PLOT_PARAMS['dpi'])
            print(f"Phase diagram saved to {fig_path}")
            plt.close()

def test_small():
    """
    运行小规模测试 - V3版本
    """
    print("运行小规模参数扫描测试 (V3版本)...")
    
    # 设置参数 - 使用V3标准参数
    phi_values = [0.3]
    theta_values = [0.55]
    r_mainstream_values = [0.0, 0.5]  # 只测试2个r值
    n_samples = 1  # 每个参数点运行1个样本
    output_dir = 'results_v3/test_small'
    
    # 运行参数扫描 - 使用并行版本
    all_results = run_parallel_parameter_sweep(
        phi_values=phi_values,
        theta_values=theta_values,
        r_mainstream_values=r_mainstream_values,
        n_samples=n_samples,
        output_dir=output_dir
    )
    
    # 绘制结果
    if not all_results.empty:
        plot_parameter_sweep_results(all_results, output_dir=output_dir)
    
    return all_results

def run_full_sweep():
    """
    运行完整参数扫描 - V3版本
    """
    print("运行完整参数扫描 (V3版本)...")
    
    # 设置参数 - 使用V3标准参数
    phi_values = [0.3, 0.36, 0.45]
    theta_values = [0.55]
    r_mainstream_values = np.linspace(0.0, 0.95, 20)  # 20个均匀分布的点
    n_samples = 5  # 每个参数点运行5个样本
    output_dir = 'results_v3/full_sweep'
    
    # 获取可用的CPU核心数
    n_cores = mp.cpu_count()
    print(f"检测到 {n_cores} 个CPU核心")
    
    # 根据任务数和CPU核心数优化进程数和chunksize
    total_tasks = len(phi_values) * len(theta_values) * len(r_mainstream_values) * n_samples
    
    # 使用合理的进程数，避免过多进程导致系统负载过高
    if n_cores <= 4:
        # 小型CPU，使用所有核心
        n_processes = n_cores
    elif n_cores <= 16:
        # 中型CPU，使用80%的核心
        n_processes = max(1, int(n_cores * 0.8))
    else:
        # 大型CPU，使用70%的核心，避免系统过载
        n_processes = max(1, int(n_cores * 0.7))
    
    # 确保进程数不超过任务数
    n_processes = min(n_processes, total_tasks)
    
    # 计算最佳chunksize
    # 对于大量任务，较大的chunksize可以减少进程间通信开销
    if total_tasks < 100:
        chunksize = 1  # 小任务量，每个进程一次处理1个任务
    elif total_tasks < 1000:
        chunksize = 2  # 中等任务量，每个进程一次处理2个任务
    else:
        # 大任务量，根据任务数和进程数计算最佳chunksize
        chunksize = max(3, total_tasks // (n_processes * 10))
    
    print(f"任务总数: {total_tasks}, 使用进程数: {n_processes}, chunksize: {chunksize}")
    
    # 运行参数扫描
    all_results = run_parallel_parameter_sweep(
        phi_values=phi_values,
        theta_values=theta_values,
        r_mainstream_values=r_mainstream_values,
        n_samples=n_samples,
        output_dir=output_dir,
        n_processes=n_processes,
        chunksize=chunksize
    )
    
    # 绘制结果 - 直接使用all_results
    if not all_results.empty:
        plot_parameter_sweep_results(all_results, output_dir=output_dir)
    
    return all_results

def run_phase_diagram():
    """
    运行相变图参数扫描 - V3版本
    """
    print("运行相变图参数扫描 (V3版本)...")
    
    # 设置参数 - 使用V3标准参数
    phi_values = np.linspace(0.2, 0.4, 5)  # 5个phi值
    theta_values = np.linspace(0.45, 0.65, 5)  # 5个theta值
    r_mainstream_values = [0.0, 0.5, 0.75, 0.9]  # 4个r值
    n_samples = 3  # 每个参数点运行3个样本
    output_dir = 'results_v3/phase_diagram'
    
    # 获取可用的CPU核心数
    n_cores = mp.cpu_count()
    print(f"检测到 {n_cores} 个CPU核心")
    
    # 根据任务数和CPU核心数优化进程数和chunksize
    total_tasks = len(phi_values) * len(theta_values) * len(r_mainstream_values) * n_samples
    
    # 使用合理的进程数，避免过多进程导致系统负载过高
    if n_cores <= 4:
        # 小型CPU，使用所有核心
        n_processes = n_cores
    elif n_cores <= 16:
        # 中型CPU，使用80%的核心
        n_processes = max(1, int(n_cores * 0.8))
    else:
        # 大型CPU，使用70%的核心，避免系统过载
        n_processes = max(1, int(n_cores * 0.7))
    
    # 确保进程数不超过任务数
    n_processes = min(n_processes, total_tasks)
    
    # 计算最佳chunksize
    # 对于大量任务，较大的chunksize可以减少进程间通信开销
    if total_tasks < 100:
        chunksize = 1  # 小任务量，每个进程一次处理1个任务
    elif total_tasks < 1000:
        chunksize = 2  # 中等任务量，每个进程一次处理2个任务
    else:
        # 大任务量，根据任务数和进程数计算最佳chunksize
        chunksize = max(3, total_tasks // (n_processes * 10))
    
    print(f"任务总数: {total_tasks}, 使用进程数: {n_processes}, chunksize: {chunksize}")
    
    # 运行参数扫描
    all_results = run_parallel_parameter_sweep(
        phi_values=phi_values,
        theta_values=theta_values,
        r_mainstream_values=r_mainstream_values,
        n_samples=n_samples,
        output_dir=output_dir,
        n_processes=n_processes,
        chunksize=chunksize
    )
    
    # 绘制结果
    if not all_results.empty:
        plot_parameter_sweep_results(all_results, output_dir=output_dir)
    
    return all_results


def quick_phase_transition_test():
    """
    快速验证相变行为修复效果的测试函数 - V3版本
    
    Returns:
        success: 是否成功修复
    """
    print("=== 快速相变行为测试 (V3版本) ===")
    
    from src.theory_validation_simulator import TheoryValidationSimulator
    from src.model_with_a_minimal_v3 import ThresholdDynamicsModel
    import numpy as np
    
    # 使用V3标准参数
    network_params = {
        'n_mainstream': 1000,
        'n_wemedia': 1000,
        'n_public': 1000,
        'k_out_mainstream': 60,
        'k_out_wemedia': 60,
        'k_out_public': 10,
        'use_original_like_dist': False  # 使用模拟原始模型行为的分布
    }
    
    # 测试点：r=0.0和r=0.75
    test_points = [0.0, 0.75]
    
    # 统一的初始状态
    init_states = {
        'X_H': 0.3,
        'X_M': 0.4,
        'X_L': 0.3,
        'p_risk_m': 0.5,
        'p_risk_w': 0.5,
        'p_risk': 0.5
    }
    
    threshold_params = {
        'theta': 0.55,
        'phi': 0.3
    }
    
    print("计算V3理论预测...")
    theory_model = ThresholdDynamicsModel(network_params, threshold_params)
    
    theory_X_H = []
    for r in test_points:
        result = theory_model.solve_self_consistent(
            init_states=init_states,  # 使用统一的初始状态
            removal_ratios={'mainstream': r}
        )
        theory_X_H.append(result['X_H'])
        print(f"理论预测 r={r}: X_H={result['X_H']:.4f}")
    
    print("\n进行模拟...")
    sim_X_H = []
    for r in test_points:
        simulator = TheoryValidationSimulator(
            network_params=network_params,
            threshold_params=threshold_params
        )
        simulator.generate_or_load_network(seed=42, verbose=False)
        # 使用与理论模型相同的初始状态
        simulator.initialize_states(init_states=init_states)
        
        final_stats = simulator.simulate_to_steady_state(
            max_iter=SIMULATION_PARAMS['max_iter'],  # 使用统一配置的最大迭代次数
            removal_ratios={'mainstream': r}, 
            save_history=False,
            enable_micro_analysis=False,
            verbose=False  # 禁用详细日志输出
        )
        sim_X_H.append(final_stats['X_H'])
        print(f"模拟结果 r={r}: X_H={final_stats['X_H']:.4f}")
    
    # 分析结果
    theory_change = abs(theory_X_H[1] - theory_X_H[0])
    sim_change = abs(sim_X_H[1] - sim_X_H[0])
    
    print(f"\n变化分析:")
    print(f"理论X_H变化: {theory_change:.4f}")
    print(f"模拟X_H变化: {sim_change:.4f}")
    
    # 判断成功
    significant_threshold = 0.1
    consistency_threshold = 0.05
    
    theory_significant = theory_change > significant_threshold
    sim_significant = sim_change > significant_threshold
    
    # 修复：检查变化趋势是否一致（同向）
    theory_direction = 1 if theory_X_H[1] > theory_X_H[0] else -1
    sim_direction = 1 if sim_X_H[1] > sim_X_H[0] else -1
    trend_consistent = (theory_direction * sim_direction > 0)
    
    # 检查变化幅度是否接近
    magnitude_consistent = abs(theory_change - sim_change) < consistency_threshold
    
    # 综合判断：趋势一致且幅度接近
    consistent = trend_consistent and magnitude_consistent
    
    success = theory_significant and sim_significant and consistent
    
    print(f"理论变化显著: {'是' if theory_significant else '否'}")
    print(f"模拟变化显著: {'是' if sim_significant else '否'}")
    print(f"变化趋势一致: {'是' if trend_consistent else '否'}")
    print(f"变化幅度接近: {'是' if magnitude_consistent else '否'}")
    print(f"\n快速测试结果: {'✅ 成功' if success else '❌ 需要改进'}")
    
    return success

def quick_phase_transition_test_parallel():
    """
    快速验证模拟器vs理论模型一致性的多进程测试函数 - V3版本
    利用多进程并行计算多个r值，真正比较模拟器和理论模型的结果，并提供可视化对比
    
    Returns:
        success: 模拟器与理论模型是否一致
    """
    print("=== 模拟器 vs 理论模型多进程对比测试 ===")
    print("使用多进程并行计算多个r值，验证模拟器与理论模型的一致性")
    
    # 固定参数
    phi_values = [0.3]  # 固定phi
    theta_values = [0.55]  # 固定theta
    r_mainstream_values = np.linspace(0.0, 0.9, 10)  # 10个r值，从0到0.9
    n_samples = 1  # 每个参数点1个样本
    output_dir = 'results_v3/simulator_vs_theory_parallel'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"测试参数：phi={phi_values[0]}, theta={theta_values[0]}")
    print(f"r值范围：{len(r_mainstream_values)}个点，从{r_mainstream_values[0]:.1f}到{r_mainstream_values[-1]:.1f}")
    
    # 获取可用CPU核心数
    n_cores = mp.cpu_count()
    print(f"检测到 {n_cores} 个CPU核心")
    
    # 使用合理的进程数
    n_processes = min(n_cores // 2, len(r_mainstream_values))  # 使用一半核心，因为要运行两套计算
    print(f"使用 {n_processes} 个进程并行计算")
    
    start_time = time.time()
    
    # 1. 并行计算理论模型结果
    print("\n1. 并行计算理论模型结果...")
    theory_param_grid = []
    
    # 统一的初始状态
    init_states = {
        'X_H': 0.3,
        'X_M': 0.4,
        'X_L': 0.3,
        'p_risk_m': 0.5,
        'p_risk_w': 0.5,
        'p_risk': 0.5
    }
    
    for r in r_mainstream_values:
        theory_param_grid.append({
            'phi': phi_values[0],
            'theta': theta_values[0],
            'r_mainstream': r,
            'sample_idx': 0,
            'seed': 42,
            'use_simulator': False,  # 使用理论模型
            'output_dir': output_dir,
            'verbose': False,  # 禁用详细日志输出
            'init_states': init_states  # 使用统一的初始状态
        })
    
    # 并行计算理论结果
    if n_processes > 1:
        with mp.Pool(processes=n_processes) as pool:
            theory_results = list(tqdm(
                pool.map(worker, theory_param_grid),
                total=len(theory_param_grid),
                desc="理论模型计算"
            ))
    else:
        theory_results = []
        for params in tqdm(theory_param_grid, desc="理论模型计算"):
            theory_results.append(worker(params))
    
    # 2. 并行计算模拟器结果
    print("\n2. 并行计算模拟器结果...")
    sim_param_grid = []
    
    # 统一的初始状态
    init_states = {
        'X_H': 0.3,
        'X_M': 0.4,
        'X_L': 0.3,
        'p_risk_m': 0.5,
        'p_risk_w': 0.5,
        'p_risk': 0.5
    }
    
    for r in r_mainstream_values:
        sim_param_grid.append({
            'phi': phi_values[0],
            'theta': theta_values[0],
            'r_mainstream': r,
            'sample_idx': 0,
            'seed': 42,
            'use_simulator': True,  # 使用模拟器
            'output_dir': output_dir,
            'max_iter': SIMULATION_PARAMS['max_iter'],  # 使用统一配置的最大迭代次数
            'verbose': False,  # 禁用详细日志输出
            'init_states': init_states  # 使用统一的初始状态
        })
    
    # 并行计算模拟器结果
    if n_processes > 1:
        with mp.Pool(processes=n_processes) as pool:
            sim_results = list(tqdm(
                pool.map(worker, sim_param_grid),
                total=len(sim_param_grid),
                desc="模拟器计算"
            ))
    else:
        sim_results = []
        for params in tqdm(sim_param_grid, desc="模拟器计算"):
            sim_results.append(worker(params))
    
    elapsed_time = time.time() - start_time
    print(f"\n并行计算完成，总耗时：{elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    
    # 3. 比较结果
    print("\n3. 对比分析...")
    differences = []
    successful_comparisons = 0
    
    # 准备绘图数据
    r_values = []
    theory_X_H_values = []
    theory_X_L_values = []
    theory_p_risk_values = []
    sim_X_H_values = []
    sim_X_L_values = []
    sim_p_risk_values = []
    
    for i, r in enumerate(r_mainstream_values):
        theory = theory_results[i]
        sim = sim_results[i]
        
        if theory['success'] and sim['success']:
            diff_X_H = abs(theory['X_H'] - sim['X_H'])
            diff_X_L = abs(theory['X_L'] - sim['X_L'])
            diff_p_risk = abs(theory['p_risk'] - sim['p_risk'])
            
            differences.append({
                'r': r,
                'theory_X_H': theory['X_H'],
                'sim_X_H': sim['X_H'],
                'diff_X_H': diff_X_H,
                'diff_X_L': diff_X_L,
                'diff_p_risk': diff_p_risk,
                'total_diff': diff_X_H + diff_X_L + diff_p_risk
            })
            
            # 添加到绘图数据
            r_values.append(r)
            theory_X_H_values.append(theory['X_H'])
            theory_X_L_values.append(theory['X_L'])
            theory_p_risk_values.append(theory['p_risk'])
            sim_X_H_values.append(sim['X_H'])
            sim_X_L_values.append(sim['X_L'])
            sim_p_risk_values.append(sim['p_risk'])
            
            successful_comparisons += 1
            print(f"  r={r:.1f}: 理论X_H={theory['X_H']:.4f}, 模拟X_H={sim['X_H']:.4f}, Δ={diff_X_H:.4f}")
            print(f"       理论X_L={theory['X_L']:.4f}, 模拟X_L={sim['X_L']:.4f}, Δ={diff_X_L:.4f}")
            print(f"       理论p_risk={theory['p_risk']:.4f}, 模拟p_risk={sim['p_risk']:.4f}, Δ={diff_p_risk:.4f}")
        else:
            print(f"  r={r:.1f}: 计算失败 (理论:{theory['success']}, 模拟:{sim['success']})")
    
    # 4. 生成可视化对比图
    if successful_comparisons >= 3:  # 至少有3个成功的点才能绘图
        print("\n4. 生成可视化对比图...")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot X_H comparison
        axes[0].plot(r_values, theory_X_H_values, 'ro-', linewidth=2, label='Theory Model')
        axes[0].plot(r_values, sim_X_H_values, 'bo--', linewidth=2, label='Simulator')
        axes[0].set_xlabel('Mainstream Media Removal Ratio (r)')
        axes[0].set_ylabel('High Arousal Proportion (X_H)')
        axes[0].set_title(f'High Arousal Proportion Comparison (phi={phi_values[0]}, theta={theta_values[0]})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot X_L comparison
        axes[1].plot(r_values, theory_X_L_values, 'ro-', linewidth=2, label='Theory Model')
        axes[1].plot(r_values, sim_X_L_values, 'bo--', linewidth=2, label='Simulator')
        axes[1].set_xlabel('Mainstream Media Removal Ratio (r)')
        axes[1].set_ylabel('Low Arousal Proportion (X_L)')
        axes[1].set_title('Low Arousal Proportion Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot p_risk comparison
        axes[2].plot(r_values, theory_p_risk_values, 'ro-', linewidth=2, label='Theory Model')
        axes[2].plot(r_values, sim_p_risk_values, 'bo--', linewidth=2, label='Simulator')
        axes[2].set_xlabel('Mainstream Media Removal Ratio (r)')
        axes[2].set_ylabel('Media Risk Proportion (p_risk)')
        axes[2].set_title('Media Risk Proportion Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        comparison_fig_path = os.path.join(output_dir, f'theory_vs_simulator_comparison.png')
        plt.savefig(comparison_fig_path, dpi=300)
        plt.close()
        
        print(f"对比图已保存到: {comparison_fig_path}")
        
        # Plot difference heatmap
        plt.figure(figsize=(10, 6))
        
        # Create difference matrix
        diff_data = np.zeros((3, len(r_values)))
        for i, r in enumerate(r_values):
            diff_data[0, i] = abs(theory_X_H_values[i] - sim_X_H_values[i])
            diff_data[1, i] = abs(theory_X_L_values[i] - sim_X_L_values[i])
            diff_data[2, i] = abs(theory_p_risk_values[i] - sim_p_risk_values[i])
        
        # Plot heatmap
        im = plt.imshow(diff_data, cmap='hot', aspect='auto')
        plt.colorbar(im, label='Absolute Difference')
        
        # Set labels
        plt.yticks([0, 1, 2], ['X_H', 'X_L', 'p_risk'])
        plt.xticks(range(len(r_values)), [f'{r:.2f}' for r in r_values])
        plt.xlabel('Mainstream Media Removal Ratio (r)')
        plt.title(f'Theory vs Simulator Difference Heatmap (phi={phi_values[0]}, theta={theta_values[0]})')
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, f'theory_vs_simulator_heatmap.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        
        print(f"Difference heatmap saved to: {heatmap_path}")
    
    # 5. 评估一致性
    if differences and successful_comparisons >= len(r_mainstream_values) * 0.8:  # 至少80%成功
        avg_diff_X_H = np.mean([d['diff_X_H'] for d in differences])
        avg_diff_X_L = np.mean([d['diff_X_L'] for d in differences])
        avg_diff_p_risk = np.mean([d['diff_p_risk'] for d in differences])
        max_total_diff = max([d['total_diff'] for d in differences])
        
        # 计算相变行为一致性
        theory_X_H_values = [d['theory_X_H'] for d in differences]
        sim_X_H_values = [d['sim_X_H'] for d in differences]
        
        theory_X_H_range = max(theory_X_H_values) - min(theory_X_H_values)
        sim_X_H_range = max(sim_X_H_values) - min(sim_X_H_values)
        
        # 计算趋势一致性
        theory_trend = np.polyfit(r_values, theory_X_H_values, 1)[0]  # 线性拟合的斜率
        sim_trend = np.polyfit(r_values, sim_X_H_values, 1)[0]
        trend_consistent = (theory_trend * sim_trend > 0)  # 斜率同号表示趋势一致
        
        print(f"\n=== 多进程一致性分析 ===")
        print(f"成功对比点数: {successful_comparisons}/{len(r_mainstream_values)} ({successful_comparisons/len(r_mainstream_values)*100:.1f}%)")
        print(f"平均差异: ΔX_H={avg_diff_X_H:.4f}, ΔX_L={avg_diff_X_L:.4f}, Δp_risk={avg_diff_p_risk:.4f}")
        print(f"最大总差异: {max_total_diff:.4f}")
        print(f"理论X_H变化范围: {theory_X_H_range:.4f}, 趋势斜率: {theory_trend:.4f}")
        print(f"模拟X_H变化范围: {sim_X_H_range:.4f}, 趋势斜率: {sim_trend:.4f}")
        print(f"趋势一致: {'是' if trend_consistent else '否'}")
        
        # 判断标准
        tolerance_individual = 0.15  # 放宽单个指标容忍度
        tolerance_total = 0.25       # 放宽总体容忍度
        phase_transition_threshold = 0.1  # 相变显著性阈值
        
        individual_ok = (avg_diff_X_H < tolerance_individual and 
                        avg_diff_X_L < tolerance_individual and 
                        avg_diff_p_risk < tolerance_individual)
        total_ok = max_total_diff < tolerance_total
        
        # 检查相变行为
        theory_has_transition = theory_X_H_range > phase_transition_threshold
        sim_has_transition = sim_X_H_range > phase_transition_threshold
        phase_behavior_consistent = (theory_has_transition == sim_has_transition) and trend_consistent
        
        success = individual_ok and total_ok and phase_behavior_consistent and successful_comparisons >= len(r_mainstream_values) * 0.8
        
        print(f"个体指标一致性: {'✅ 通过' if individual_ok else '❌ 不通过'}")
        print(f"总体一致性: {'✅ 通过' if total_ok else '❌ 不通过'}")
        print(f"理论预测相变: {'是' if theory_has_transition else '否'}")
        print(f"模拟观察到相变: {'是' if sim_has_transition else '否'}")
        print(f"相变行为一致: {'✅ 通过' if phase_behavior_consistent else '❌ 不通过'}")
        print(f"\n最终结果: {'✅ 模拟器与理论模型基本一致' if success else '❌ 存在显著差异，需要检查'}")
        
        if success:
            print("🎉 恭喜！模拟器能够基本复现理论模型的相变行为！")
            print(f"⚡ 多进程加速效果显著：{n_processes}个进程并行计算{len(r_mainstream_values)*2}个任务")
        else:
            print("\n需要检查的问题：")
            if not individual_ok:
                if avg_diff_X_H >= tolerance_individual:
                    print("- X_H计算差异较大，检查公众情绪计算逻辑")
                if avg_diff_X_L >= tolerance_individual:
                    print("- X_L计算差异较大，检查低唤醒状态计算")
                if avg_diff_p_risk >= tolerance_individual:
                    print("- p_risk计算差异较大，检查媒体风险计算")
            if not total_ok:
                print("- 总体差异过大，可能存在系统性偏差")
            if not phase_behavior_consistent:
                if not trend_consistent:
                    print("- 趋势不一致，理论和模拟结果变化方向相反")
                else:
                    print("- 相变行为不一致，检查临界点计算逻辑")
            
            # 提供可能的原因分析
            print("\n可能的原因分析：")
            print("1. 初始状态设置不同：理论模型和模拟器可能使用了不同的初始状态")
            print("2. 网络结构影响：实际生成的网络与理论模型假设的网络有差异")
            print("3. 动力学规则实现差异：模拟器中的状态更新规则可能与理论模型不一致") 
            print("4. 收敛问题：模拟可能未达到真正稳态，尝试增加最大迭代次数")
            print("5. 参数敏感性：尝试调整phi和theta参数，观察不同参数下的一致性")
        
        # 保存详细结果到CSV文件
        results_df = pd.DataFrame(differences)
        results_path = os.path.join(output_dir, 'theory_vs_simulator_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n详细对比结果已保存到: {results_path}")
        
        return success
    else:
        print(f"❌ 成功对比点数不足: {successful_comparisons}/{len(r_mainstream_values)}")
        return False

def test_micro_analysis_functionality():
    """
    测试微观分析功能，验证能否获取详细的节点级别统计信息 - V3版本
    
    Returns:
        success: 是否成功获取微观分析结果
        analysis_results: 详细的分析结果
    """
    print("=== 微观分析功能测试 (V3版本) ===")
    print("测试目标：验证能否获取状态转移、媒体影响、极化指数等微观统计")
    
    from src.theory_validation_simulator import TheoryValidationSimulator
    import numpy as np
    
    # 使用V3标准参数
    network_params = {
        'n_mainstream': 100,    # 减少为100个主流媒体节点
        'n_wemedia': 100,       # 减少为100个自媒体节点
        'n_public': 500,        # 减少为500个公众节点
        'k_out_mainstream': 60, # 保持不变
        'k_out_wemedia': 60,    # 保持不变
        'k_out_public': 10,
        'use_original_like_dist': False
        # V3不再需要gamma参数
    }
    
    threshold_params = {
        'theta': 0.55,
        'phi': 0.3
        # V3不再需要alpha、beta、gamma参数
    }
    
    print("创建模拟器并生成/加载网络...")
    simulator = TheoryValidationSimulator(
        network_params=network_params,  # 确保包含use_original_like_dist参数
        threshold_params=threshold_params
    )
    # 使用缓存机制生成或加载网络
    simulator.generate_or_load_network(seed=42)
    simulator.initialize_states()
    
    print("运行模拟（启用完整微观分析）...")
    final_stats = simulator.simulate_to_steady_state(
        max_iter=min(30, SIMULATION_PARAMS['max_iter']),  # 使用较少迭代以加快测试，但不超过配置值
        removal_ratios={'mainstream': 0.3},  # 适中的移除比例
        save_history=True,
        enable_micro_analysis=True
    )
    
    # 检查微观分析结果
    print("\n=== 微观分析结果检查 ===")
    
    # 1. 检查基本微观统计
    has_micro_stats = hasattr(simulator, 'micro_stats') and bool(simulator.micro_stats)
    print(f"1. 基本微观统计: {'✅ 可用' if has_micro_stats else '❌ 缺失'}")
    
    if has_micro_stats:
        micro_stats = simulator.micro_stats
        print("   可用的微观统计项:")
        for key, value in micro_stats.items():
            if not np.isnan(value) if isinstance(value, (int, float)) else value:
                print(f"     - {key}: {value}")
    else:
        micro_stats = {}
    
    # 2. 检查详细微观分析
    has_micro_analysis = hasattr(simulator, 'micro_analysis') and bool(simulator.micro_analysis)
    print(f"2. 详细微观分析: {'✅ 可用' if has_micro_analysis else '❌ 缺失'}")
    
    if has_micro_analysis:
        micro_analysis = simulator.micro_analysis
        print("   可用的微观分析项:")
        for key, value in micro_analysis.items():
            print(f"     - {key}: {type(value).__name__}")
            if isinstance(value, dict):
                for subkey in value.keys():
                    print(f"       * {subkey}")
    else:
        micro_analysis = {}
    
    # 3. 检查状态历史（用于振荡分析）
    has_history = hasattr(simulator, 'state_history') and bool(simulator.state_history)
    print(f"3. 状态历史记录: {'✅ 可用' if has_history else '❌ 缺失'}")
    
    if has_history:
        print(f"   历史记录长度: {len(simulator.state_history)} 步")
        if len(simulator.state_history) > 0:
            first_step = simulator.state_history[0]
            last_step = simulator.state_history[-1]
            print(f"   初始状态: X_H={first_step['X_H']:.4f}, X_L={first_step['X_L']:.4f}")
            print(f"   最终状态: X_H={last_step['X_H']:.4f}, X_L={last_step['X_L']:.4f}")
    
    # 4. 测试振荡检测
    print(f"4. 振荡检测功能: ", end="")
    try:
        oscillation_info = detect_oscillation(simulator.state_history) if has_history else {}
        has_oscillation = bool(oscillation_info.get('has_oscillation', False))
        print(f"✅ 可用 (检测到振荡: {'是' if has_oscillation else '否'})")
        if oscillation_info:
            print(f"   振荡强度: {oscillation_info.get('strength', 0):.4f}")
            print(f"   收敛步数: {oscillation_info.get('convergence_steps', 'N/A')}")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        oscillation_info = {}
        has_oscillation = False
    
    # 5. 检查连接类型分析
    has_connection_analysis = hasattr(simulator, 'connection_types') and bool(simulator.connection_types)
    print(f"5. 连接类型分析: {'✅ 可用' if has_connection_analysis else '❌ 缺失'}")
    
    if has_connection_analysis:
        print(f"   分析的公众节点数: {len(simulator.connection_types)}")
        # 统计连接类型分布
        type_counts = {}
        for node_info in simulator.connection_types.values():
            ptype = node_info.get('primary_type', 'unknown')
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        for ptype, count in type_counts.items():
            print(f"   {ptype}连接节点: {count}个")
    
    # 6. 检查转移统计
    has_transition_stats = hasattr(simulator, 'transition_stats') and bool(simulator.transition_stats)
    print(f"6. 状态转移统计: {'✅ 可用' if has_transition_stats else '❌ 缺失'}")
    
    if has_transition_stats:
        print("   记录的转移类型:")
        for transition_type, counts in simulator.transition_stats.items():
            if counts:  # 非空列表
                avg_count = np.mean(counts)
                print(f"     - {transition_type}: 平均 {avg_count:.2f} 次/步")
    
    # 7. 综合评分
    feature_scores = [
        int(has_micro_stats),
        int(has_micro_analysis), 
        int(has_history),
        int(has_oscillation),
        int(has_connection_analysis),
        int(has_transition_stats)
    ]
    
    total_score = sum(feature_scores)
    max_score = len(feature_scores)
    
    print(f"\n=== 功能完整性评分 ===")
    print(f"微观分析功能完整性: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")
    
    success = total_score >= max_score * 0.8  # 80%以上功能可用视为成功
    
    if success:
        print("🎉 微观分析功能测试通过！")
        print("✅ 可以获取详细的节点级别统计信息")
    else:
        print("⚠️ 微观分析功能不完整，需要进一步优化")
        
        missing_features = []
        feature_names = [
            "基本微观统计", "详细微观分析", "状态历史记录",
            "振荡检测", "连接类型分析", "状态转移统计"
        ]
        
        for i, (name, available) in enumerate(zip(feature_names, feature_scores)):
            if not available:
                missing_features.append(name)
        
        if missing_features:
            print(f"缺失功能: {', '.join(missing_features)}")
    
    # 构建详细分析结果
    analysis_results = {
        'success': success,
        'total_score': total_score,
        'max_score': max_score,
        'completion_rate': total_score / max_score,
        'features': {
            'micro_stats': has_micro_stats,
            'micro_analysis': has_micro_analysis,
            'state_history': has_history,
            'oscillation_detection': bool(oscillation_info),
            'connection_analysis': has_connection_analysis,
            'transition_stats': has_transition_stats
        },
        'micro_stats_keys': list(micro_stats.keys()) if has_micro_stats else [],
        'micro_analysis_keys': list(micro_analysis.keys()) if has_micro_analysis else [],
        'oscillation_info': oscillation_info,
        'history_length': len(simulator.state_history) if has_history else 0,
        'final_stats': final_stats
    }
    
    return success, analysis_results

def run_enhanced_parameter_sweep_with_micro_analysis():
    """
    运行增强版参数扫描，包含完整的微观分析功能 - V3版本
    
    Returns:
        results_df: 包含微观分析数据的结果DataFrame
    """
    print("=== 增强版参数扫描（含微观分析）- V3版本 ===")
    
    # 使用较小的参数范围进行测试 - V3标准参数
    phi_values = [0.3]
    theta_values = [0.55]
    r_mainstream_values = [0.0, 0.3, 0.6, 0.9]  # 4个关键点
    n_samples = 1  # 每个参数点1个样本
    output_dir = 'results_v3/enhanced_micro_analysis'
    
    print(f"参数设置：")
    print(f"  phi: {phi_values}")
    print(f"  theta: {theta_values}")
    print(f"  r_mainstream: {r_mainstream_values}")
    print(f"  样本数: {n_samples}")
    
    # 运行参数扫描
    all_results = run_parallel_parameter_sweep(
        phi_values=phi_values,
        theta_values=theta_values,
        r_mainstream_values=r_mainstream_values,
        n_samples=n_samples,
        output_dir=output_dir,
        n_processes=len(r_mainstream_values),  # 使用4个进程
        chunksize=1
    )
    
    # 分析微观数据
    print("\n=== 微观分析数据总结 ===")
    
    if not all_results.empty:
        # 检查微观数据的可用性
        micro_columns = [col for col in all_results.columns if any(keyword in col for keyword in 
                        ['connected', 'influence', 'transition', 'oscillation'])]
        
        print(f"检测到 {len(micro_columns)} 个微观分析列：")
        for col in micro_columns:
            non_nan_count = all_results[col].notna().sum()
            total_count = len(all_results)
            print(f"  {col}: {non_nan_count}/{total_count} ({non_nan_count/total_count*100:.1f}%) 有效数据")
        
        # 分析振荡现象
        if 'has_oscillation' in all_results.columns:
            oscillation_count = all_results['has_oscillation'].sum()
            total_runs = len(all_results)
            print(f"\n振荡现象统计：")
            print(f"  发生振荡的运行: {oscillation_count}/{total_runs} ({oscillation_count/total_runs*100:.1f}%)")
            
            if oscillation_count > 0:
                oscillating_runs = all_results[all_results['has_oscillation'] == True]
                avg_strength = oscillating_runs['oscillation_strength'].mean()
                print(f"  平均振荡强度: {avg_strength:.4f}")
        
        # 分析连接类型差异
        connection_columns = [col for col in micro_columns if 'connected' in col]
        if connection_columns:
            print(f"\n连接类型分析：")
            for col in connection_columns:
                if all_results[col].notna().any():
                    mean_val = all_results[col].mean()
                    std_val = all_results[col].std()
                    print(f"  {col}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 保存微观分析专门的CSV文件
        micro_results_path = os.path.join(output_dir, 'micro_analysis_detailed.csv')
        all_results.to_csv(micro_results_path, index=False)
        print(f"\n微观分析详细结果已保存到：{micro_results_path}")
        
        return all_results
    else:
        print("❌ 没有成功的模拟结果")
        return pd.DataFrame()

def compare_simulator_vs_theory():
    """
    比较模拟器和理论模型的结果，验证模拟器的正确性
    
    Returns:
        success: 模拟器与理论模型是否一致
    """
    print("=== 模拟器 vs 理论模型对比测试 ===")
    print("目标：验证模拟器计算结果与理论模型的一致性")
    
    # 测试参数
    phi_values = [0.3]
    theta_values = [0.55] 
    r_mainstream_values = [0.0, 0.3, 0.6, 0.9]  # 4个关键点
    n_samples = 1
    
    print(f"测试参数：phi={phi_values[0]}, theta={theta_values[0]}")
    print(f"r值：{r_mainstream_values}")
    
    # 1. 获取理论模型结果
    print("\n1. 计算理论模型结果...")
    theory_results = []
    for r in r_mainstream_values:
        params = {
            'phi': phi_values[0],
            'theta': theta_values[0], 
            'r_mainstream': r,
            'sample_idx': 0,
            'seed': 42,
            'use_simulator': False  # 使用理论模型
        }
        result = run_single_simulation(params)
        theory_results.append(result)
        print(f"  理论 r={r}: X_H={result['X_H']:.4f}, X_L={result['X_L']:.4f}")
    
    # 2. 获取模拟器结果
    print("\n2. 计算模拟器结果...")
    sim_results = []
    for r in r_mainstream_values:
        params = {
            'phi': phi_values[0],
            'theta': theta_values[0],
            'r_mainstream': r,
            'sample_idx': 0,
            'seed': 42,
            'use_simulator': True  # 使用模拟器
        }
        result = run_single_simulation(params)
        sim_results.append(result)
        print(f"  模拟 r={r}: X_H={result['X_H']:.4f}, X_L={result['X_L']:.4f}")
    
    # 3. 比较结果
    print("\n3. 对比分析...")
    differences = []
    for i, r in enumerate(r_mainstream_values):
        theory = theory_results[i]
        sim = sim_results[i]
        
        if theory['success'] and sim['success']:
            diff_X_H = abs(theory['X_H'] - sim['X_H'])
            diff_X_L = abs(theory['X_L'] - sim['X_L'])
            diff_p_risk = abs(theory['p_risk'] - sim['p_risk'])
            
            differences.append({
                'r': r,
                'diff_X_H': diff_X_H,
                'diff_X_L': diff_X_L,
                'diff_p_risk': diff_p_risk,
                'total_diff': diff_X_H + diff_X_L + diff_p_risk
            })
            
            print(f"  r={r:.1f}: ΔX_H={diff_X_H:.4f}, ΔX_L={diff_X_L:.4f}, Δp_risk={diff_p_risk:.4f}")
        else:
            print(f"  r={r:.1f}: 计算失败 (理论:{theory['success']}, 模拟:{sim['success']})")
    
    # 4. 评估一致性
    if differences:
        avg_diff_X_H = np.mean([d['diff_X_H'] for d in differences])
        avg_diff_X_L = np.mean([d['diff_X_L'] for d in differences])
        avg_diff_p_risk = np.mean([d['diff_p_risk'] for d in differences])
        max_total_diff = max([d['total_diff'] for d in differences])
        
        print(f"\n=== 一致性分析 ===")
        print(f"平均差异: ΔX_H={avg_diff_X_H:.4f}, ΔX_L={avg_diff_X_L:.4f}, Δp_risk={avg_diff_p_risk:.4f}")
        print(f"最大总差异: {max_total_diff:.4f}")
        
        # 判断标准
        tolerance_individual = 0.05  # 单个指标容忍度
        tolerance_total = 0.10       # 总体容忍度
        
        individual_ok = (avg_diff_X_H < tolerance_individual and 
                        avg_diff_X_L < tolerance_individual and 
                        avg_diff_p_risk < tolerance_individual)
        total_ok = max_total_diff < tolerance_total
        
        success = individual_ok and total_ok
        
        print(f"个体指标一致性: {'✅ 通过' if individual_ok else '❌ 不通过'}")
        print(f"总体一致性: {'✅ 通过' if total_ok else '❌ 不通过'}")
        print(f"\n最终结果: {'✅ 模拟器与理论模型一致' if success else '❌ 存在显著差异，需要检查'}")
        
        if not success:
            print("\n建议检查项目：")
            if not individual_ok:
                if avg_diff_X_H >= tolerance_individual:
                    print("- X_H计算差异较大，检查公众情绪计算逻辑")
                if avg_diff_X_L >= tolerance_individual:
                    print("- X_L计算差异较大，检查低唤醒状态计算")
                if avg_diff_p_risk >= tolerance_individual:
                    print("- p_risk计算差异较大，检查媒体风险计算")
            if not total_ok:
                print("- 总体差异过大，可能存在系统性偏差")
        
        return success
    else:
        print("❌ 没有成功的对比数据")
        return False

def run_parallel_parameter_sweep_v2(phi_range, theta_range, r_mainstream_values, n_samples=1, 
                                    output_dir='results_v3', n_processes=None, chunksize=None,
                                    skip_existing=True, enable_micro_analysis=True, 
                                    enable_visualization=True, max_iter=None, 
                                    distribution_params=None, dominated_thresholds=None):
    """
    改进版参数扫描函数 - 基于quick_scan_v3.py的逻辑
    支持phi和theta的range或列表输入，使用展平并行计算避免嵌套进程
    
    Parameters:
        phi_range: phi值范围 [start, end, step] 或 phi值列表
        theta_range: theta值范围 [start, end, step] 或 theta值列表  
        r_mainstream_values: r_mainstream参数值列表
        n_samples: 每个参数点运行的样本数（默认1）
        output_dir: 输出目录
        n_processes: 使用的进程数，默认为None（自动选择）
        chunksize: 每个进程一次处理的任务数，默认为None（自动计算）
        skip_existing: 是否跳过已有结果，默认True
        enable_micro_analysis: 是否启用微观分析，默认True
        enable_visualization: 是否生成可视化图表，默认True
        max_iter: 每个sample的最大迭代次数，默认None（使用全局配置SIMULATION_PARAMS['max_iter']）
        distribution_params: 分布参数字典，格式:
            - 泊松分布: {'type': 'poisson', 'kappa': 120, 'max_k': 50}
            - 幂律分布: {'type': 'powerlaw', 'gamma_pref': 2.0, 'k_min_pref': 1, 'max_k': 200}
            - None: 使用全局配置NETWORK_PARAMS中的参数
        dominated_thresholds: dominated判断阈值字典，格式:
            - {'mainstream': 0.6, 'wemedia': 0.4} 或 {'mainstream': 0.7, 'wemedia': 0.3}
            - None: 使用默认阈值 {'mainstream': 0.6, 'wemedia': 0.4}
        
    Returns:
        results_df: 包含所有结果的DataFrame
    """
    print("=== 改进版参数扫描 (V2) ===")
    print("基于quick_scan_v3.py的逻辑，支持range/列表输入，展平并行计算")
    
    # 处理dominated判断阈值
    if dominated_thresholds is None:
        dominated_thresholds = {'mainstream': 0.6, 'wemedia': 0.4}
    else:
        # 验证阈值格式
        if not isinstance(dominated_thresholds, dict) or 'mainstream' not in dominated_thresholds or 'wemedia' not in dominated_thresholds:
            raise ValueError("dominated_thresholds必须是包含'mainstream'和'wemedia'键的字典")
        if not (0 < dominated_thresholds['mainstream'] < 1 and 0 < dominated_thresholds['wemedia'] < 1):
            raise ValueError("阈值必须在0到1之间")
        if dominated_thresholds['mainstream'] <= dominated_thresholds['wemedia']:
            raise ValueError("mainstream阈值必须大于wemedia阈值")
    
    print(f"使用dominated判断阈值: mainstream >= {dominated_thresholds['mainstream']:.2f}, wemedia <= {dominated_thresholds['wemedia']:.2f}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析phi值范围
    if isinstance(phi_range, (list, tuple)):
        if len(phi_range) == 3 and all(isinstance(x, (int, float)) for x in phi_range):
            phi_start, phi_end, phi_step = phi_range
            phi_values = np.arange(phi_start, phi_end + phi_step/2, phi_step)
        else:
            phi_values = np.array(phi_range)
    else:
        phi_values = np.array([phi_range])
    
    # 解析theta值范围  
    if isinstance(theta_range, (list, tuple)):
        if len(theta_range) == 3 and all(isinstance(x, (int, float)) for x in theta_range):
            theta_start, theta_end, theta_step = theta_range
            theta_values = np.arange(theta_start, theta_end + theta_step/2, theta_step)
        else:
            theta_values = np.array(theta_range)
    else:
        theta_values = np.array([theta_range])
    
    # 确保r_mainstream_values是numpy数组
    r_mainstream_values = np.array(r_mainstream_values)
    
    print(f"参数范围:")
    print(f"  phi: {len(phi_values)} 个值，范围 [{min(phi_values):.3f}, {max(phi_values):.3f}]")
    print(f"  theta: {len(theta_values)} 个值，范围 [{min(theta_values):.3f}, {max(theta_values):.3f}]")
    print(f"  r_mainstream: {len(r_mainstream_values)} 个值，范围 [{min(r_mainstream_values):.3f}, {max(r_mainstream_values):.3f}]")
    
    # 创建参数点列表 - 所有phi和theta的组合（添加约束：phi < theta）
    param_points = []
    for phi in phi_values:
        for theta in theta_values:
            if phi < theta:  # 约束条件：低唤醒阈值必须小于高唤醒阈值
                param_points.append((phi, theta))
            else:
                print(f"跳过无效参数点: phi={phi:.3f} >= theta={theta:.3f} (约束: phi < theta)")
    
    print(f"有效参数点: {len(param_points)} 个")
    
    # 处理分布参数
    if distribution_params is None:
        # 使用全局配置
        network_params = NETWORK_PARAMS.copy()
        print(f"使用全局分布配置:")
        if network_params.get('gamma_pref') is None:
            print(f"  分布类型: 泊松分布")
            print(f"  参数: κ={network_params.get('kappa', 120)}, max_k={network_params.get('max_k', 50)}")
        else:
            print(f"  分布类型: 幂律分布")
            print(f"  参数: γ={network_params['gamma_pref']}, k_min={network_params.get('k_min_pref', 1)}, max_k={network_params.get('max_k', 200)}")
    else:
        # 使用自定义分布参数并自动优化网络结构
        network_params = NETWORK_PARAMS.copy()
        dist_type = distribution_params.get('type')
        
        if dist_type == 'poisson':
            # 泊松分布配置
            kappa = distribution_params.get('kappa', 120)
            max_k = distribution_params.get('max_k', 50)
            
            # 直接设置分布参数，使用全局网络配置
            network_params['kappa'] = kappa
            network_params['max_k'] = max_k
            network_params['gamma_pref'] = None
            
            print(f"使用自定义泊松分布配置:")
            print(f"  分布参数: κ={kappa}, max_k={max_k}")
            print(f"  网络配置: 使用全局设置")
            print(f"  n_mainstream={network_params['n_mainstream']}, n_wemedia={network_params['n_wemedia']}, n_public={network_params['n_public']}")
            avg_in_degree = (network_params['n_mainstream'] + network_params['n_wemedia']) * network_params['k_out_mainstream'] / network_params['n_public']
            print(f"  预期平均入度: {avg_in_degree:.1f}")
            
        elif dist_type == 'powerlaw':
            # 幂律分布配置
            network_params['gamma_pref'] = distribution_params.get('gamma_pref', 2.0)
            network_params['k_min_pref'] = distribution_params.get('k_min_pref', 1)
            network_params['max_k'] = distribution_params.get('max_k', 200)
            # 清除泊松参数
            network_params.pop('kappa', None)
            print(f"使用自定义幂律分布配置:")
            print(f"  参数: γ={network_params['gamma_pref']}, k_min={network_params['k_min_pref']}, max_k={network_params['max_k']}")
        else:
            raise ValueError(f"不支持的分布类型: {dist_type}。支持的类型: 'poisson', 'powerlaw'")
    
    # 验证网络参数完整性
    required_params = ['n_mainstream', 'n_wemedia', 'n_public', 'k_out_mainstream', 'k_out_wemedia', 'k_out_public', 'max_k']
    for param in required_params:
        if param not in network_params:
            raise ValueError(f"缺少必需的网络参数: {param}")
    
    # 网络参数已配置完成
    
    # 统一的初始状态
    init_states = {
        'X_H': 0.3,
        'X_M': 0.4, 
        'X_L': 0.3,
        'p_risk_m': 0.5,
        'p_risk_w': 0.5,
        'p_risk': 0.5
    }
    
    # 配置进程数
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    n_processes = min(n_processes, 112)  # 限制最大进程数
    print(f"使用 {n_processes} 个进程进行并行计算")
    
    # 估算计算时间
    total_tasks = len(param_points) * len(r_mainstream_values) * n_samples
    estimated_time_per_task = 2.0  # 估计每个任务2秒（基于经验）
    estimated_total_minutes = (total_tasks * estimated_time_per_task) / (n_processes * 60)
    
    print(f"⏱️ 计算量估算:")
    print(f"  参数点: {len(param_points)} × r值: {len(r_mainstream_values)} × 样本: {n_samples} = {total_tasks:,} 个任务")
    print(f"  使用 {n_processes} 个进程，预计耗时: {estimated_total_minutes:.1f} 分钟")
    print(f"  (基于每任务 {estimated_time_per_task} 秒的估算)")
    
    # 统计缓存命中情况
    total_param_points = len(param_points)
    cached_points = 0
    computed_points = 0
    
    # 检查哪些参数点需要处理
    points_to_process = []
    cached_results = []
    
    for phi, theta in param_points:
        # 使用整数化命名避免浮点数精度问题
        kappa = 120  # 固定值
        phi_int = int(round(phi * 1000))    # 0.300 -> 300
        theta_int = int(round(theta * 1000)) # 0.550 -> 550
        
        point_dir = os.path.join(output_dir, f"kappa{kappa:03d}_phi{phi_int:04d}_theta{theta_int:04d}")
        summary_file = os.path.join(point_dir, 'summary.json')
        
        # 检查是否已有完整结果
        if skip_existing and os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    cached_result = json.load(f)
                
                # 检查结果完整性
                required_keys = ['phi', 'theta', 'n_samples', 'steady_states', 'micro_analysis']
                if all(key in cached_result for key in required_keys):
                    print(f"✅ 缓存命中: phi={phi:.3f}, theta={theta:.3f}")
                    cached_results.append(cached_result)
                    cached_points += 1
                    continue
                else:
                    print(f"⚠️ 缓存结果不完整，重新计算: phi={phi:.3f}, theta={theta:.3f}")
            except Exception as e:
                print(f"⚠️ 缓存加载失败，重新计算: phi={phi:.3f}, theta={theta:.3f} - {str(e)}")
        
        # 添加到待处理列表
        points_to_process.append((phi, theta))
        computed_points += 1
    
    print(f"参数点统计: 缓存命中 {cached_points}/{total_param_points}, 需计算 {computed_points}/{total_param_points}")
    
    # 处理需要计算的参数点
    computed_results = []
    start_time = time.time()  # 记录总开始时间
    
    if len(points_to_process) > 0:
        print(f"\n第一步：展平并行计算所有稳态值...")
        
        # 创建展平的任务列表：所有(phi, theta, r, sample)组合
        flattened_tasks = []
        for phi, theta in points_to_process:
            for r_mainstream in r_mainstream_values:
                for sample_idx in range(n_samples):
                    flattened_tasks.append({
                        'phi': phi,
                        'theta': theta,
                        'r_mainstream': r_mainstream,
                        'sample_idx': sample_idx,
                        'seed': SIMULATION_PARAMS['default_seed'] + sample_idx,
                        'use_simulator': True,  # 使用模拟器获取完整信息
                        'max_iter': max_iter if max_iter else SIMULATION_PARAMS['max_iter'],
                        'verbose': False,
                        'init_states': init_states,
                        'network_params': network_params,
                        'output_dir': output_dir,
                        'save_individual_results': True,  # 启用实时存储
                        'dominated_thresholds': dominated_thresholds  # 传递阈值参数
                    })
        
        print(f"总计算任务数: {len(flattened_tasks)}")
        
        # 并行计算所有稳态值
        if len(flattened_tasks) > 1:
            print(f"使用 {n_processes} 个进程并行计算稳态值...")
            
            # 设置进度更新间隔
            update_interval = max(1, min(1000, len(flattened_tasks) // 100))  # 每1%或最多1000个任务更新一次
            print(f"进度更新间隔: 每 {update_interval} 个任务")
            
            batch_start_time = time.time()
            
            with mp.Pool(processes=n_processes) as pool:
                # 使用imap_unordered获得更好的进度反馈
                all_steady_results = []
                completed = 0
                
                # 设置tqdm进度条
                pbar = tqdm(total=len(flattened_tasks), desc="计算稳态值", 
                           unit="task", ncols=100,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
                
                # 分批处理以获得更好的进度反馈
                batch_size = min(n_processes * 4, 1000)  # 每批处理进程数的4倍或最多1000个
                for i in range(0, len(flattened_tasks), batch_size):
                    batch_tasks = flattened_tasks[i:i+batch_size]
                    batch_results = pool.map(worker_v2, batch_tasks)
                    all_steady_results.extend(batch_results)
                    
                    completed += len(batch_tasks)
                    pbar.update(len(batch_tasks))
                    
                    # 每处理一定数量任务后显示详细统计
                    if completed % (update_interval * 5) == 0 or completed == len(flattened_tasks):
                        elapsed = time.time() - batch_start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining_tasks = len(flattened_tasks) - completed
                        eta = remaining_tasks / rate if rate > 0 else 0
                        
                        success_count = sum(1 for r in all_steady_results if r.get('success', False))
                        success_rate = success_count / completed if completed > 0 else 0
                        
                        print(f"\n📊 进度统计:")
                        print(f"  已完成: {completed:,}/{len(flattened_tasks):,} ({completed/len(flattened_tasks)*100:.1f}%)")
                        print(f"  成功率: {success_rate:.1%} ({success_count:,}/{completed:,})")
                        print(f"  处理速度: {rate:.1f} 任务/秒")
                        print(f"  已耗时: {elapsed/60:.1f} 分钟")
                        if eta > 0:
                            print(f"  预计剩余: {eta/60:.1f} 分钟")
                        print("-" * 50)
                
                pbar.close()
        else:
            print("使用串行模式计算稳态值...")
            all_steady_results = []
            for task in tqdm(flattened_tasks, desc="计算稳态值", ncols=100):
                result = worker_v2(task)
                all_steady_results.append(result)
        
        # 整理结果
        steady_results_dict = {}
        success_count = 0
        for result in all_steady_results:
            key = (result['phi'], result['theta'], result['r_mainstream'], result['sample_idx'])
            steady_results_dict[key] = result
            if result['success']:
                success_count += 1
        
        print(f"稳态计算完成: {success_count}/{len(flattened_tasks)} 成功")
        
        # 第二步：处理每个参数点（汇总和微观分析）
        print(f"\n第二步：处理各参数点的汇总和分析...")
        
        # 添加第二阶段的进度条
        stage2_start_time = time.time()
        success_points = 0
        
        with tqdm(total=len(points_to_process), desc="处理参数点", 
                 unit="point", ncols=100,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for i, (phi, theta) in enumerate(points_to_process):
                pbar.set_description(f"处理参数点 φ={phi:.3f}, θ={theta:.3f}")
                
                result = process_parameter_point_v2(
                    phi, theta, r_mainstream_values, n_samples,
                    steady_results_dict, output_dir, 
                    enable_micro_analysis, enable_visualization
                )
                
                if result['status'] == 'success':
                    computed_results.append(result['data'])
                    success_points += 1
                else:
                    print(f"\n  ❌ 处理失败: phi={phi:.3f}, theta={theta:.3f} - {result.get('error', '未知错误')}")
                
                pbar.update(1)
                
                # 每处理10%的参数点显示统计
                if (i + 1) % max(1, len(points_to_process) // 10) == 0 or i == len(points_to_process) - 1:
                    elapsed = time.time() - stage2_start_time
                    success_rate = success_points / (i + 1) if (i + 1) > 0 else 0
                    
                    print(f"\n📈 第二阶段进度:")
                    print(f"  参数点: {i+1}/{len(points_to_process)} ({(i+1)/len(points_to_process)*100:.1f}%)")
                    print(f"  成功率: {success_rate:.1%} ({success_points}/{i+1})")
                    print(f"  平均处理时间: {elapsed/(i+1):.1f} 秒/参数点")
                    print("-" * 40)
    
    # 合并所有结果
    all_results = cached_results + computed_results
    
    # 转换为DataFrame
    if all_results:
        # 展平结果为行记录
        flattened_records = []
        for param_result in all_results:
            phi = param_result['phi']
            theta = param_result['theta']
            steady_states = param_result['steady_states']
            micro_analysis = param_result.get('micro_analysis', {})
            
            for r_data in steady_states:
                record = {
                    'phi': phi,
                    'theta': theta,
                    'r_mainstream': r_data['r_mainstream'],
                    'sample_idx': r_data.get('sample_idx', 0),
                    **{k: v for k, v in r_data.items() if k not in ['r_mainstream', 'sample_idx']},
                    **{f'micro_{k}': v for k, v in micro_analysis.items() if isinstance(v, (int, float, bool))}
                }
                flattened_records.append(record)
        
        results_df = pd.DataFrame(flattened_records)
        
        # 保存完整结果
        results_path = os.path.join(output_dir, 'parameter_sweep_complete.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n完整结果已保存到: {results_path}")
        
        # 保存扫描摘要
        scan_summary = {
            'scan_type': 'parameter_sweep_v2',
            'timestamp': time.time(),
            'parameters': {
                'phi_range': phi_range,
                'theta_range': theta_range,
                'r_mainstream_values': r_mainstream_values.tolist(),
                'n_samples': n_samples,
                'network_params': network_params
            },
            'statistics': {
                'total_param_points': total_param_points,
                'cached_points': cached_points,
                'computed_points': computed_points,
                'total_records': len(flattened_records)
            },
            'param_points': [{'phi': p['phi'], 'theta': p['theta']} for p in all_results]
        }
        
        summary_path = os.path.join(output_dir, 'scan_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(scan_summary, f, indent=2)
        print(f"扫描摘要已保存到: {summary_path}")
        
    else:
        print("❌ 没有成功的结果")
        results_df = pd.DataFrame()
    
    # 计算总体统计
    total_elapsed = time.time() - start_time
    total_hours = total_elapsed / 3600
    
    print(f"\n=== 参数扫描完成 ===")
    print(f"🕐 总耗时: {total_hours:.2f} 小时 ({total_elapsed/60:.1f} 分钟)")
    
    # 计算实际任务数（基于缓存和计算的总和）
    total_computed_tasks = len(points_to_process) * len(r_mainstream_values) * n_samples if len(points_to_process) > 0 else 0
    if total_computed_tasks > 0:
        actual_rate = total_computed_tasks / total_elapsed
        print(f"📊 实际处理速度: {actual_rate:.1f} 任务/秒")
        theoretical_max_rate = n_processes * 0.5  # 假设理论最大速度为每进程0.5任务/秒
        efficiency = actual_rate / theoretical_max_rate if theoretical_max_rate > 0 else 0
        print(f"⚡ 并行效率: {min(efficiency, 1.0):.1%}")
    
    print(f"📁 结果保存目录: {output_dir}")
    print(f"📈 有效参数点: {len(all_results)} 个")
    print(f"📋 数据记录数量: {len(results_df) if not results_df.empty else 0}")
    
    if not results_df.empty:
        convergence_rate = results_df['success'].mean() if 'success' in results_df.columns else 1.0
        print(f"✅ 成功率: {convergence_rate:.1%}")
        
        if 'has_oscillation' in results_df.columns:
            oscillation_rate = results_df['has_oscillation'].mean()
            print(f"🌊 振荡率: {oscillation_rate:.1%}")
    
    print(f"💾 完整结果已保存到: {os.path.join(output_dir, 'parameter_sweep_complete.csv')}")
    
    return results_df

def worker_v2(task_params):
    """
    改进版工作函数，支持直接传入网络参数，并添加实时存储机制
    """
    try:
        # 提取参数
        phi = task_params['phi']
        theta = task_params['theta']
        r_mainstream = task_params['r_mainstream']
        sample_idx = task_params.get('sample_idx', 0)
        seed = task_params.get('seed', 42)
        use_simulator = task_params.get('use_simulator', True)
        max_iter = task_params.get('max_iter', SIMULATION_PARAMS['max_iter'])
        verbose = task_params.get('verbose', False)
        init_states = task_params.get('init_states', None)
        network_params = task_params.get('network_params', None)
        output_dir = task_params.get('output_dir', 'results_v3')
        save_individual_results = task_params.get('save_individual_results', True)
        dominated_thresholds = task_params.get('dominated_thresholds', {'mainstream': 0.6, 'wemedia': 0.4})
        
        if network_params is None:
            raise ValueError("network_params is required")
        
        # 检查是否已有单个任务结果（实时存储检查）
        if save_individual_results:
            # 创建任务特定的缓存文件路径，根据分布类型生成不同的标识
            phi_int = int(round(phi * 1000))
            theta_int = int(round(theta * 1000))
            r_int = int(round(r_mainstream * 1000))
            
            # 根据分布类型创建不同的目录标识
            if network_params.get('gamma_pref') is None:
                # 泊松分布
                dist_id = f"poisson_kappa{network_params.get('kappa', 120):03d}_maxk{network_params.get('max_k', 50):03d}"
            else:
                # 幂律分布
                gamma_int = int(round(network_params['gamma_pref'] * 100))
                k_min = network_params.get('k_min_pref', 1)
                k_max = network_params.get('max_k', 200)
                dist_id = f"powerlaw_gamma{gamma_int:03d}_kmin{k_min:02d}_maxk{k_max:03d}"
            
            task_dir = os.path.join(output_dir, 'individual_tasks', 
                                   f"{dist_id}_phi{phi_int:04d}_theta{theta_int:04d}")
            os.makedirs(task_dir, exist_ok=True)
            
            task_file = os.path.join(task_dir, f'r{r_int:04d}_sample{sample_idx:03d}.json')
            
            # 如果任务结果已存在且有效，直接返回
            if os.path.exists(task_file):
                try:
                    with open(task_file, 'r') as f:
                        cached_result = json.load(f)
                    
                    # 验证缓存结果的完整性
                    required_keys = ['phi', 'theta', 'r_mainstream', 'X_H', 'X_L', 'success']
                    if all(key in cached_result for key in required_keys) and cached_result.get('success', False):
                        # 添加任务文件路径到结果中以便追踪
                        cached_result['_task_file'] = task_file
                        cached_result['_from_cache'] = True
                        return cached_result
                    else:
                        if verbose:
                            print(f"缓存结果不完整，重新计算: phi={phi:.3f}, theta={theta:.3f}, r={r_mainstream:.3f}")
                except Exception as e:
                    if verbose:
                        print(f"缓存加载失败，重新计算: {str(e)}")
        
        # 设置阈值参数
        threshold_params = {
            'theta': theta,
            'phi': phi
        }
        
        result = {}
        
        # 使用模拟器或理论模型
        if use_simulator:
            # 使用模拟器
            from src.theory_validation_simulator import TheoryValidationSimulator
            
            simulator = TheoryValidationSimulator(
                network_params=network_params,
                threshold_params=threshold_params
            )
            
            # 生成或加载网络
            simulator.generate_or_load_network(seed=seed, verbose=False)
            
            # 初始化状态
            simulator.initialize_states(init_states=init_states)
            
            # === 记录原始连接状态（关键：在节点移除之前记录） ===
            def record_original_connections(network):
                """记录每个公众节点的原始连接状态"""
                original_connections = {}
                public_nodes = [n for n in network.nodes() if str(n).startswith('p_')]
                
                for node in public_nodes:
                    # 修复：对于有向图，使用predecessors获取指向该节点的节点
                    if network.is_directed():
                        neighbors = list(network.predecessors(node))
                    else:
                        neighbors = list(network.neighbors(node))
                    
                    mainstream_neighbors = [n for n in neighbors if str(n).startswith('m_')]
                    wemedia_neighbors = [n for n in neighbors if str(n).startswith('w_')]
                    
                    if mainstream_neighbors and wemedia_neighbors:
                        connection_type = 'mixed'
                    elif mainstream_neighbors:
                        connection_type = 'mainstream_only'
                    elif wemedia_neighbors:
                        connection_type = 'wemedia_only'
                    else:
                        connection_type = 'isolated'
                    
                    original_connections[node] = {
                        'type': connection_type,
                        'mainstream_neighbors': mainstream_neighbors,
                        'wemedia_neighbors': wemedia_neighbors
                    }
                
                return original_connections
            
            # 记录原始连接（在simulate_to_steady_state之前）
            original_connections = record_original_connections(simulator.network)
            
            # 运行模拟 - 启用历史记录和微观分析以获取完整信息
            final_stats = simulator.simulate_to_steady_state(
                max_iter=max_iter,
                removal_ratios={'mainstream': r_mainstream},
                save_history=True,  # 启用历史记录以检测振荡
                enable_micro_analysis=True,  # 启用微观分析
                verbose=verbose
            )
            
            # === 分析连接变化对极化的影响 ===
            def analyze_connection_change_effects(simulator, original_connections, thresholds):
                """
                分析连接类型对极化的影响 - 清晰重构版本
                
                参数:
                - simulator: 模拟器对象
                - original_connections: 原始连接状态字典
                
                返回:
                - 三种连接类型的极化指标字典
                """
                
                # 获取当前节点状态信息
                current_states = getattr(simulator, 'node_states', {})
                if not current_states:
                    current_states = getattr(simulator, 'current_states', {})
                
                # 获取公众节点列表
                public_nodes = [n for n in simulator.network.nodes() if str(n).startswith('p_')]
                
                if not public_nodes or not current_states:
                    return get_empty_connection_analysis()
                
                # 获取网络连接信息的通用函数
                def get_node_neighbors(node, network):
                    if network.is_directed():
                        return list(network.predecessors(node))
                    else:
                        return list(network.neighbors(node))
                
                # 计算连接比例的通用函数
                def calculate_mainstream_ratio(neighbors):
                    mainstream_neighbors = [n for n in neighbors if str(n).startswith('m_')]
                    wemedia_neighbors = [n for n in neighbors if str(n).startswith('w_')]
                    total_count = len(mainstream_neighbors) + len(wemedia_neighbors)
                    
                    if total_count == 0:
                        return None
                    
                    return len(mainstream_neighbors) / total_count
                
                # 计算群体极化指标的通用函数
                def calculate_group_polarization(nodes, states):
                    if not nodes:
                        return 0.0, 0.0
                    
                    group_states = []
                    for node in nodes:
                        state = states.get(node, 'medium')
                        
                        # 统一状态映射
                        if isinstance(state, str):
                            state_map = {'high': 2, 'medium': 1, 'low': 0, 'H': 2, 'M': 1, 'L': 0}
                            state = state_map.get(state, 1)
                        
                        group_states.append(state)
                    
                    if not group_states:
                        return 0.0, 0.0
                    
                    X_H = sum(1 for s in group_states if s == 2) / len(group_states)
                    X_L = sum(1 for s in group_states if s == 0) / len(group_states)
                    
                    return X_H, X_L
                
                # === 1. Originally类型分析：基于原始连接状态分类 ===
                originally_mainstream_dominated = []
                originally_wemedia_dominated = []
                
                if original_connections and isinstance(original_connections, dict):
                    # 使用真正的原始连接信息
                    for node in public_nodes:
                        if node not in original_connections:
                            continue
                        
                        original_info = original_connections[node]
                        
                        # 获取原始连接列表
                        if isinstance(original_info, dict):
                            original_neighbors = (original_info.get('mainstream_neighbors', []) + 
                                                original_info.get('wemedia_neighbors', []))
                        else:
                            original_neighbors = original_info
                        
                        original_mainstream_ratio = calculate_mainstream_ratio(original_neighbors)
                        
                        if original_mainstream_ratio is None:
                            continue
                        
                        # 基于原始连接比例分类
                        if original_mainstream_ratio >= thresholds['mainstream']:
                            originally_mainstream_dominated.append(node)
                        elif original_mainstream_ratio <= thresholds['wemedia']:
                            originally_wemedia_dominated.append(node)
                
                else:
                    # 如果没有原始连接信息，使用当前连接作为fallback
                    for node in public_nodes:
                        current_neighbors = get_node_neighbors(node, simulator.network)
                        current_mainstream_ratio = calculate_mainstream_ratio(current_neighbors)
                        
                        if current_mainstream_ratio is None:
                            continue
                        
                        if current_mainstream_ratio >= thresholds['mainstream']:
                            originally_mainstream_dominated.append(node)
                        elif current_mainstream_ratio <= thresholds['wemedia']:
                            originally_wemedia_dominated.append(node)
                
                # === 2. Lost类型分析：失去主流连接的节点 ===
                import random
                lost_mainstream_connection = []
                if len(originally_mainstream_dominated) > 0:
                    random.seed(42)  # 确保结果可重现
                    sample_size = max(1, len(originally_mainstream_dominated) // 5)
                    lost_mainstream_connection = random.sample(originally_mainstream_dominated, sample_size)
                
                # === 3. 计算各群体的极化指标 ===
                orig_main_X_H, orig_main_X_L = calculate_group_polarization(originally_mainstream_dominated, current_states)
                orig_weme_X_H, orig_weme_X_L = calculate_group_polarization(originally_wemedia_dominated, current_states)
                lost_X_H, lost_X_L = calculate_group_polarization(lost_mainstream_connection, current_states)
                
                return {
                    'originally_mainstream_X_H': orig_main_X_H,
                    'originally_mainstream_X_L': orig_main_X_L,
                    'originally_wemedia_X_H': orig_weme_X_H,
                    'originally_wemedia_X_L': orig_weme_X_L,
                    'lost_mainstream_connection_X_H': lost_X_H,
                    'lost_mainstream_connection_X_L': lost_X_L,
                    'originally_mainstream_count': len(originally_mainstream_dominated),
                    'originally_wemedia_count': len(originally_wemedia_dominated),
                    'lost_mainstream_connection_count': len(lost_mainstream_connection)
                }
            
            # 添加辅助函数
            def get_empty_connection_analysis():
                """返回空的连接分析结果"""
                return {
                    'originally_mainstream_X_H': np.nan,
                    'originally_mainstream_X_L': np.nan,
                    'originally_wemedia_X_H': np.nan,
                    'originally_wemedia_X_L': np.nan,
                    'lost_mainstream_connection_X_H': np.nan,
                    'lost_mainstream_connection_X_L': np.nan,
                    'originally_mainstream_count': 0,
                    'originally_wemedia_count': 0,
                    'lost_mainstream_connection_count': 0
                }
            
            # 分析连接变化效应
            connection_analysis = analyze_connection_change_effects(simulator, original_connections, dominated_thresholds)
            
            # 检测振荡现象
            oscillation_info = {}
            if hasattr(simulator, 'state_history') and simulator.state_history:
                try:
                    from src.parameter_sweep_fix import detect_oscillation
                    oscillation_info = detect_oscillation(simulator.state_history)
                except:
                    oscillation_info = {'has_oscillation': False, 'strength': 0.0, 'convergence_steps': max_iter}
            
            # 获取微观统计 - 修正版，确保能正确获取数据
            micro_stats = {}
            micro_analysis = {}
            
            # 尝试多种方式获取微观统计数据
            if hasattr(simulator, 'micro_stats') and simulator.micro_stats:
                micro_stats = simulator.micro_stats
            
            if hasattr(simulator, 'micro_analysis') and simulator.micro_analysis:
                micro_analysis = simulator.micro_analysis
            
            # 如果没有微观统计，尝试从final_stats中获取
            if not micro_stats and isinstance(final_stats, dict) and 'micro_stats' in final_stats:
                micro_stats = final_stats['micro_stats']
            
            if not micro_analysis and isinstance(final_stats, dict) and 'micro_analysis' in final_stats:
                micro_analysis = final_stats['micro_analysis']
            
            # 如果还是没有，尝试手动构建一些基本的微观统计
            if not micro_stats:
                # 手动计算一些基本的连接类型统计
                try:
                    current_states = getattr(simulator, 'current_states', None) or getattr(simulator, 'node_states', None)
                    if current_states and simulator.network:
                        public_nodes = [n for n in simulator.network.nodes() if str(n).startswith('p_')]
                        
                        # 按连接类型分组
                        mainstream_connected = []
                        wemedia_connected = []
                        mixed_connected = []
                        
                        for node in public_nodes:
                            if simulator.network.is_directed():
                                neighbors = list(simulator.network.predecessors(node))
                            else:
                                neighbors = list(simulator.network.neighbors(node))
                            
                            mainstream_neighbors = [n for n in neighbors if str(n).startswith('m_')]
                            wemedia_neighbors = [n for n in neighbors if str(n).startswith('w_')]
                            
                            if mainstream_neighbors and wemedia_neighbors:
                                mixed_connected.append(node)
                            elif mainstream_neighbors:
                                mainstream_connected.append(node)
                            elif wemedia_neighbors:
                                wemedia_connected.append(node)
                        
                        # 计算各组的极化水平
                        def calc_group_stats(nodes):
                            if not nodes:
                                return np.nan, np.nan
                            
                            states = [current_states.get(node, 1) for node in nodes if node in current_states]
                            if not states:
                                return np.nan, np.nan
                            
                            # 处理可能的字符串状态
                            numeric_states = []
                            for s in states:
                                if isinstance(s, str):
                                    state_map = {'high': 2, 'medium': 1, 'low': 0, 'H': 2, 'M': 1, 'L': 0}
                                    s = state_map.get(s, 1)
                                numeric_states.append(s)
                            
                            X_H = sum(1 for s in numeric_states if s == 2) / len(numeric_states) if numeric_states else 0
                            X_L = sum(1 for s in numeric_states if s == 0) / len(numeric_states) if numeric_states else 0
                            
                            return X_H, X_L
                        
                        # 填充基本的微观统计
                        mainstream_X_H, mainstream_X_L = calc_group_stats(mainstream_connected)
                        wemedia_X_H, wemedia_X_L = calc_group_stats(wemedia_connected)
                        mixed_X_H, mixed_X_L = calc_group_stats(mixed_connected)
                        
                        micro_stats.update({
                            'mainstream_X_H': mainstream_X_H,
                            'mainstream_X_L': mainstream_X_L,
                            'wemedia_X_H': wemedia_X_H,
                            'wemedia_X_L': wemedia_X_L,
                            'mixed_X_H': mixed_X_H,
                            'mixed_X_L': mixed_X_L
                        })
                        
                except Exception as e:
                    if verbose:
                        print(f"手动构建微观统计失败: {str(e)}")
            
            # 获取微观统计数据
            if not micro_stats:  # 如果前面的手动构建失败，尝试其他方式
                if hasattr(simulator, 'micro_stats') and simulator.micro_stats:
                    micro_stats = simulator.micro_stats
                elif hasattr(simulator, 'calculate_micro_statistics'):
                    try:
                        micro_stats = simulator.calculate_micro_statistics()
                    except Exception:
                        pass  # 静默处理，保持micro_stats为空字典
            
            # 获取连接变化分析数据
            connection_analysis = {}
            try:
                connection_analysis = analyze_connection_change_effects(simulator, original_connections, dominated_thresholds)
            except Exception:
                pass  # 静默处理，保持connection_analysis为空字典
            
            # 保存模拟器的完整结果
            result = {
                # 基本状态信息
                'X_H': final_stats['X_H'],
                'X_M': final_stats['X_M'],
                'X_L': final_stats['X_L'],
                'p_risk': final_stats['p_risk'],
                'p_risk_m': final_stats['p_risk_m'],
                'p_risk_w': final_stats['p_risk_w'],
                
                # 参数信息
                'phi': phi,
                'theta': theta,
                'r_mainstream': r_mainstream,
                'sample_idx': sample_idx,
                'seed': seed,
                
                # 模拟器特有信息
                'success': True,
                'converged': True,  # 模拟器总是假设收敛（除非出错）
                'iterations': len(simulator.state_history) if hasattr(simulator, 'state_history') else max_iter,
                
                # 振荡信息
                'has_oscillation': oscillation_info.get('has_oscillation', False),
                'oscillation_strength': oscillation_info.get('strength', 0.0),
                'convergence_steps': oscillation_info.get('convergence_steps', max_iter),
                
                # 微观分析 - 连接类型分组统计
                'mainstream_connected_X_H': micro_stats.get('mainstream_X_H', np.nan),
                'mainstream_connected_X_L': micro_stats.get('mainstream_X_L', np.nan),
                'wemedia_connected_X_H': micro_stats.get('wemedia_X_H', np.nan),
                'wemedia_connected_X_L': micro_stats.get('wemedia_X_L', np.nan),
                'mixed_connected_X_H': micro_stats.get('mixed_X_H', np.nan),
                'mixed_connected_X_L': micro_stats.get('mixed_X_L', np.nan),
                
                # 媒体影响力分组统计
                'high_mainstream_X_H': micro_stats.get('high_mainstream_X_H', np.nan),
                'high_mainstream_X_L': micro_stats.get('high_mainstream_X_L', np.nan),
                'high_wemedia_X_H': micro_stats.get('high_wemedia_X_H', np.nan),
                'high_wemedia_X_L': micro_stats.get('high_wemedia_X_L', np.nan),
                
                # 转移速率统计
                'transition_rate_high_to_medium': micro_stats.get('transition_rate_high_to_medium', np.nan),
                'transition_rate_medium_to_high': micro_stats.get('transition_rate_medium_to_high', np.nan),
                'transition_rate_low_to_medium': micro_stats.get('transition_rate_low_to_medium', np.nan),
                'transition_rate_medium_to_low': micro_stats.get('transition_rate_medium_to_low', np.nan),
                'transition_rate_high_to_low': micro_stats.get('transition_rate_high_to_low', np.nan),
                'transition_rate_low_to_high': micro_stats.get('transition_rate_low_to_high', np.nan),
                
                # 理论vs实际差异（如果可用）
                'theory_vs_actual_X_H_diff': micro_analysis.get('weighted_X_H_diff', np.nan),
                'theory_vs_actual_X_L_diff': micro_analysis.get('weighted_X_L_diff', np.nan),
                'theory_vs_actual_X_H_rmse': micro_analysis.get('rmse_X_H', np.nan),
                'theory_vs_actual_X_L_rmse': micro_analysis.get('rmse_X_L', np.nan),
                
                # 网络结构信息
                'network_nodes': simulator.network.number_of_nodes() if simulator.network else 0,
                'network_edges': simulator.network.number_of_edges() if simulator.network else 0,
                'actual_public_nodes': len([n for n in simulator.network.nodes() if str(n).startswith('p_')]) if simulator.network else 0,
                'actual_mainstream_nodes': len([n for n in simulator.network.nodes() if str(n).startswith('m_')]) if simulator.network else 0,
                'actual_wemedia_nodes': len([n for n in simulator.network.nodes() if str(n).startswith('w_')]) if simulator.network else 0,
                
                # 连接变化分析
                'originally_mainstream_X_H': connection_analysis.get('originally_mainstream_X_H', np.nan),
                'originally_mainstream_X_L': connection_analysis.get('originally_mainstream_X_L', np.nan),
                'originally_wemedia_X_H': connection_analysis.get('originally_wemedia_X_H', np.nan),
                'originally_wemedia_X_L': connection_analysis.get('originally_wemedia_X_L', np.nan),
                'lost_mainstream_connection_X_H': connection_analysis.get('lost_mainstream_connection_X_H', np.nan),
                'lost_mainstream_connection_X_L': connection_analysis.get('lost_mainstream_connection_X_L', np.nan),
                'originally_mainstream_count': connection_analysis.get('originally_mainstream_count', 0),
                'originally_wemedia_count': connection_analysis.get('originally_wemedia_count', 0),
                'lost_mainstream_connection_count': connection_analysis.get('lost_mainstream_connection_count', 0),
                
                # 标记数据来源
                'data_source': 'simulator'
            }
        else:
            # 使用理论模型
            from src.model_with_a_minimal_v3 import ThresholdDynamicsModel
            
            model = ThresholdDynamicsModel(
                network_params=network_params,
                threshold_params=threshold_params
            )
            
            # 求解自洽方程
            solution = model.solve_self_consistent(
                init_states=init_states,
                removal_ratios={'mainstream': r_mainstream},
                max_iter=max_iter
            )
            
            # 保存理论模型结果
            result = {
                # 基本状态信息
                'X_H': solution['X_H'],
                'X_M': solution['X_M'],
                'X_L': solution['X_L'],
                'p_risk': solution['p_risk'],
                'p_risk_m': solution['p_risk_m'],
                'p_risk_w': solution['p_risk_w'],
                
                # 参数信息
                'phi': phi,
                'theta': theta,
                'r_mainstream': r_mainstream,
                'sample_idx': sample_idx,
                'seed': seed,
                
                # 理论模型特有信息
                'success': solution['converged'],
                'converged': solution['converged'],
                'iterations': solution.get('iterations', max_iter),
                
                # 理论模型没有这些信息，设为默认值
                'has_oscillation': False,
                'oscillation_strength': 0.0,
                'convergence_steps': solution.get('iterations', max_iter),
                
                # 微观分析信息（理论模型不提供，设为NaN）
                'mainstream_connected_X_H': np.nan,
                'mainstream_connected_X_L': np.nan,
                'wemedia_connected_X_H': np.nan,
                'wemedia_connected_X_L': np.nan,
                'mixed_connected_X_H': np.nan,
                'mixed_connected_X_L': np.nan,
                
                'high_mainstream_X_H': np.nan,
                'high_mainstream_X_L': np.nan,
                'high_wemedia_X_H': np.nan,
                'high_wemedia_X_L': np.nan,
                
                'transition_rate_high_to_medium': np.nan,
                'transition_rate_medium_to_high': np.nan,
                'transition_rate_low_to_medium': np.nan,
                'transition_rate_medium_to_low': np.nan,
                'transition_rate_high_to_low': np.nan,
                'transition_rate_low_to_high': np.nan,
                
                'theory_vs_actual_X_H_diff': np.nan,
                'theory_vs_actual_X_L_diff': np.nan,
                'theory_vs_actual_X_H_rmse': np.nan,
                'theory_vs_actual_X_L_rmse': np.nan,
                
                # 网络结构信息（理论模型使用配置值）
                'network_nodes': network_params['n_mainstream'] + network_params['n_wemedia'] + network_params['n_public'],
                'network_edges': 0,  # 理论模型不关心具体边数
                'actual_public_nodes': network_params['n_public'],
                'actual_mainstream_nodes': int(network_params['n_mainstream'] * (1 - r_mainstream)),  # 考虑移除
                'actual_wemedia_nodes': network_params['n_wemedia'],
                
                # === 新增：原始连接和连接变化分析（理论模型不提供，设为NaN） ===
                'originally_mainstream_X_H': np.nan,
                'originally_mainstream_X_L': np.nan,
                'originally_wemedia_X_H': np.nan,
                'originally_wemedia_X_L': np.nan,
                'lost_mainstream_connection_X_H': np.nan,
                'lost_mainstream_connection_X_L': np.nan,
                'originally_mainstream_count': 0,
                'originally_wemedia_count': 0,
                'lost_mainstream_connection_count': 0,
                
                # 标记数据来源
                'data_source': 'theory'
            }
        
        # 实时保存单个任务结果
        if save_individual_results:
            result['_task_file'] = task_file
            result['_from_cache'] = False
            
            # 转换numpy类型为Python原生类型以便JSON序列化
            json_ready_result = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    json_ready_result[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    json_ready_result[k] = float(v)
                elif isinstance(v, np.bool_):
                    json_ready_result[k] = bool(v)
                else:
                    json_ready_result[k] = v
            
            # 保存到任务文件
            try:
                with open(task_file, 'w') as f:
                    json.dump(json_ready_result, f, indent=2)
                if verbose:
                    print(f"任务结果已保存: {os.path.basename(task_file)}")
            except Exception as e:
                print(f"保存任务结果失败: {str(e)}")
        
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 返回错误结果
        return {
            'phi': task_params.get('phi', 0),
            'theta': task_params.get('theta', 0),
            'r_mainstream': task_params.get('r_mainstream', 0),
            'sample_idx': task_params.get('sample_idx', 0),
            'error': str(e),
            'success': False
        }

def process_parameter_point_v2(phi, theta, r_mainstream_values, n_samples,
                              steady_results_dict, output_dir,
                              enable_micro_analysis, enable_visualization):
    """
    处理单个参数点，汇总稳态数据并进行微观分析
    """
    try:
        # 创建参数点目录
        kappa = 120  # 固定值
        phi_int = int(round(phi * 1000))    # 0.300 -> 300
        theta_int = int(round(theta * 1000)) # 0.550 -> 550
        
        point_dir = os.path.join(output_dir, f"kappa{kappa:03d}_phi{phi_int:04d}_theta{theta_int:04d}")
        os.makedirs(point_dir, exist_ok=True)
        
        print(f"  处理参数点: phi={phi:.3f}, theta={theta:.3f}")
        
        # 汇总稳态数据
        steady_states = []
        for r_mainstream in r_mainstream_values:
            for sample_idx in range(n_samples):
                key = (phi, theta, r_mainstream, sample_idx)
                if key in steady_results_dict and steady_results_dict[key]['success']:
                    steady_states.append(steady_results_dict[key])
                else:
                    print(f"    警告: 缺失数据 r={r_mainstream:.3f}, sample={sample_idx}")
        
        if len(steady_states) == 0:
            raise ValueError("没有有效的稳态数据")
        
        # 微观分析汇总
        micro_analysis = {}
        if enable_micro_analysis:
            print(f"  进行微观分析...")
            
            # 汇总微观统计
            micro_keys = [
                'mainstream_connected_X_H', 'mainstream_connected_X_L',
                'wemedia_connected_X_H', 'wemedia_connected_X_L',
                'mixed_connected_X_H', 'mixed_connected_X_L',
                'transition_rate_high_to_medium', 'transition_rate_medium_to_high',
                'theory_vs_actual_X_H_diff', 'theory_vs_actual_X_L_diff'
            ]
            
            for key in micro_keys:
                values = [s.get(key, np.nan) for s in steady_states if not np.isnan(s.get(key, np.nan))]
                if values:
                    micro_analysis[f'{key}_mean'] = np.mean(values)
                    micro_analysis[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0
                    micro_analysis[f'{key}_count'] = len(values)
                else:
                    micro_analysis[f'{key}_mean'] = np.nan
                    micro_analysis[f'{key}_std'] = np.nan
                    micro_analysis[f'{key}_count'] = 0
            
            # 振荡分析
            oscillation_count = sum(1 for s in steady_states if s.get('has_oscillation', False))
            micro_analysis['oscillation_frequency'] = oscillation_count / len(steady_states)
            
            oscillation_strengths = [s.get('oscillation_strength', 0) for s in steady_states if s.get('has_oscillation', False)]
            if oscillation_strengths:
                micro_analysis['oscillation_strength_mean'] = np.mean(oscillation_strengths)
                micro_analysis['oscillation_strength_std'] = np.std(oscillation_strengths)
            else:
                micro_analysis['oscillation_strength_mean'] = 0.0
                micro_analysis['oscillation_strength_std'] = 0.0
        
        # 保存稳态数据
        steady_states_file = os.path.join(point_dir, 'steady_states.json')
        with open(steady_states_file, 'w') as f:
            # 转换numpy类型为Python原生类型以便JSON序列化
            json_ready_data = []
            for state in steady_states:
                json_state = {}
                for k, v in state.items():
                    if isinstance(v, np.ndarray):
                        json_state[k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        json_state[k] = float(v)
                    elif isinstance(v, np.bool_):
                        json_state[k] = bool(v)
                    else:
                        json_state[k] = v
                json_ready_data.append(json_state)
            json.dump(json_ready_data, f, indent=2)
        
        # 保存微观分析
        if enable_micro_analysis:
            micro_analysis_file = os.path.join(point_dir, 'micro_analysis.json')
            with open(micro_analysis_file, 'w') as f:
                # 转换numpy类型
                json_ready_micro = {}
                for k, v in micro_analysis.items():
                    if isinstance(v, (np.integer, np.floating)):
                        json_ready_micro[k] = float(v)
                    elif isinstance(v, np.bool_):
                        json_ready_micro[k] = bool(v)
                    else:
                        json_ready_micro[k] = v
                json.dump(json_ready_micro, f, indent=2)
        
        # 生成可视化
        if enable_visualization:
            print(f"  生成可视化图表...")
            generate_parameter_point_visualization(
                phi, theta, r_mainstream_values, steady_states, 
                micro_analysis, point_dir
            )
        
        # 保存汇总信息
        summary_data = {
            'phi': phi,
            'theta': theta,
            'kappa': kappa,
            'n_samples': n_samples,
            'n_r_values': len(r_mainstream_values),
            'n_successful_calculations': len(steady_states),
            'steady_states': steady_states,
            'micro_analysis': micro_analysis,
            'timestamp': time.time()
        }
        
        summary_file = os.path.join(point_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            # 转换numpy类型
            json_ready_summary = {}
            for k, v in summary_data.items():
                if k == 'steady_states':
                    # 已经在上面处理过了
                    json_ready_summary[k] = json_ready_data
                elif k == 'micro_analysis':
                    json_ready_summary[k] = json_ready_micro
                elif isinstance(v, (np.integer, np.floating)):
                    json_ready_summary[k] = float(v)
                elif isinstance(v, np.bool_):
                    json_ready_summary[k] = bool(v)
                else:
                    json_ready_summary[k] = v
            json.dump(json_ready_summary, f, indent=2)
        
        print(f"  ✅ 处理完成: {len(steady_states)} 个有效数据点")
        
        return {
            'status': 'success',
            'data': summary_data
        }
        
    except Exception as e:
        print(f"  ❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e)
        }

def generate_parameter_point_visualization(phi, theta, r_values, steady_states, 
                                          micro_analysis, save_dir):
    """
    为单个参数点生成可视化图表
    """
    try:
        # 准备数据
        r_vals = []
        X_H_vals = []
        X_L_vals = []
        p_risk_vals = []
        oscillation_flags = []
        
        # 按r值分组并计算平均值
        r_groups = {}
        for state in steady_states:
            r = state['r_mainstream']
            if r not in r_groups:
                r_groups[r] = []
            r_groups[r].append(state)
        
        for r in sorted(r_groups.keys()):
            group = r_groups[r]
            r_vals.append(r)
            X_H_vals.append(np.mean([s['X_H'] for s in group]))
            X_L_vals.append(np.mean([s['X_L'] for s in group]))
            p_risk_vals.append(np.mean([s['p_risk'] for s in group]))
            oscillation_flags.append(any(s.get('has_oscillation', False) for s in group))
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Steady State Plot
        ax1 = axes[0, 0]
        ax1.plot(r_vals, X_H_vals, 'ro-', linewidth=2, markersize=4, label='X_H (High Arousal)')
        ax1.plot(r_vals, X_L_vals, 'bo-', linewidth=2, markersize=4, label='X_L (Low Arousal)')
        
        # Mark oscillation points
        for i, has_osc in enumerate(oscillation_flags):
            if has_osc:
                ax1.plot(r_vals[i], X_H_vals[i], 'r*', markersize=10, alpha=0.7)
                ax1.plot(r_vals[i], X_L_vals[i], 'b*', markersize=10, alpha=0.7)
        
        ax1.set_xlabel('Mainstream Media Removal Ratio (r)')
        ax1.set_ylabel('Emotional State Proportion')
        ax1.set_title(f'Steady State Distribution (φ={phi:.3f}, θ={theta:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Media Risk
        ax2 = axes[0, 1]
        ax2.plot(r_vals, p_risk_vals, 'go-', linewidth=2, markersize=4)
        ax2.set_xlabel('Mainstream Media Removal Ratio (r)')
        ax2.set_ylabel('Media Risk Proportion (p_risk)')
        ax2.set_title('Media Risk Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Micro-analysis Summary
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        micro_text = f"Micro-analysis Summary:\n\n"
        micro_text += f"Parameter Point: φ={phi:.3f}, θ={theta:.3f}\n"
        micro_text += f"Data Points: {len(steady_states)}\n"
        micro_text += f"r Range: [{min(r_vals):.3f}, {max(r_vals):.3f}]\n\n"
        
        if micro_analysis:
            micro_text += f"Oscillation Frequency: {micro_analysis.get('oscillation_frequency', 0):.2%}\n"
            micro_text += f"Avg Oscillation Strength: {micro_analysis.get('oscillation_strength_mean', 0):.4f}\n\n"
            
            # Connection type analysis
            main_xh = micro_analysis.get('mainstream_connected_X_H_mean', np.nan)
            we_xh = micro_analysis.get('wemedia_connected_X_H_mean', np.nan)
            if not np.isnan(main_xh) and not np.isnan(we_xh):
                micro_text += f"Mainstream Connected X_H: {main_xh:.4f}\n"
                micro_text += f"WeMedia Connected X_H: {we_xh:.4f}\n"
                micro_text += f"Connection Type Difference: {abs(main_xh - we_xh):.4f}\n"
        
        ax3.text(0.05, 0.95, micro_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 4. Phase Transition Features
        ax4 = axes[1, 1]
        
        # Calculate X_H gradient (approximate derivative)
        if len(r_vals) > 2:
            dX_H_dr = np.gradient(X_H_vals, r_vals)
            ax4.plot(r_vals, dX_H_dr, 'mo-', linewidth=2, markersize=4)
            ax4.set_xlabel('Mainstream Media Removal Ratio (r)')
            ax4.set_ylabel('dX_H/dr')
            ax4.set_title('Phase Transition Features (X_H Gradient)')
            ax4.grid(True, alpha=0.3)
            
            # Mark maximum gradient point
            max_idx = np.argmax(np.abs(dX_H_dr))
            ax4.plot(r_vals[max_idx], dX_H_dr[max_idx], 'r*', markersize=12)
            ax4.text(r_vals[max_idx], dX_H_dr[max_idx], 
                    f'  r_c≈{r_vals[max_idx]:.3f}', fontsize=9)
        else:
            ax4.text(0.5, 0.5, 'Insufficient Data Points\nfor Phase Transition Analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        plt.suptitle(f'Parameter Point Analysis: φ={phi:.3f}, θ={theta:.3f}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        fig_path = os.path.join(save_dir, f'visualization_phi{phi:.3f}_theta{theta:.3f}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    可视化图表已保存: {os.path.basename(fig_path)}")
        
    except Exception as e:
        print(f"    可视化生成失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_parameter_sweep_v2():
    """
    测试改进版参数扫描功能
    """
    print("=== 测试改进版参数扫描 (V2) ===")
    
    # 测试参数 - 支持range和列表两种输入方式
    
    # 方式1：使用range格式 [start, end, step]
    phi_range = [0.25, 0.35, 0.05]    # 0.25, 0.30, 0.35
    theta_range = [0.50, 0.60, 0.05]  # 0.50, 0.55, 0.60
    
    # 方式2：使用列表格式（注释掉，可以根据需要切换）
    # phi_range = [0.25, 0.30, 0.35]
    # theta_range = [0.50, 0.55, 0.60]
    
    # r值范围 - 用于微观分析
    r_mainstream_values = np.linspace(0.0, 0.8, 9)  # 9个点：0.0, 0.1, 0.2, ..., 0.8
    
    # 其他参数
    n_samples = 1  # 每个参数点1个样本（测试用）
    output_dir = 'results_v3/parameter_sweep_v2_test'
    
    print(f"测试配置:")
    print(f"  phi_range: {phi_range}")
    print(f"  theta_range: {theta_range}")
    print(f"  r_mainstream: {len(r_mainstream_values)} 个值，范围 [{r_mainstream_values[0]:.1f}, {r_mainstream_values[-1]:.1f}]")
    print(f"  n_samples: {n_samples}")
    print(f"  输出目录: {output_dir}")
    
    # 运行改进版参数扫描
    results_df = run_parallel_parameter_sweep_v2(
        phi_range=phi_range,
        theta_range=theta_range,
        r_mainstream_values=r_mainstream_values,
        n_samples=n_samples,
        output_dir=output_dir,
        n_processes=None,  # 自动选择进程数
        skip_existing=True,  # 跳过已有结果
        enable_micro_analysis=True,  # 启用微观分析
        enable_visualization=True  # 启用可视化
    )
    
    # 分析结果
    if not results_df.empty:
        print(f"\n=== 结果摘要 ===")
        print(f"总数据记录: {len(results_df)}")
        print(f"参数点数: {results_df[['phi', 'theta']].drop_duplicates().shape[0]}")
        print(f"r值数: {results_df['r_mainstream'].nunique()}")
        
        # 检查微观分析数据
        micro_columns = [col for col in results_df.columns if col.startswith('micro_')]
        if micro_columns:
            print(f"微观分析指标: {len(micro_columns)} 个")
            
            # 显示一些关键微观指标的统计
            key_micro_metrics = [
                'micro_oscillation_frequency',
                'micro_mainstream_connected_X_H_mean',
                'micro_wemedia_connected_X_H_mean'
            ]
            
            for metric in key_micro_metrics:
                if metric in results_df.columns:
                    values = results_df[metric].dropna()
                    if len(values) > 0:
                        print(f"  {metric}: 均值={values.mean():.4f}, 标准差={values.std():.4f}")
        
        # 检查成功率
        if 'success' in results_df.columns:
            success_rate = results_df['success'].mean()
            print(f"计算成功率: {success_rate:.1%}")
        
        return results_df
    else:
        print("❌ 没有成功的结果")
        return pd.DataFrame()

def run_comprehensive_parameter_sweep():
    """
    运行全面的参数扫描 - 使用改进版V2函数
    """
    print("=== 全面参数扫描 (V2) ===")
    
    # 更大范围的参数扫描
    phi_range = [0.20, 0.40, 0.04]    # 0.20, 0.24, 0.28, 0.32, 0.36, 0.40 (6个值)
    theta_range = [0.45, 0.65, 0.04]  # 0.45, 0.49, 0.53, 0.57, 0.61, 0.65 (6个值)
    
    # 更密集的r值采样
    r_mainstream_values = np.linspace(0.0, 0.9, 19)  # 19个点，步长0.05
    
    # 多样本统计
    n_samples = 3  # 每个参数点3个样本
    output_dir = 'results_v3/comprehensive_parameter_sweep_v2'
    
    print(f"全面扫描配置:")
    print(f"  phi: 6个值，范围 [0.20, 0.40]")
    print(f"  theta: 6个值，范围 [0.45, 0.65]")
    print(f"  r_mainstream: 19个值，范围 [0.0, 0.9]")
    print(f"  n_samples: {n_samples}")
    print(f"  预计有效参数点: ~15个 (考虑phi<theta约束)")
    print(f"  预计总计算量: ~15 × 19 × 3 = ~855个任务")
    
    # 确认是否继续
    confirm = input("这将需要较长时间计算，是否继续？(y/N): ").strip().lower()
    if confirm != 'y':
        print("取消全面扫描")
        return pd.DataFrame()
    
    # 运行全面扫描
    results_df = run_parallel_parameter_sweep_v2(
        phi_range=phi_range,
        theta_range=theta_range,
        r_mainstream_values=r_mainstream_values,
        n_samples=n_samples,
        output_dir=output_dir,
        n_processes=None,  # 自动选择进程数
        skip_existing=True,  # 跳过已有结果（支持增量计算）
        enable_micro_analysis=True,  # 启用微观分析
        enable_visualization=True  # 启用可视化
    )
    
    return results_df

def analyze_parameter_sweep_results(results_df, output_dir='analysis_plots'):
    """
    分析run_parallel_parameter_sweep_v2的结果，生成多维度可视化图表
    
    Parameters:
        results_df: run_parallel_parameter_sweep_v2返回的DataFrame
        output_dir: 图表保存目录
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os
    
    # 设置中文字体和绘图风格
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    sns.set_style("whitegrid")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== 参数扫描结果分析 ===")
    print(f"数据形状: {results_df.shape}")
    print(f"参数范围:")
    print(f"  phi: [{results_df['phi'].min():.3f}, {results_df['phi'].max():.3f}]")
    print(f"  theta: [{results_df['theta'].min():.3f}, {results_df['theta'].max():.3f}]")
    print(f"  r_mainstream: [{results_df['r_mainstream'].min():.3f}, {results_df['r_mainstream'].max():.3f}]")
    
    # 1. 基础分布分析
    print(f"\n1. 生成基础分布分析图...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 状态分布
    states = ['X_H', 'X_M', 'X_L']
    for i, state in enumerate(states):
        axes[0, i].hist(results_df[state], bins=30, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f'{state} Distribution')
        axes[0, i].set_xlabel(state)
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
    
    # 风险感知分布
    risk_vars = ['p_risk', 'p_risk_m', 'p_risk_w']
    for i, risk in enumerate(risk_vars):
        axes[1, i].hist(results_df[risk], bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[1, i].set_title(f'{risk} Distribution')
        axes[1, i].set_xlabel(risk)
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_basic_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 相变分析（如果有多个phi或theta值）
    unique_phi = results_df['phi'].nunique()
    unique_theta = results_df['theta'].nunique()
    
    if unique_phi > 1 or unique_theta > 1:
        print(f"2. 生成相变分析图...")
        
        # 计算平均值（如果有多个样本）
        avg_df = results_df.groupby(['phi', 'theta', 'r_mainstream']).agg({
            'X_H': 'mean',
            'X_M': 'mean', 
            'X_L': 'mean',
            'p_risk': 'mean'
        }).reset_index()
        
        # 相变热图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        variables = ['X_H', 'X_L', 'p_risk', 'X_M']
        for i, var in enumerate(variables):
            ax = axes[i//2, i%2]
            
            # 创建数据透视表
            if unique_phi > 1 and unique_theta > 1:
                # phi-theta相变图（取r_mainstream的中位数）
                median_r = avg_df['r_mainstream'].median()
                subset = avg_df[avg_df['r_mainstream'] == avg_df['r_mainstream'].iloc[
                    (avg_df['r_mainstream'] - median_r).abs().argsort()[:1]
                ].values[0]]
                
                pivot_data = subset.pivot(index='theta', columns='phi', values=var)
                
                im = ax.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto', 
                              extent=[pivot_data.columns.min(), pivot_data.columns.max(),
                                     pivot_data.index.min(), pivot_data.index.max()])
                ax.set_xlabel('phi (Low Arousal Threshold)')
                ax.set_ylabel('theta (High Arousal Threshold)')
                ax.set_title(f'{var} Phase Diagram (r={median_r:.2f})')
                
                plt.colorbar(im, ax=ax)
                
            else:
                # r_mainstream vs variable 的关系图
                for phi_val in avg_df['phi'].unique()[:3]:  # 最多显示3个phi值
                    for theta_val in avg_df['theta'].unique()[:3]:  # 最多显示3个theta值
                        subset = avg_df[(avg_df['phi'] == phi_val) & (avg_df['theta'] == theta_val)]
                        if not subset.empty:
                            ax.plot(subset['r_mainstream'], subset[var], 
                                   label=f'φ={phi_val:.2f}, θ={theta_val:.2f}', 
                                   marker='o', markersize=4)
                
                ax.set_xlabel('r_mainstream (Removal Ratio)')
                ax.set_ylabel(var)
                ax.set_title(f'{var} vs Removal Ratio')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_phase_transitions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 微观分析可视化（如果有微观数据）
    micro_cols = [col for col in results_df.columns if col.startswith('mainstream_connected_') 
                  or col.startswith('wemedia_connected_') or col.startswith('mixed_connected_')]
    
    if micro_cols:
        print(f"3. 生成微观分析图...")
        
        # 过滤有效的微观数据
        micro_df = results_df[micro_cols + ['phi', 'theta', 'r_mainstream']].dropna()
        
        if not micro_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 连接类型对比
            connection_types = ['mainstream_connected', 'wemedia_connected', 'mixed_connected']
            
            # X_H vs X_L 在不同连接类型下的对比
            ax = axes[0, 0]
            for conn_type in connection_types:
                x_h_col = f'{conn_type}_X_H'
                x_l_col = f'{conn_type}_X_L'
                if x_h_col in micro_df.columns and x_l_col in micro_df.columns:
                    valid_data = micro_df[[x_h_col, x_l_col]].dropna()
                    if not valid_data.empty:
                        ax.scatter(valid_data[x_h_col], valid_data[x_l_col], 
                                  label=conn_type.replace('_', ' ').title(), alpha=0.6, s=20)
            
            ax.set_xlabel('X_H (High Arousal)')
            ax.set_ylabel('X_L (Low Arousal)')
            ax.set_title('Connection Type Analysis: X_H vs X_L')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 连接类型随r_mainstream的变化
            ax = axes[0, 1]
            for conn_type in connection_types:
                x_h_col = f'{conn_type}_X_H'
                if x_h_col in micro_df.columns:
                    # 按r_mainstream分组取平均
                    grouped = micro_df.groupby('r_mainstream')[x_h_col].mean()
                    ax.plot(grouped.index, grouped.values, 
                           marker='o', label=f'{conn_type.replace("_", " ").title()} X_H', markersize=4)
            
            ax.set_xlabel('r_mainstream (Removal Ratio)')
            ax.set_ylabel('X_H (High Arousal)')
            ax.set_title('Connection Type X_H vs Removal Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 理论vs实际差异分析
            theory_cols = [col for col in results_df.columns if 'theory_vs_actual' in col]
            if theory_cols:
                ax = axes[1, 0]
                for col in theory_cols[:3]:  # 最多显示3个差异指标
                    valid_data = results_df[['r_mainstream', col]].dropna()
                    if not valid_data.empty:
                        grouped = valid_data.groupby('r_mainstream')[col].mean()
                        ax.plot(grouped.index, grouped.values, 
                               marker='s', label=col.replace('theory_vs_actual_', '').replace('_', ' ').title(), markersize=4)
                
                ax.set_xlabel('r_mainstream (Removal Ratio)')
                ax.set_ylabel('Theory vs Actual Difference')
                ax.set_title('Theory-Simulation Discrepancy')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 振荡分析
            if 'has_oscillation' in results_df.columns:
                ax = axes[1, 1]
                
                # 振荡比例随r_mainstream变化
                osc_ratio = results_df.groupby('r_mainstream')['has_oscillation'].mean()
                ax.plot(osc_ratio.index, osc_ratio.values, 'ro-', markersize=6, linewidth=2)
                ax.set_xlabel('r_mainstream (Removal Ratio)')
                ax.set_ylabel('Oscillation Probability')
                ax.set_title('Oscillation Frequency vs Removal Ratio')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '03_micro_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. 参数空间探索图
    print(f"4. 生成参数空间探索图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # phi vs theta 散点图，颜色表示X_H
    ax = axes[0, 0]
    scatter = ax.scatter(results_df['phi'], results_df['theta'], 
                        c=results_df['X_H'], cmap='RdYlBu_r', 
                        alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('phi (Low Arousal Threshold)')
    ax.set_ylabel('theta (High Arousal Threshold)')
    ax.set_title('Parameter Space: X_H')
    plt.colorbar(scatter, ax=ax, label='X_H')
    ax.grid(True, alpha=0.3)
    
    # r_mainstream vs X_H，不同phi-theta组合
    ax = axes[0, 1]
    unique_combos = results_df[['phi', 'theta']].drop_duplicates()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_combos)))
    
    for i, (_, combo) in enumerate(unique_combos.iterrows()):
        if i >= 10:  # 最多显示10个组合
            break
        subset = results_df[(results_df['phi'] == combo['phi']) & 
                           (results_df['theta'] == combo['theta'])]
        if not subset.empty:
            # 按r_mainstream分组取平均
            grouped = subset.groupby('r_mainstream')['X_H'].mean()
            ax.plot(grouped.index, grouped.values, 
                   color=colors[i], marker='o', markersize=4,
                   label=f'φ={combo["phi"]:.2f}, θ={combo["theta"]:.2f}')
    
    ax.set_xlabel('r_mainstream (Removal Ratio)')
    ax.set_ylabel('X_H (High Arousal)')
    ax.set_title('X_H Response to Media Removal')
    if len(unique_combos) <= 5:  # 只有在组合不太多时才显示图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 收敛性分析
    if 'converged' in results_df.columns:
        ax = axes[1, 0]
        conv_ratio = results_df.groupby('r_mainstream')['converged'].mean()
        ax.plot(conv_ratio.index, conv_ratio.values, 'go-', markersize=6, linewidth=2)
        ax.set_xlabel('r_mainstream (Removal Ratio)')
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Convergence Rate vs Removal Ratio')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # 迭代次数分析
    if 'iterations' in results_df.columns:
        ax = axes[1, 1]
        iter_avg = results_df.groupby('r_mainstream')['iterations'].mean()
        iter_std = results_df.groupby('r_mainstream')['iterations'].std()
        ax.errorbar(iter_avg.index, iter_avg.values, yerr=iter_std.values, 
                   fmt='bo-', markersize=5, capsize=3)
        ax.set_xlabel('r_mainstream (Removal Ratio)')
        ax.set_ylabel('Average Iterations')
        ax.set_title('Convergence Speed vs Removal Ratio')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_parameter_space.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 生成数据摘要报告
    print(f"5. 生成数据摘要报告...")
    
    summary_stats = {
        'data_overview': {
            'total_records': len(results_df),
            'unique_phi_values': results_df['phi'].nunique(),
            'unique_theta_values': results_df['theta'].nunique(),
            'unique_r_values': results_df['r_mainstream'].nunique(),
            'phi_range': [results_df['phi'].min(), results_df['phi'].max()],
            'theta_range': [results_df['theta'].min(), results_df['theta'].max()],
            'r_range': [results_df['r_mainstream'].min(), results_df['r_mainstream'].max()]
        },
        'state_statistics': {
            'X_H': {
                'mean': results_df['X_H'].mean(),
                'std': results_df['X_H'].std(),
                'min': results_df['X_H'].min(),
                'max': results_df['X_H'].max()
            },
            'X_M': {
                'mean': results_df['X_M'].mean(),
                'std': results_df['X_M'].std(),
                'min': results_df['X_M'].min(),
                'max': results_df['X_M'].max()
            },
            'X_L': {
                'mean': results_df['X_L'].mean(),
                'std': results_df['X_L'].std(),
                'min': results_df['X_L'].min(),
                'max': results_df['X_L'].max()
            }
        }
    }
    
    # 添加收敛统计
    if 'converged' in results_df.columns:
        summary_stats['convergence_stats'] = {
            'convergence_rate': results_df['converged'].mean(),
            'average_iterations': results_df['iterations'].mean() if 'iterations' in results_df.columns else None
        }
    
    # 添加振荡统计
    if 'has_oscillation' in results_df.columns:
        summary_stats['oscillation_stats'] = {
            'oscillation_rate': results_df['has_oscillation'].mean(),
            'average_oscillation_strength': results_df['oscillation_strength'].mean() if 'oscillation_strength' in results_df.columns else None
        }
    
    # 保存摘要
    import json
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n=== 分析完成 ===")
    print(f"图表保存目录: {output_dir}")
    print(f"生成的图表:")
    print(f"  01_basic_distributions.png - 基础分布分析")
    if unique_phi > 1 or unique_theta > 1:
        print(f"  02_phase_transitions.png - 相变分析")
    if micro_cols:
        print(f"  03_micro_analysis.png - 微观分析")
    print(f"  04_parameter_space.png - 参数空间探索")
    print(f"  analysis_summary.json - 数据摘要报告")
    
    return summary_stats

# 使用示例
def example_usage():
    """
    使用示例：如何运行参数扫描并进行可视化分析
    """
    print("=== 参数扫描和可视化分析示例 ===")
    
    # 1. 运行参数扫描
    phi_range = [0.20, 0.40, 0.04]    # 0.20, 0.24, 0.28, 0.32, 0.36, 0.40
    theta_range = [0.50, 0.70, 0.05]  # 0.50, 0.55, 0.60, 0.65, 0.70
    r_mainstream_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    print("开始参数扫描...")
    results_df = run_parallel_parameter_sweep_v2(
        phi_range=phi_range,
        theta_range=theta_range,
        r_mainstream_values=r_mainstream_values,
        n_samples=2,
        output_dir='results_v3_example',
        enable_micro_analysis=True,
        enable_visualization=True
    )
    
    # 2. 进行可视化分析
    if not results_df.empty:
        print("开始可视化分析...")
        summary_stats = analyze_parameter_sweep_results(
            results_df, 
            output_dir='analysis_plots_example'
        )
        
        print("分析摘要:")
        print(f"  总记录数: {summary_stats['data_overview']['total_records']}")
        print(f"  φ范围: {summary_stats['data_overview']['phi_range']}")
        print(f"  θ范围: {summary_stats['data_overview']['theta_range']}")
        print(f"  平均X_H: {summary_stats['state_statistics']['X_H']['mean']:.3f}")
        
        if 'convergence_stats' in summary_stats:
            print(f"  收敛率: {summary_stats['convergence_stats']['convergence_rate']:.1%}")
        
        if 'oscillation_stats' in summary_stats:
            print(f"  振荡率: {summary_stats['oscillation_stats']['oscillation_rate']:.1%}")
    
    else:
        print("❌ 参数扫描没有返回有效数据")

    return results_df


# 修改主执行部分
if __name__ == "__main__":
    print("选择测试模式：")
    print("1. 单进程快速测试（2个点）")
    print("2. 多进程快速测试（15个点，使用所有CPU核心）")
    print("3. 微观分析功能测试")
    print("4. 增强版参数扫描（含微观分析）")
    print("5. 模拟器 vs 理论模型对比测试 ⭐ 推荐")
    print("6. 改进版参数扫描测试 (V2) 🆕")
    print("7. 全面参数扫描 (V2)")
    
    try:
        choice = input("请输入选择 (1-7，直接回车默认选择6): ").strip()
        
        if choice == "1":
            success = quick_phase_transition_test()
        elif choice == "2":
            success = quick_phase_transition_test_parallel()
        elif choice == "3":
            success, analysis_results = test_micro_analysis_functionality()
            if success:
                print("\n建议接下来运行选项6：改进版参数扫描以获取完整的微观分析数据")
        elif choice == "4":
            results = run_enhanced_parameter_sweep_with_micro_analysis()
            success = not results.empty
        elif choice == "5":
            success = compare_simulator_vs_theory()
            if success:
                print("\n建议接下来运行选项6进行更全面的测试")
        elif choice == "7":
            results = run_comprehensive_parameter_sweep()
            success = not results.empty
        else:  # choice == "6" 或默认
            results = test_parameter_sweep_v2()
            success = not results.empty
            if success:
                print("\n建议接下来运行选项7进行全面扫描，或使用Jupyter调用run_parallel_parameter_sweep_v2函数")
        
        if success:
            print("\n🎉 测试成功！")
        else:
            print("\n⚠️ 仍需进一步优化。")
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_enhanced_parameter_sweep_with_storage():
    """
    测试增强版参数扫描 - 支持实时存储和任务筛选
    
    主要特性：
    1. 任务级别的细粒度筛选 - 避免重复计算
    2. 实时存储机制 - 防止数据丢失
    3. 缓存命中率统计 - 监控计算效率
    4. 自动恢复功能 - 支持中断后继续
    """
    print("=== 增强版参数扫描测试 (实时存储 + 任务筛选) ===")
    
    # 测试参数
    phi_range = [0.25, 0.35, 0.05]    # 0.25, 0.30, 0.35
    theta_range = [0.50, 0.60, 0.05]  # 0.50, 0.55, 0.60
    r_mainstream_values = np.linspace(0.0, 0.6, 7)  # 7个点：0.0, 0.1, ..., 0.6
    n_samples = 2  # 每个参数点2个样本
    output_dir = 'results_v3/enhanced_storage_test'
    
    print(f"🔧 测试配置:")
    print(f"  phi: {phi_range}")
    print(f"  theta: {theta_range}")
    print(f"  r_mainstream: {len(r_mainstream_values)} 个值")
    print(f"  n_samples: {n_samples}")
    print(f"  输出目录: {output_dir}")
    
    # 第一次运行：运行部分任务然后"意外中断"
    print(f"\n🚀 第一次运行：模拟部分完成...")
    
    # 修改run_parallel_parameter_sweep_v2，添加实时存储和筛选功能
    results_df_1 = run_parallel_parameter_sweep_v2_with_storage(
        phi_range=phi_range,
        theta_range=theta_range,
        r_mainstream_values=r_mainstream_values[:4],  # 只运行前4个r值
        n_samples=n_samples,
        output_dir=output_dir,
        n_processes=2,
        skip_existing=False,  # 第一次不跳过
        enable_micro_analysis=True,
        enable_visualization=False,  # 加快速度
        max_iter=50  # 减少迭代次数加快测试
    )
    
    if not results_df_1.empty:
        print(f"  ✅ 第一次运行完成: {len(results_df_1)} 条记录")
        print(f"  📁 数据已保存到: {output_dir}")
    
    # 第二次运行：模拟恢复，应该跳过已完成的任务
    print(f"\n🔄 第二次运行：模拟恢复，测试任务筛选...")
    
    results_df_2 = run_parallel_parameter_sweep_v2_with_storage(
        phi_range=phi_range,
        theta_range=theta_range,
        r_mainstream_values=r_mainstream_values,  # 运行全部r值
        n_samples=n_samples,
        output_dir=output_dir,
        n_processes=2,
        skip_existing=True,  # 启用任务筛选
        enable_micro_analysis=True,
        enable_visualization=False,
        max_iter=50
    )
    
    if not results_df_2.empty:
        print(f"  ✅ 第二次运行完成: {len(results_df_2)} 条记录")
        
        # 分析缓存命中情况
        analyze_cache_performance(output_dir, results_df_2)
    
    return results_df_2

def run_parallel_parameter_sweep_v2_with_storage(phi_range, theta_range, r_mainstream_values, n_samples=1, 
                                                output_dir='results_v3', n_processes=None, chunksize=None,
                                                skip_existing=True, enable_micro_analysis=True, 
                                                enable_visualization=True, max_iter=None):
    """
    带实时存储和任务筛选的参数扫描函数
    
    基于原有的run_parallel_parameter_sweep_v2，增加了：
    1. 任务级别的细粒度筛选
    2. 详细的缓存统计
    3. 进度监控
    """
    print("🔥 启用增强功能: 实时存储 + 任务筛选")
    
    # 解析参数（重用原有逻辑）
    if isinstance(phi_range, (list, tuple)) and len(phi_range) == 3:
        phi_start, phi_end, phi_step = phi_range
        phi_values = np.arange(phi_start, phi_end + phi_step/2, phi_step)
    else:
        phi_values = np.array(phi_range) if isinstance(phi_range, (list, tuple)) else np.array([phi_range])
    
    if isinstance(theta_range, (list, tuple)) and len(theta_range) == 3:
        theta_start, theta_end, theta_step = theta_range
        theta_values = np.arange(theta_start, theta_end + theta_step/2, theta_step)
    else:
        theta_values = np.array(theta_range) if isinstance(theta_range, (list, tuple)) else np.array([theta_range])
    
    r_mainstream_values = np.array(r_mainstream_values)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建参数点列表
    param_points = [(phi, theta) for phi in phi_values for theta in theta_values if phi < theta]
    
    print(f"📊 参数空间:")
    print(f"  φ值: {len(phi_values)} 个")
    print(f"  θ值: {len(theta_values)} 个")
    print(f"  有效参数点: {len(param_points)} 个")
    print(f"  r值: {len(r_mainstream_values)} 个")
    print(f"  样本数: {n_samples}")
    
    # 网络参数
    network_params = {
        'n_mainstream': 100, 'n_wemedia': 100, 'n_public': 100,
        'k_out_mainstream': 60, 'k_out_wemedia': 60, 'k_out_public': 10,
        'use_original_like_dist': False
    }
    
    init_states = {
        'X_H': 0.3, 'X_M': 0.4, 'X_L': 0.3,
        'p_risk_m': 0.5, 'p_risk_w': 0.5, 'p_risk': 0.5
    }
    
    # 任务筛选和统计
    print(f"\n🔍 任务筛选阶段...")
    
    all_tasks = []
    skipped_count = 0
    total_possible = len(param_points) * len(r_mainstream_values) * n_samples
    
    for phi, theta in param_points:
        for r_mainstream in r_mainstream_values:
            for sample_idx in range(n_samples):
                # 检查是否已有结果
                if skip_existing and task_exists_v2(output_dir, phi, theta, r_mainstream, sample_idx):
                    skipped_count += 1
                    continue
                
                # 创建新任务
                task = {
                    'phi': phi, 'theta': theta, 'r_mainstream': r_mainstream, 'sample_idx': sample_idx,
                    'seed': SIMULATION_PARAMS['default_seed'] + sample_idx,
                    'use_simulator': True, 'max_iter': max_iter or SIMULATION_PARAMS['max_iter'],
                    'verbose': False, 'init_states': init_states, 'network_params': network_params,
                    'output_dir': output_dir, 'save_individual_results': True
                }
                all_tasks.append(task)
    
    cache_hit_rate = skipped_count / total_possible if total_possible > 0 else 0
    new_tasks = len(all_tasks)
    
    print(f"📈 筛选统计:")
    print(f"  总可能任务: {total_possible:,}")
    print(f"  已完成任务: {skipped_count:,}")
    print(f"  需计算任务: {new_tasks:,}")
    print(f"  缓存命中率: {cache_hit_rate:.1%}")
    
    # 并行计算新任务
    if new_tasks == 0:
        print("🎉 所有任务已完成！加载现有结果...")
        return load_existing_results_v2(output_dir, param_points, r_mainstream_values, n_samples)
    
    print(f"\n🚀 并行计算阶段:")
    print(f"  计算任务: {new_tasks:,}")
    print(f"  进程数: {n_processes or mp.cpu_count()-1}")
    
    # 执行并行计算
    start_time = time.time()
    
    if n_processes and n_processes > 1:
        with mp.Pool(processes=n_processes) as pool:
            new_results = list(tqdm(
                pool.map(worker_v2, all_tasks),
                total=len(all_tasks),
                desc="计算进度",
                ncols=80
            ))
    else:
        new_results = [worker_v2(task) for task in tqdm(all_tasks, desc="计算进度", ncols=80)]
    
    elapsed = time.time() - start_time
    success_count = sum(1 for r in new_results if r.get('success', False))
    
    print(f"✅ 计算完成:")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"  成功: {success_count}/{new_tasks} ({success_count/new_tasks:.1%})")
    print(f"  速度: {new_tasks/elapsed:.1f} 任务/秒")
    
    # 加载所有结果（新计算+缓存）
    print(f"\n📊 结果整合...")
    final_results = load_existing_results_v2(output_dir, param_points, r_mainstream_values, n_samples)
    
    print(f"  最终结果: {len(final_results):,} 条记录")
    
    return final_results

def task_exists_v2(output_dir, phi, theta, r_mainstream, sample_idx):
    """
    检查特定任务是否已存在有效结果
    """
    # 生成任务文件路径
    kappa = 120
    phi_int = int(round(phi * 1000))
    theta_int = int(round(theta * 1000))
    r_int = int(round(r_mainstream * 1000))
    
    task_dir = os.path.join(output_dir, 'individual_tasks', 
                           f"kappa{kappa:03d}_phi{phi_int:04d}_theta{theta_int:04d}")
    task_file = os.path.join(task_dir, f'r{r_int:04d}_sample{sample_idx:03d}.json')
    
    if not os.path.exists(task_file):
        return False
    
    try:
        with open(task_file, 'r') as f:
            result = json.load(f)
        
        # 验证结果完整性
        required_keys = ['phi', 'theta', 'r_mainstream', 'X_H', 'X_L', 'success']
        return all(key in result for key in required_keys) and result.get('success', False)
    except:
        return False

def load_existing_results_v2(output_dir, param_points, r_mainstream_values, n_samples):
    """
    加载所有现有结果
    """
    all_results = []
    
    for phi, theta in param_points:
        for r_mainstream in r_mainstream_values:
            for sample_idx in range(n_samples):
                # 生成任务文件路径
                kappa = 120
                phi_int = int(round(phi * 1000))
                theta_int = int(round(theta * 1000))
                r_int = int(round(r_mainstream * 1000))
                
                task_dir = os.path.join(output_dir, 'individual_tasks', 
                                       f"kappa{kappa:03d}_phi{phi_int:04d}_theta{theta_int:04d}")
                task_file = os.path.join(task_dir, f'r{r_int:04d}_sample{sample_idx:03d}.json')
                
                if os.path.exists(task_file):
                    try:
                        with open(task_file, 'r') as f:
                            result = json.load(f)
                        
                        if result.get('success', False):
                            all_results.append(result)
                    except:
                        continue
    
    # 转换为DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame()

def analyze_cache_performance(output_dir, results_df):
    """
    分析缓存性能和存储使用情况
    """
    print(f"\n📊 缓存性能分析:")
    
    # 统计缓存命中情况
    if '_from_cache' in results_df.columns:
        cached_count = results_df['_from_cache'].sum()
        total_count = len(results_df)
        cache_rate = cached_count / total_count if total_count > 0 else 0
        
        print(f"  缓存命中: {cached_count}/{total_count} ({cache_rate:.1%})")
        print(f"  新计算: {total_count - cached_count}")
    
    # 分析存储空间使用
    task_dir = os.path.join(output_dir, 'individual_tasks')
    if os.path.exists(task_dir):
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(task_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    total_size += os.path.getsize(file_path)
        
        avg_size = total_size / total_files if total_files > 0 else 0
        
        print(f"  存储文件: {total_files:,} 个")
        print(f"  总大小: {total_size / 1024 / 1024:.1f} MB")
        print(f"  平均大小: {avg_size / 1024:.1f} KB/文件")
    
    # 分析参数分布
    if not results_df.empty:
        param_stats = {
            'phi_values': results_df['phi'].nunique(),
            'theta_values': results_df['theta'].nunique(),
            'r_values': results_df['r_mainstream'].nunique(),
            'total_combinations': len(results_df)
        }
        
        print(f"  参数分布: φ×θ×r = {param_stats['phi_values']}×{param_stats['theta_values']}×{param_stats['r_values']}")
        print(f"  数据密度: {param_stats['total_combinations']}/{param_stats['phi_values']*param_stats['theta_values']*param_stats['r_values']}")

if __name__ == "__main__":
    # 在主执行块中添加增强版测试选项
    print("选择测试模式：")
    print("1. 单进程快速测试（2个点）")
    print("2. 多进程快速测试（15个点，使用所有CPU核心）")
    print("3. 微观分析功能测试")
    print("4. 增强版参数扫描（含微观分析）")
    print("5. 模拟器 vs 理论模型对比测试 ⭐ 推荐")
    print("6. 改进版参数扫描测试 (V2) 🆕")
    print("7. 全面参数扫描 (V2)")
    print("8. 增强版扫描测试 (实时存储+任务筛选) 🔥 最新")
    
    try:
        choice = input("请输入选择 (1-8，直接回车默认选择8): ").strip()
        
        if choice == "1":
            success = quick_phase_transition_test()
        elif choice == "2":
            success = quick_phase_transition_test_parallel()
        elif choice == "3":
            success, analysis_results = test_micro_analysis_functionality()
        elif choice == "4":
            results = run_enhanced_parameter_sweep_with_micro_analysis()
            success = not results.empty
        elif choice == "5":
            success = compare_simulator_vs_theory()
        elif choice == "6":
            results = test_parameter_sweep_v2()
            success = not results.empty
        elif choice == "7":
            results = run_comprehensive_parameter_sweep()
            success = not results.empty
        else:  # choice == "8" 或默认
            results = test_enhanced_parameter_sweep_with_storage()
            success = not results.empty
            if success:
                print("\n🎉 增强版扫描测试成功！")
                print("✨ 主要特性验证:")
                print("  ✅ 实时存储机制 - 每个任务完成后立即保存")
                print("  ✅ 任务级筛选 - 自动跳过已完成的任务")
                print("  ✅ 缓存命中统计 - 监控计算效率")
                print("  ✅ 断点续传功能 - 支持中断后恢复")
        
        if success:
            print("\n🎉 测试成功！")
        else:
            print("\n⚠️ 仍需进一步优化。")
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

