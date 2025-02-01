# Standard library imports
from collections import Counter
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party library imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm  # Progress bar functionality

def meets_error_criteria(error: float, threshold: float = 0.07) -> bool:
    """
    Determine if the given error meets the acceptance criteria.
    
    Args:
        error (float): The calculated error value to evaluate
        threshold (float, optional): Maximum acceptable error threshold. Defaults to 0.07
        
    Returns:
        bool: True if error is below threshold, False otherwise
        
    Example:
        >>> meets_error_criteria(0.05)
        True
        >>> meets_error_criteria(0.08)
        False
    """
    return error < threshold

def simulate_network_for_parameters(network, param, empirical_data, num_seeds=3):
    """
    Run network simulation with given parameters and multiple random seeds.
    Returns summary statistics of simulation results.
    
    Args:
        network: Network object to simulate
        param: Parameter set for simulation
        empirical_data: Empirical data for comparison
        num_seeds: Number of random seeds to use
    """
    errors = []
    for seed in range(num_seeds):
        np.random.seed(seed)
        simulation_results = network.simulate_steps(90, *param)
        i = 0
        pre_w_risk_p = {}
        pre_m_risk_p = {}
        pre_sentiment_high_p = {}
        pre_sentiment_middle_p = {}

        # Calculate weekly averages
        for index in range(1, 90, 7):
            start = index
            end = start + 7
            results_slice = simulation_results[start:end]
            m_media_risk_p = np.mean([step['m_media'].get('R', 0) for step in results_slice])
            w_media_risk_p = np.mean([step['w_media'].get('R', 0) for step in results_slice])
            o_people_high_p = np.mean([step['o_people'].get('H', 0) for step in results_slice])
            o_people_middle_p = np.mean([step['o_people'].get('M', 0) for step in results_slice])
            
            pre_w_risk_p[i] = w_media_risk_p
            pre_m_risk_p[i] = m_media_risk_p
            pre_sentiment_high_p[i] = o_people_high_p
            pre_sentiment_middle_p[i] = o_people_middle_p
            i += 1

        pre_data = {
            'w_risk_p': pre_w_risk_p,
            'm_risk_p': pre_m_risk_p,
            'sentiment_high_p': pre_sentiment_high_p,
            'sentiment_middle_p': pre_sentiment_middle_p
        }
        error = calculate_error(pre_data, empirical_data)
        errors.append(error)

    avg_error = np.mean(errors)
    
    if meets_error_criteria(avg_error):
        return {'result': 'fitting', 'param': param, 'error': avg_error}
    return {'result': 'not fitting', 'param': param, 'error': avg_error}

def calculate_error(simulation_results, empirical_data):
    """
    Calculate error between simulation results and empirical data.
    
    Args:
        simulation_results: Dictionary containing simulation results
        empirical_data: Dictionary containing empirical data
    Returns:
        float: Mean error value
    """
    error_sum = 0
    for key in simulation_results.keys():
        for step in range(1, 14):
            if step in simulation_results[key]:
                error_sum += abs(simulation_results[key][step] - empirical_data[key][step])
    mean_error = error_sum / 52
    return mean_error

def plot_history_with_empirical(history, empirical_data):
    """
    Plot simulation history with empirical data comparison.
    
    Args:
        history: List of simulation states over time
        empirical_data: Dictionary containing empirical data points
    """
    time_steps = list(range(0, len(history)))
    data = {
        'm_media_R': [step['m_media'].get('R', 0) for step in history],
        'w_media_R': [step['w_media'].get('R', 0) for step in history],
        'o_people_H': [step['o_people'].get('H', 0) for step in history],
        'o_people_M': [step['o_people'].get('M', 0) for step in history],
        'l_people_L': [step['o_people'].get('L', 0) for step in history],
    }
    
    plt.figure(figsize=(10, 6))
    for label, values in data.items():
        plt.plot(time_steps, values, label=label)
    
    # Plot empirical data points
    week_to_day = lambda week: week * 7
    
    for week, risk_p in empirical_data['w_risk_p'].items():
        plt.scatter(week_to_day(week), risk_p, color='orange', 
                   label='Empirical W Risk P' if week == 0 else "")
        
    for week, risk_p in empirical_data['m_risk_p'].items():
        plt.scatter(week_to_day(week), risk_p, color='blue', 
                   label='Empirical M Risk P' if week == 0 else "")
        
    for week, sentiment_p in empirical_data['sentiment_high_p'].items():
        plt.scatter(week_to_day(week), sentiment_p, color='green', 
                   label='Empirical High Sentiment P' if week == 0 else "")
    
    for week, sentiment_p in empirical_data['sentiment_middle_p'].items():
        plt.scatter(week_to_day(week), sentiment_p, color='red', 
                   label='Empirical Middle Sentiment P' if week == 0 else "")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title('State Proportions Over Time with Empirical Data')
    plt.xlabel('Time Step (Days)')
    plt.ylabel('Proportion')
    plt.grid(True)
    plt.show()

from typing import Dict, Tuple

def calculate_error_detailed(simulated: Dict, empirical: Dict, model_type: str) -> Tuple[float, Dict]:
    """
    计算单次模拟结果与经验数据之间的误差，并返回详细的误差分布
    
    Args:
        simulated: 单次模拟结果，格式为 {'w_media': {'R': float}, 'm_media': {'R': float}, ...}
        empirical: 经验数据，格式为 {'w_risk_p': dict, 'm_risk_p': dict, ...}
        model_type: 模型类型 ('no_mainstream', 'no_wemedia', 'homophily_only')
    
    Returns:
        Tuple[float, Dict]: (总体误差, 详细误差字典)
    """
    # 转换模拟结果格式
    result_values = {}
    if 'w_media' in simulated:
        result_values['w_risk_p'] = simulated['w_media']['R']
    if 'm_media' in simulated:
        result_values['m_risk_p'] = simulated['m_media']['R']
    if 'o_people' in simulated:
        result_values['sentiment_high_p'] = simulated['o_people']['H']
        result_values['sentiment_middle_p'] = simulated['o_people']['M']
        result_values['sentiment_low_p'] = simulated['o_people']['L']
    
    # 根据模型类型选择要比较的指标
    if model_type == 'no_mainstream':
        metrics = ['w_risk_p', 'sentiment_high_p', 'sentiment_middle_p', 'sentiment_low_p']
    elif model_type == 'no_wemedia':
        metrics = ['m_risk_p', 'sentiment_high_p', 'sentiment_middle_p', 'sentiment_low_p']
    else:  # homophily_only
        metrics = ['sentiment_high_p', 'sentiment_middle_p', 'sentiment_low_p']
    
    # 存储详细误差信息
    error_details = {
        'total_error': 0,
        'metric_errors': {},
        'weekly_errors': {},
        'raw_values': {
            'simulated': result_values,
            'empirical': empirical
        }
    }
    
    total_error = 0
    n_metrics = len(metrics)
    
    for metric in metrics:
        if metric in result_values and metric in empirical:
            weekly_errors = {}
            squared_errors = []
            
            for week in empirical[metric].keys():
                sim_value = result_values[metric]
                emp_value = empirical[metric][week]
                squared_error = (sim_value - emp_value) ** 2
                weekly_errors[week] = {
                    'simulated': sim_value,
                    'empirical': emp_value,
                    'error': np.sqrt(squared_error),  # RMSE for single point
                    'squared_error': squared_error,
                    'relative_error': abs(sim_value - emp_value) / emp_value if emp_value != 0 else float('inf')
                }
                squared_errors.append(squared_error)
            
            if squared_errors:
                metric_rmse = np.sqrt(np.mean(squared_errors))
                total_error += metric_rmse
                
                error_details['metric_errors'][metric] = {
                    'rmse': metric_rmse,
                    'mean_error': np.mean([w['error'] for w in weekly_errors.values()]),
                    'max_error': max([w['error'] for w in weekly_errors.values()]),
                    'min_error': min([w['error'] for w in weekly_errors.values()]),
                    'std_error': np.std([w['error'] for w in weekly_errors.values()]),
                    'mean_relative_error': np.mean([w['relative_error'] for w in weekly_errors.values() 
                                                  if w['relative_error'] != float('inf')])
                }
                error_details['weekly_errors'][metric] = weekly_errors
    
    # 计算总体误差
    final_error = total_error / n_metrics if n_metrics > 0 else float('inf')
    error_details['total_error'] = final_error
    
    # 添加统计摘要
    error_details['summary'] = {
        'total_metrics': n_metrics,
        'average_rmse': final_error,
        'worst_metric': max(error_details['metric_errors'].items(), 
                          key=lambda x: x[1]['rmse'])[0] if error_details['metric_errors'] else None,
        'best_metric': min(error_details['metric_errors'].items(), 
                         key=lambda x: x[1]['rmse'])[0] if error_details['metric_errors'] else None
    }
    
    return final_error, error_details


from typing import List, Dict

def calculate_weekly_average(history: List[Dict]) -> Dict[str, Dict[str, List[float]]]:
    """
    将每日数据转换为周平均数据
    
    Args:
        history: 模型生成的每日状态历史记录
        
    Returns:
        Dict: 包含各类型节点的周平均状态分布
    """
    # 初始化结果字典
    weekly_data = {
        'm_media': {'R': [], 'NR': []},
        'w_media': {'R': [], 'NR': []},
        'o_people': {'H': [], 'M': [], 'L': []}
    }
    
    # 按周计算平均值
    for week in range(len(history) // 7):
        start_idx = week * 7
        end_idx = start_idx + 7
        week_data = history[start_idx:end_idx]
        
        # 计算这一周的平均值
        for node_type in weekly_data:
            for state in weekly_data[node_type]:
                avg_value = np.mean([day[node_type][state] for day in week_data])
                weekly_data[node_type][state].append(avg_value)
    
    return weekly_data

def calculate_rmse(history: List[Dict], empirical_data: Dict[str, float], 
                  node_type: str, state: str) -> float:
    """
    计算模型数据和经验数据之间的均方根误差(RMSE)
    
    Args:
        history: 模型生成的每日状态历史记录
        empirical_data: 经验数据字典，键为周数，值为对应的数据
        node_type: 节点类型 ('m_media', 'w_media', 'o_people')
        state: 状态类型 ('R', 'NR', 'H', 'M', 'L')
        
    Returns:
        float: RMSE值
    """
    # 计算周平均值
    weekly_avg = calculate_weekly_average(history)
    model_values = np.array(weekly_avg[node_type][state])
    
    # 获取经验数据对应的时间点的值
    empirical_weeks = np.array([int(x) for x in empirical_data.keys()])
    empirical_values = np.array(list(empirical_data.values()))
    
    # 确保长度匹配（取最小长度）
    min_len = min(len(model_values), len(empirical_values))
    model_values = model_values[:min_len]
    empirical_values = empirical_values[:min_len]
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((model_values - empirical_values) ** 2))
    return rmse