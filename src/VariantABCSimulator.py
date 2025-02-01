import numpy as np
import pandas as pd
from collections import Counter
import math
import joblib
import multiprocessing as mp
from scipy.stats.qmc import LatinHypercube
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import os
import platform

from src.other_modules import (
    NoMainstreamModel,
    NoWeMediaModel,
    HomophilyOnlyModel
)

class VariantABCSimulator:
    def __init__(self, empirical_data: Dict, 
                 model_type: str,
                 n_particles: int = 100,
                 n_iterations: int = 3):
        """
        初始化变体ABC模拟器
        
        Args:
            empirical_data: 经验数据字典
            model_type: 模型类型 ('no_mainstream', 'no_wemedia', 'homophily_only')
            n_particles: 每次迭代的粒子数量
            n_iterations: SMC迭代次数
        """
        self.empirical_data = empirical_data
        self.model_type = model_type
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        
        # 根据模型类型设置参数范围
        if model_type == 'no_mainstream':
            self.param_ranges = {
                'beta': (0.3, 2.0),
                'theta': (0.01, 50),
                'sigma': (0.01, 50),
                'zeta': (0.15, 0.9),
                'miu': (0.01, 1),
                'g_m': (0.01, 100)  # 保持g_m用于自适应调控
            }
        elif model_type == 'no_wemedia':
            self.param_ranges = {
                'alpha': (0.1, 50),
                'theta': (0.01, 50),
                'sigma': (0.01, 50),
                'zeta': (0.15, 0.9),
                'miu': (0.01, 1),
                'g_m': (0.01, 100)
            }
        else:  # homophily_only
            self.param_ranges = {
                'sigma': (0.01, 50),
                'zeta': (0.15, 0.9),
                'g_m': (0.01, 100)  # 保持g_m用于自适应调控
            }
        
        # 记录最佳结果
        self.best_error = float('inf')
        self.best_params = None
        self.best_result = None
        
        # 记录每次迭代的统计信息
        self.iteration_stats = []
        self.parameter_sensitivities = {}
    
    
    def calculate_error(self, simulated: Dict, empirical: Dict) -> float:
        """
        计算模拟结果与经验数据之间的误差
        使用归一化的均方根误差(NRMSE)
        
        Args:
            simulated: 多次模拟的结果列表
            empirical: 经验数据，格式为 {'w_risk_p': dict, 'm_risk_p': dict, ...}
        
        Returns:
            float: 归一化的均方根误差
        """
        # 计算多次模拟的平均结果
        avg_result = {}
        if 'w_media' in simulated[0]:
            avg_result['w_risk_p'] = np.mean([s['w_media']['R'] for s in simulated])
        if 'm_media' in simulated[0]:
            avg_result['m_risk_p'] = np.mean([s['m_media']['R'] for s in simulated])
        if 'o_people' in simulated[0]:
            avg_result['sentiment_high_p'] = np.mean([s['o_people']['H'] for s in simulated])
            avg_result['sentiment_middle_p'] = np.mean([s['o_people']['M'] for s in simulated])
            avg_result['sentiment_low_p'] = np.mean([s['o_people']['L'] for s in simulated])
        
        # 根据模型类型选择要比较的指标
        if self.model_type == 'no_mainstream':
            metrics = ['w_risk_p', 'sentiment_high_p', 'sentiment_middle_p', 'sentiment_low_p']
        elif self.model_type == 'no_wemedia':
            metrics = ['m_risk_p', 'sentiment_high_p', 'sentiment_middle_p', 'sentiment_low_p']
        else:  # homophily_only
            metrics = ['sentiment_high_p', 'sentiment_middle_p', 'sentiment_low_p']
        
        # 计算误差
        total_error = 0
        n_metrics = len(metrics)
        
        for metric in metrics:
            if metric in avg_result and metric in empirical:
                errors = []
                for week in empirical[metric].keys():
                    sim_value = avg_result[metric]
                    emp_value = empirical[metric][week]
                    errors.append((sim_value - emp_value) ** 2)
                
                if errors:
                    metric_error = np.sqrt(np.mean(errors))
                    total_error += metric_error
        
        return total_error / n_metrics if n_metrics > 0 else float('inf')

    
    def analyze_parameter_sensitivity(self, params: Dict[str, float], 
                                   param_range: float = 0.1) -> Dict[str, float]:
        """
        分析参数敏感性
        
        Args:
            params: 要分析的参数集
            param_range: 参数变化范围（相对值）
            
        Returns:
            Dict[str, float]: 各参数的敏感性指标
        """
        sensitivities = {}
        base_result = self.simulate_network(params)
        base_error = base_result['error']
        
        for param_name, value in params.items():
            delta = value * param_range
            results = []
            
            # 测试参数增减对结果的影响
            for new_value in [value - delta, value + delta]:
                if (self.param_ranges[param_name][0] <= new_value <= 
                    self.param_ranges[param_name][1]):
                    test_params = params.copy()
                    test_params[param_name] = new_value
                    result = self.simulate_network(test_params)
                    results.append(abs(result['error'] - base_error))
            
            # 计算敏感性指标
            if results:
                sensitivities[param_name] = np.mean(results) / (2 * delta)
        
        return sensitivities
    
    def update_param_ranges(self, accepted_particles: List[Dict], 
                          expansion_factor: float = 1.5) -> None:
        """
        根据接受的粒子动态调整参数范围
        
        Args:
            accepted_particles: 被接受的粒子列表
            expansion_factor: IQR扩展因子
        """
        if not accepted_particles:
            return
        
        new_ranges = {}
        for param_name in self.param_ranges.keys():
            values = [p[param_name] for p in accepted_particles]
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            
            # 根据参数类型设置不同的限制
            if param_name == 'zeta':
                # 概率参数限制在[0,1]
                new_min = max(0, q1 - expansion_factor * iqr)
                new_max = min(1, q3 + expansion_factor * iqr)
            else:
                # 其他参数保持正数
                new_min = max(0.01, q1 - expansion_factor * iqr)
                new_max = q3 + expansion_factor * iqr
            
            new_ranges[param_name] = (new_min, new_max)
        
        self.param_ranges = new_ranges

    def simulate_batch(self, params_list: List[Dict[str, float]]) -> List[Dict]:
        """
        批量执行网络模拟，根据操作系统选择多进程或单进程
        
        Args:
            params_list: 参数列表
            
        Returns:
            List[Dict]: 模拟结果列表
        """
        # 检查操作系统和环境
        system = platform.system()
        use_multiprocessing = (system == 'Linux' or system == 'Darwin')
        
        if use_multiprocessing:
            try:
                # 尝试使用多进程
                with mp.Pool() as pool:
                    results = list(tqdm(
                        pool.imap(self.simulate_network, params_list),
                        total=len(params_list),
                        desc="Simulating networks (MP)"
                    ))
            except Exception as e:
                print(f"Multiprocessing failed: {e}")
                print("Falling back to single process...")
                use_multiprocessing = False
        
        if not use_multiprocessing:
            # 单进程版本
            results = []
            for params in tqdm(params_list, desc="Simulating networks (SP)"):
                result = self.simulate_network(params)
                results.append(result)
        
        return results

    def simulate_network(self, params: Dict[str, float], num_seeds: int = 5) -> Dict:
        """
        单次网络模拟
        
        Args:
            params: 模型参数字典
            num_seeds: 重复模拟次数
            
        Returns:
            Dict: 包含模拟结果和误差的字典
        """
        try:
            simulation_results = []
            for seed in range(num_seeds):
                # 设置随机种子
                if platform.system() == 'Windows':
                    np.random.seed(seed + int(os.getpid() * np.random.random()))
                else:
                    np.random.seed(seed)
                
                network = joblib.load('networks/simple_all_new.pkl')
                
                # 根据模型类型创建对应的变体模型
                if self.model_type == 'no_mainstream':
                    model = NoMainstreamModel(network)
                elif self.model_type == 'no_wemedia':
                    model = NoWeMediaModel(network)
                else:
                    model = HomophilyOnlyModel(network)
                
                # 设置初始状态并执行模拟
                model.t = 0
                model.history = []
                history = model.simulate_steps(steps=96, **params)
                simulation_results.append(history[-1])
            
            # 计算误差
            error = self.calculate_error(simulation_results, self.empirical_data)
            
            return {
                'param': params,
                'error': error,
                'result': simulation_results[-1]  # 保存最后一次模拟的结果
            }
        except Exception as e:
            print(f"Error in simulation: {e}")
            return {
                'param': params,
                'error': float('inf'),
                'result': None
            }
    
    def sample_from_prior(self) -> List[Dict[str, float]]:
        """
        从先验分布采样
        
        Returns:
            List[Dict[str, float]]: 参数样本列表
        """
        sampler = LatinHypercube(d=len(self.param_ranges))
        samples = sampler.random(n=self.n_particles)
        
        params_list = []
        for sample in samples:
            params = {}
            for i, (name, (low, high)) in enumerate(self.param_ranges.items()):
                params[name] = low + (high - low) * sample[i]
            params_list.append(params)
        
        return params_list
    
    def generate_new_particles(self, previous_particles: List[Dict]) -> List[Dict]:
        """
        基于上一轮的结果生成新的粒子
        
        Args:
            previous_particles: 上一轮接受的粒子列表
            
        Returns:
            List[Dict]: 新的参数样本列表
        """
        new_params = []
        for _ in range(self.n_particles):
            # 随机选择一个粒子
            base_particle = previous_particles[np.random.randint(len(previous_particles))]
            # 添加扰动
            new_param = {}
            for name, (low, high) in self.param_ranges.items():
                std = 0.1 * (high - low)
                new_value = base_particle[name] + np.random.normal(0, std)
                new_value = np.clip(new_value, low, high)
                new_param[name] = new_value
            new_params.append(new_param)
        return new_params
    
    def run_abc(self) -> Dict:
        """
        执行ABC优化
        
        Returns:
            Dict: 包含优化结果的字典
        """
        all_particles = []
        all_errors = []
        
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            
            # 生成参数
            if iteration == 0:
                params_list = self.sample_from_prior()
            else:
                self.update_param_ranges(all_particles[-1])
                params_list = self.generate_new_particles(all_particles[-1])
            
            # 运行模拟
            results = self.simulate_batch(params_list)
            
            # 计算自适应阈值
            errors = [r['error'] for r in results]
            threshold = np.percentile(errors, 25)
            
            # 筛选结果
            accepted = [r for r in results if r['error'] <= threshold]
            
            # 更新最佳结果
            min_error = min(errors)
            if min_error < self.best_error:
                best_result = min(results, key=lambda x: x['error'])
                self.best_error = min_error
                self.best_params = best_result['param']
                self.best_result = best_result['result']
                
                # 对最佳参数进行敏感性分析
                self.parameter_sensitivities = self.analyze_parameter_sensitivity(
                    self.best_params
                )
            
            # 记录统计信息
            stats = {
                'iteration': iteration,
                'mean_error': np.mean(errors),
                'min_error': min_error,
                'accepted_particles': len(accepted),
                'threshold': threshold,
                'param_ranges': self.param_ranges.copy(),
                'sensitivities': self.parameter_sensitivities.copy()
            }
            self.iteration_stats.append(stats)
            
            # 保存接受的粒子
            accepted_particles = [r['param'] for r in accepted]
            all_particles.append(accepted_particles)
            all_errors.append([r['error'] for r in accepted])
            
            # 打印当前迭代信息
            print(f"Accepted particles: {len(accepted)}")
            print(f"Mean error: {stats['mean_error']:.4f}")
            print(f"Best error: {self.best_error:.4f}")
            
            # 打印参数敏感性
            if self.parameter_sensitivities:
                print("\nParameter sensitivities:")
                for param, sensitivity in self.parameter_sensitivities.items():
                    print(f"{param}: {sensitivity:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_error': self.best_error,
            'best_result': self.best_result,
            'all_particles': all_particles,
            'all_errors': all_errors,
            'stats': self.iteration_stats,
            'final_sensitivities': self.parameter_sensitivities
        }