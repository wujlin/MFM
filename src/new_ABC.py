import tqdm
import numpy as np
import pandas as pd
from collections import Counter
import math
import joblib
import multiprocessing as mp
from scipy.stats.qmc import LatinHypercube
from tqdm.auto import tqdm
from src.new_modules import *

class ABCSimulator:
    def __init__(self, empirical_data, 
                 network_path, 
                 n_particles=100, 
                 n_iterations=3,
                 n_cores=32):
        """
        初始化ABC模拟器
        
        参数:
        - empirical_data: 经验数据字典
        - n_particles: 每次迭代的粒子数量
        - n_iterations: SMC迭代次数
        """
        self.empirical_data = empirical_data
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.network_path = network_path
        self.n_cores = n_cores
        # 定义参数范围
        self.param_ranges = {
            'alpha': (0.1, 50),    # 根据最优分布调整下限
            'beta': (0.3, 2.0),   # 缩小上限到最优分布附近
            'theta': (0.01, 50),   # 根据最优分布调整下限
            'sigma': (0.01, 50),    # 根据最优分布调整下限
            'zeta': (0.15, 0.9),     # 保持[0,1]范围，略微调整下限
            'miu': (0.01, 1),      # 保持[0,1]范围，略微调整下限
            'g_m': (0.01, 100)      # 保持现有范围
        }
        
        # 记录最佳结果
        self.best_error = float('inf')
        self.best_params = None
        self.best_result = None
        
        # 记录每次迭代的统计信息
        self.iteration_stats = []
        self.parameter_sensitivity = {}

    def analyze_parameter_sensitivity(self, params, param_range=0.1):
        """
        分析参数敏感性
        
        参数:
        - params: 要分析的参数集
        - param_range: 参数变化范围（相对值）
        
        返回:
        - sensitivities: 各参数的敏感性指标
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

    def update_param_ranges(self, accepted_particles, expansion_factor=1.5):
        """
        根据接受的粒子动态调整参数范围
        
        参数:
        - accepted_particles: 被接受的粒子列表
        - expansion_factor: IQR扩展因子
        """
        if not accepted_particles:
            return
        
        new_ranges = {}
        for param_name in self.param_ranges.keys():
            values = [p[param_name] for p in accepted_particles]
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            
            # 根据参数类型设置不同的限制
            if param_name in ['zeta', 'miu']:
                # 概率参数限制在[0,1]
                new_min = max(0, q1 - expansion_factor * iqr)
                new_max = min(1, q3 + expansion_factor * iqr)
            else:
                # 其他参数保持正数
                new_min = max(0.01, q1 - expansion_factor * iqr)
                new_max = q3 + expansion_factor * iqr
            
            new_ranges[param_name] = (new_min, new_max)
        
        self.param_ranges = new_ranges

    def calculate_error(self, simulated, empirical):
        """
        计算模拟结果与经验数据之间的误差
        使用归一化的均方根误差(NRMSE)
        """
        total_error = 0
        n_points = 0
        
        for metric in empirical.keys():
            errors = []
            for week in empirical[metric].keys():
                if week in simulated[metric]:
                    sim_value = simulated[metric][week]
                    emp_value = empirical[metric][week]
                    errors.append((sim_value - emp_value) ** 2)
                    n_points += 1
            
            if errors:
                metric_error = np.sqrt(np.mean(errors))
                total_error += metric_error
        
        return total_error / len(empirical.keys()) if n_points > 0 else float('inf')

    def simulate_network(self, params, num_seeds=5):
        """单次网络模拟"""
        results = []
        for seed in range(num_seeds):
            np.random.seed(seed)
            network = joblib.load(self.network_path)
            setattr(network, 't', 0)
            setattr(network, 'history', [])
            
            history = network.simulate_steps(
                steps=96,
                alpha=params['alpha'],
                beta=params['beta'],
                theta=params['theta'],
                sigma=params['sigma'],
                zeta=params['zeta'],
                miu=params['miu'],
                g_m=params['g_m']
            )
            
            result = {
                'w_risk_p': {},
                'm_risk_p': {},
                'sentiment_high_p': {},
                'sentiment_middle_p': {},
                'sentiment_low_p': {}
            }
            
            for week in range(14):
                day = week * 7
                if day < len(history):
                    state = history[day]
                    result['w_risk_p'][week] = state['w_media']['R']
                    result['m_risk_p'][week] = state['m_media']['R']
                    result['sentiment_high_p'][week] = state['o_people']['H']
                    result['sentiment_middle_p'][week] = state['o_people']['M']
                    result['sentiment_low_p'][week] = state['o_people']['L']
            
            results.append(result)
        
        # 计算平均结果
        avg_result = {
            'w_risk_p': {},
            'm_risk_p': {},
            'sentiment_high_p': {},
            'sentiment_middle_p': {},
            'sentiment_low_p': {}
        }
        
        for week in range(14):
            if all(week in r['w_risk_p'] for r in results):
                for metric in avg_result.keys():
                    avg_result[metric][week] = np.mean([r[metric][week] for r in results])
        
        error = self.calculate_error(avg_result, self.empirical_data)
        
        return {
            'param': params,
            'error': error,
            'result': avg_result
        }

    def simulate_batch(self, params_list):
        """并行处理一批参数组合"""
        with mp.Pool(processes=min(self.n_cores, mp.cpu_count())) as pool:
            results = list(tqdm(
                pool.imap(self.simulate_network, params_list),
                total=len(params_list),
                desc="Simulating batch"
            ))
        return results

    def run_abc(self):
        """修改run_abc方法以包含新功能"""
        all_particles = []
        all_errors = []
        
        for iteration in range(self.n_iterations):
            print(f"\nIteration {iteration + 1}/{self.n_iterations}")
            
            # 生成参数
            if iteration == 0:
                params_list = self.sample_from_prior()
            else:
                # 在生成新粒子前更新参数范围
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
                
                # # 对最佳参数进行敏感性分析
                # self.parameter_sensitivities = self.analyze_parameter_sensitivity(
                #     self.best_params
                # )
            
            # 记录统计信息
            stats = {
                'iteration': iteration,
                'mean_error': np.mean(errors),
                'min_error': min_error,
                'accepted_particles': len(accepted),
                'threshold': threshold,
                'param_ranges': self.param_ranges.copy()
                # 'sensitivities': self.parameter_sensitivities.copy()
            }
            self.iteration_stats.append(stats)
            
            # 保存接受的粒子
            accepted_particles = [r['param'] for r in accepted]
            all_particles.append(accepted_particles)
            all_errors.append([r['error'] for r in accepted])
            
            print(f"Accepted particles: {len(accepted)}")
            print(f"Mean error: {stats['mean_error']:.4f}")
            print(f"Best error: {self.best_error:.4f}")
            
            # # 打印参数敏感性
            # if self.parameter_sensitivities:
            #     print("\nParameter sensitivities:")
            #     for param, sensitivity in self.parameter_sensitivities.items():
            #         print(f"{param}: {sensitivity:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_error': self.best_error,
            'best_result': self.best_result,
            'all_particles': all_particles,
            'all_errors': all_errors,
            'stats': self.iteration_stats
            # 'final_sensitivities': self.parameter_sensitivities
        }

    def sample_from_prior(self):
        """从先验分布采样"""
        sampler = LatinHypercube(d=len(self.param_ranges))
        samples = sampler.random(n=self.n_particles)
        
        params_list = []
        for sample in samples:
            params = {}
            for i, (name, (low, high)) in enumerate(self.param_ranges.items()):
                params[name] = low + (high - low) * sample[i]
            params_list.append(params)
        
        return params_list

    def generate_new_particles(self, previous_particles):
        """基于上一轮的结果生成新的粒子"""
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