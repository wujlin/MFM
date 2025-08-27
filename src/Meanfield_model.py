#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版的简化阈值动力学模型 - V3
主要修复：
1. 正确处理大均值泊松分布的数值问题
2. 动态调整k的范围以包含分布的主要部分
3. 新增幂律分布支持，通过gamma_pref参数控制

网络度分布选项：
- gamma_pref=None: 使用泊松分布（默认）
- gamma_pref=2.5: 使用幂律分布P(k)~k^(-2.5)，对应无标度网络
- use_original_like_dist=True: 使用原始模拟行为的分布（兼容性选项）

参数说明：
- gamma_pref: 幂律指数，必须>1，典型值2.0-3.0
- k_min_pref: 幂律分布下界，默认为1
- max_k: 所有分布类型的上界，用于截断
"""

import numpy as np
from scipy.stats import binom, poisson
import math
from src.config import MAX_ITER, CONVERGENCE_TOL
from scipy.stats import nbinom
import numpy as np


class ThresholdDynamicsModel:
    def __init__(self, network_params, threshold_params, 
                 init_states=None):
        """
        初始化简化的阈值动力学模型
        """
        # 修复可变默认参数陷阱
        if init_states is None:
            init_states = {
                'X_H': 0.3,
                'X_M': 0.4,
                'X_L': 0.3,
                'p_risk_m': 0.5,
                'p_risk_w': 0.5,
                'p_risk': 0.5
            }
        
        # 参数验证
        self._validate_network_params(network_params)
        self._validate_threshold_params(threshold_params)
        
        self.network_params = network_params
        self.threshold_params = threshold_params
        
        # 计算公众入度均值
        public_in_mean = (network_params['k_out_mainstream'] + 
                         network_params['k_out_wemedia'])
        
        # print(f"\n[Minimal Model V3] 初始化参数:")
        # print(f"  网络规模: {network_params['n_mainstream']}个主流媒体, "
        #       f"{network_params['n_wemedia']}个自媒体, "
        #       f"{network_params['n_public']}个公众")
        # print(f"  媒体出度: 主流={network_params['k_out_mainstream']}, "
        #       f"自媒体={network_params['k_out_wemedia']}")
        # print(f"  公众入度均值: {public_in_mean}")
        
    
        
        # 生成公众入度分布 - 根据配置选择分布类型
        gamma_pref = network_params.get('gamma_pref')
        
        if gamma_pref is not None:
            # 优先级1: 使用幂律分布
            k_min = network_params.get('k_min_pref', 1)
            k_max = network_params['max_k']
            
            self.public_in_dist = self._generate_power_law_dist(
                gamma=gamma_pref, k_min=k_min, k_max=k_max
            )
            
            # 计算分布统计信息
            k_values = list(self.public_in_dist.keys())
            mean_degree = sum(k * p for k, p in self.public_in_dist.items())
            max_degree = max(k_values)
            min_degree = min(k_values)
            
            # print(f'使用幂律度分布: γ={gamma_pref}, k∈[{min_degree}, {max_degree}]')
            # print(f'  分布均值: {mean_degree:.2f}, 期望均值: {public_in_mean:.2f}')
            
        elif network_params.get('use_original_like_dist', False):
            # 优先级2: 使用模拟原始模型行为的分布
            public_in_dist_ini = self._generate_poisson_dist_ini(mean=public_in_mean, max_k=network_params['max_k'])
            self.public_in_dist = self.exact_copy_corrected_dist(public_in_dist_ini)
            
            k_values = list(self.public_in_dist.keys())
            mean_degree = sum(k * p for k, p in self.public_in_dist.items())
            
            # print(f'使用模拟原始模型行为的泊松分布: k∈[{min(k_values)}, {max(k_values)}]')
            # print(f'  分布均值: {mean_degree:.2f}, 期望均值: {public_in_mean:.2f}')
            
        else:
            # 优先级3: 使用标准泊松分布
            self.public_in_dist = self._generate_poisson_dist_improved(mean=public_in_mean)
            
            k_values = list(self.public_in_dist.keys())
            mean_degree = sum(k * p for k, p in self.public_in_dist.items())
            
            # print(f'使用标准泊松分布: k∈[{min(k_values)}, {max(k_values)}]')
            # print(f'  分布均值: {mean_degree:.2f}, 期望均值: {public_in_mean:.2f}')
        
        # 分布质量统计
        total_prob = sum(self.public_in_dist.values())
        if abs(total_prob - 1.0) > 1e-6:
            print(f'⚠️  分布归一化偏差: {abs(total_prob - 1.0):.8f}')
        
        # 分布形状统计
        # if len(self.public_in_dist) > 1:
        #     k_list = sorted(self.public_in_dist.keys())
        #     p_list = [self.public_in_dist[k] for k in k_list]
        #     max_prob_k = k_list[p_list.index(max(p_list))]
        #     print(f'  分布峰值: k={max_prob_k}, 支撑点数: {len(k_list)}')
        
        # 保存初始状态
        self.initial_states = init_states.copy()
        
        # 初始化状态
        self.X_H = init_states['X_H']
        self.X_M = init_states['X_M']
        self.X_L = init_states['X_L']
        self.p_risk_m = init_states['p_risk_m']
        self.p_risk_w = init_states['p_risk_w']
        self.p_risk = init_states['p_risk']

    def _validate_network_params(self, network_params):
        """验证网络参数"""
        required_keys = ['n_mainstream', 'n_wemedia', 'n_public', 
                        'k_out_public', 'k_out_mainstream', 'k_out_wemedia', 'max_k']
        
        for key in required_keys:
            if key not in network_params:
                raise KeyError(f"缺少必需的网络参数: {key}")
            
            value = network_params[key]
            
            # 简化验证：直接检查数值是否为正
            if value <= 0:
                raise ValueError(f"网络参数 {key} 必须为正数，当前值: {value}")
            
            # max_k 需要是整数
            if key == 'max_k' and not isinstance(value, int):
                raise ValueError(f"网络参数 max_k 必须为正整数，当前值: {value} (类型: {type(value)})")
        
        # 验证可选的幂律分布参数
        gamma_pref = network_params.get('gamma_pref')
        if gamma_pref is not None:
            if not isinstance(gamma_pref, (int, float)) or gamma_pref <= 0:
                raise ValueError(f"幂律指数 gamma_pref 必须为正数，当前值: {gamma_pref}")
            if gamma_pref <= 1:
                raise ValueError(f"幂律指数 gamma_pref 必须大于1以确保分布可归一化，当前值: {gamma_pref}")
        
        # 验证可选的幂律分布下界参数
        k_min_pref = network_params.get('k_min_pref', 1)
        if not isinstance(k_min_pref, int) or k_min_pref < 1:
            raise ValueError(f"幂律分布下界 k_min_pref 必须为正整数，当前值: {k_min_pref}")
        
        if k_min_pref >= network_params['max_k']:
            raise ValueError(f"幂律分布下界 k_min_pref ({k_min_pref}) 必须小于上界 max_k ({network_params['max_k']})")

    def _validate_threshold_params(self, threshold_params):
        """验证阈值参数"""
        required_keys = ['theta', 'phi']
        
        for key in required_keys:
            if key not in threshold_params:
                raise KeyError(f"缺少必需的阈值参数: {key}")
            
            value = threshold_params[key]
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValueError(f"阈值参数 {key} 应该在 [0,1] 范围内，当前值: {value}")
        
        if threshold_params['theta'] <= threshold_params['phi']:
            raise ValueError(f"高唤醒阈值 theta ({threshold_params['theta']}) "
                           f"应该大于低唤醒阈值 phi ({threshold_params['phi']})")


    def _generate_poisson_dist_ini(self, mean, max_k=50):
        """生成截断泊松分布"""
        
        k_values = np.arange(1, max_k + 1)
        p_k = np.exp(-mean) * np.power(mean, k_values) / np.array([math.factorial(k) for k in k_values])
        
        # 检查是否全为零
        if np.sum(p_k) == 0:
            return {1: 1.0}
            
        p_k = p_k / np.sum(p_k)  # 归一化
        return {k: p for k, p in zip(k_values, p_k)}

    def exact_copy_corrected_dist(self, original_dist):
        """
        方案A: 精确拷贝原始分布并进行最小修正
        直接使用原始模型的分布数据，只修正数学上不合理的部分
        """
        corrected_dist = {}
        
        # print(f"  原始分布异常值检查:")
        anomaly_count = 0
        
        # 第1步：拷贝原始分布并检查异常
        for k, p in original_dist.items():
            if p > 1.0:
                # print(f"    k={k}: P={p:.3f} > 1.0 (异常)")
                corrected_dist[k] = 0.95  # 截断过大的概率
                anomaly_count += 1
            elif p < 0.0:
                # print(f"    k={k}: P={p:.3f} < 0.0 (异常)")
                corrected_dist[k] = 0.001  # 修正负概率
                anomaly_count += 1
            else:
                corrected_dist[k] = p
        
        # print(f"    发现 {anomaly_count} 个异常概率值")
        
        # 第2步：重新归一化
        total_prob = sum(corrected_dist.values())
        if total_prob > 0:
            for k in corrected_dist:
                corrected_dist[k] /= total_prob
        
        # 验证修正后的特征
        corrected_mean = sum(k * p for k, p in corrected_dist.items())
        # print(f"    修正后均值: {corrected_mean:.3f}")
        
        return corrected_dist

        
    def _generate_poisson_dist_improved(self, mean):
        """
        生成改进的泊松分布，正确处理大均值情况
        """
        if mean <= 0:
            return {1: 1.0}
        
        # 动态确定k的范围，确保包含分布的主要部分
        # 使用3-sigma规则：包含99.7%的概率质量
        std_dev = np.sqrt(mean)
        k_min = max(1, int(mean - 3 * std_dev))
        k_max = int(mean + 3 * std_dev)
        
        # 使用scipy的泊松分布，它能正确处理大均值
        k_values = np.arange(k_min, k_max + 1)
        p_k = poisson.pmf(k_values, mean)
        
        # 归一化
        p_k = p_k / np.sum(p_k)
        
        # 转换为字典
        dist_dict = {k: p for k, p in zip(k_values, p_k) if p > 1e-10}
        
        # 验证
        total_prob = sum(dist_dict.values())
        if abs(total_prob - 1.0) > 1e-6:
            # 重新归一化
            factor = 1.0 / total_prob
            dist_dict = {k: p * factor for k, p in dist_dict.items()}
        
        return dist_dict

    def _generate_biased_poisson_dist(self, mean, max_k=50):
        """生成有偏的泊松分布，但避免负概率"""
        if mean <= 0:
            return {1: 1.0}
        
        # 固定范围[1,max_k]，类似V2
        k_values = np.arange(1, max_k + 1)
        
        # 使用scipy的泊松分布避免数值问题
        p_k = poisson.pmf(k_values, mean)
        
        # 归一化
        p_k = p_k / np.sum(p_k)
        
        # 转换为字典，过滤极小值
        return {k: p for k, p in zip(k_values, p_k) if p > 1e-10}

    def _generate_power_law_dist(self, gamma, k_min=1, k_max=200):
        """
        生成截断幂律分布 P(k) ~ k^(-γ)
        
        参数:
        - gamma: 幂律指数，必须 > 1
        - k_min: 分布下界，默认为1
        - k_max: 分布上界，用于截断
        
        返回:
        - 度分布字典 {k: p_k}
        """
        if gamma <= 0:
            raise ValueError("幂律指数 gamma 必须为正数")
        if gamma <= 1:
            raise ValueError("幂律指数 gamma 必须大于1以确保可归一化")
        if k_min < 1:
            raise ValueError("度分布下界 k_min 必须 >= 1")
        if k_max <= k_min:
            raise ValueError("上界 k_max 必须大于下界 k_min")
        
        # 度值范围
        k_values = np.arange(k_min, k_max + 1)
        
        # 计算未归一化的幂律概率密度
        raw_probs = k_values.astype(float) ** (-gamma)
        
        # 归一化常数（分母）
        Z = np.sum(raw_probs)
        
        # 归一化概率分布
        p_k = raw_probs / Z
        
        # 转换为字典格式，过滤极小概率值
        dist_dict = {k: float(p) for k, p in zip(k_values, p_k) if p > 1e-10}
        
        # 二次验证：确保概率和为1
        total_prob = sum(dist_dict.values())
        if abs(total_prob - 1.0) > 1e-6:
            # 如果存在数值误差，重新归一化
            factor = 1.0 / total_prob
            dist_dict = {k: p * factor for k, p in dist_dict.items()}
        
        return dist_dict


    
    def generate_nbinom_dist(self, mean, dispersion=0.5):
        """生成无截断的负二项分布"""
        # 参数转换
        r = 1.0 / dispersion
        p = r / (r + mean)
        
        # 确定合适的范围（基于分布特性自动确定）
        k_max = int(nbinom.ppf(0.9999, r, p))  # 覆盖99.99%的概率质量
        k_values = np.arange(1, k_max + 1)
        
        # 计算PMF
        p_k = nbinom.pmf(k_values, r, p)
        
        # 条件化（确保k>=1）并归一化
        p_k = p_k / np.sum(p_k)
        
        # 过滤极小概率
        return {k: p for k, p in zip(k_values, p_k) if p > 1e-10}


    
    def _calc_public_emotions(self, p_risk):
        """计算公众情绪状态分布"""
        theta = self.threshold_params['theta']
        phi = self.threshold_params['phi']
        
        # 边界处理
        epsilon = 1e-10
        p_risk = max(min(p_risk, 1.0 - epsilon), epsilon)
        
        X_H = 0.0
        X_L = 0.0
        
        for k_in, p_k in self.public_in_dist.items():
            # 高唤醒：收到的风险信号数 >= theta * k_in
            min_risk = math.ceil(theta * k_in)
            for s in range(min_risk, k_in + 1):
                X_H += p_k * binom.pmf(s, k_in, p_risk)
            
            # 低唤醒：收到的风险信号数 <= phi * k_in
            max_risk = math.floor(phi * k_in)
            for s in range(0, max_risk + 1):
                X_L += p_k * binom.pmf(s, k_in, p_risk)
        
        X_M = 1.0 - X_H - X_L
        return X_H, X_M, X_L

    def _calc_mainstream_risk_simplified(self, X_H, X_L):
        """
        计算主流媒体风险 - 归一化线性版本
        p_risk_m = (1 - X_H + X_L) / 2
        
        逻辑：
        - X_H 增加 → 风险减少 (负反馈抑制高唤醒)
        - X_L 增加 → 风险增加 (激活低唤醒状态)  
        - 归一化确保结果在[0,1]范围内
        - 完全线性，保持左右对称性
        """
        return (1.0 - X_H + X_L) / 2.0

    def _calc_wemedia_risk_simplified(self, X_H):
        """计算自媒体风险"""
        return X_H
    
    def _calc_overall_risk(self, p_risk_m, p_risk_w, removal_ratio_m=0.0, removal_ratio_w=0.0):
        """计算综合媒体风险比例"""
        n_m = self.network_params['n_mainstream']
        n_w = self.network_params['n_wemedia']
        
        n_m_eff = n_m * (1 - removal_ratio_m)
        n_w_eff = n_w * (1 - removal_ratio_w)
        
        denominator = n_m_eff + n_w_eff
        
        if denominator <= 0:
            return 0.5  # 所有媒体被移除，返回中性值
            
        return (p_risk_m * n_m_eff + p_risk_w * n_w_eff) / denominator
    
    def solve_self_consistent(self, init_states=None, removal_ratios=None, 
                            max_iter=None, tol=None):
        """
        求解自洽方程
        """
        # 默认值
        if removal_ratios is None:
            removal_ratios = {}
        if max_iter is None:
            max_iter = MAX_ITER
        if tol is None:
            tol = CONVERGENCE_TOL
        
        # 如果没有提供init_states，则使用构造函数中保存的初始值
        if init_states is None:
            self.X_H = self.initial_states['X_H']
            self.X_M = self.initial_states['X_M']
            self.X_L = self.initial_states['X_L']
            self.p_risk_m = self.initial_states['p_risk_m']
            self.p_risk_w = self.initial_states['p_risk_w']
            self.p_risk = self.initial_states['p_risk']
        else:
            # 使用提供的初始值
            self.X_H = init_states.get('X_H', self.initial_states['X_H'])
            self.X_M = init_states.get('X_M', self.initial_states['X_M'])
            self.X_L = init_states.get('X_L', self.initial_states['X_L'])
            self.p_risk_m = init_states.get('p_risk_m', self.initial_states['p_risk_m'])
            self.p_risk_w = init_states.get('p_risk_w', self.initial_states['p_risk_w'])
            self.p_risk = init_states.get('p_risk', self.initial_states['p_risk'])
        
        # 提取移除比例
        r_high = removal_ratios.get('high', 0.0)
        r_mid = removal_ratios.get('mid', 0.0)
        r_low = removal_ratios.get('low', 0.0)
        r_mainstream = removal_ratios.get('mainstream', 0.0)
        r_wemedia = removal_ratios.get('wemedia', 0.0)
        
        # 历史记录
        history = {
            'X_H': [self.X_H],
            'X_M': [self.X_M],
            'X_L': [self.X_L],
            'p_risk_m': [self.p_risk_m],
            'p_risk_w': [self.p_risk_w],
            'p_risk': [self.p_risk]
        }
        
        # 迭代求解
        converged = False
        for i in range(max_iter):
            # 保存旧值
            old_X_H = self.X_H
            old_X_M = self.X_M
            old_X_L = self.X_L
            old_p_risk_m = self.p_risk_m
            old_p_risk_w = self.p_risk_w
            old_p_risk = self.p_risk
            
            # 1. 计算媒体风险
            self.p_risk_m = self._calc_mainstream_risk_simplified(self.X_H, self.X_L)
            self.p_risk_w = self._calc_wemedia_risk_simplified(self.X_H)
            
            # 2. 计算总体风险
            self.p_risk = self._calc_overall_risk(
                self.p_risk_m, self.p_risk_w, r_mainstream, r_wemedia
            )
            
            # 3. 计算公众情绪
            X_H_new, X_M_new, X_L_new = self._calc_public_emotions(self.p_risk)
            
            # 4. 应用情绪节点移除并重新归一化
            X_H_removed = X_H_new * (1 - r_high)
            X_M_removed = X_M_new * (1 - r_mid)
            X_L_removed = X_L_new * (1 - r_low)
            
            total = X_H_removed + X_M_removed + X_L_removed
            if total > 0:
                self.X_H = X_H_removed / total
                self.X_M = X_M_removed / total
                self.X_L = X_L_removed / total
            else:
                # 极端情况：保持不变
                pass
            
            # 记录历史
            history['X_H'].append(self.X_H)
            history['X_M'].append(self.X_M)
            history['X_L'].append(self.X_L)
            history['p_risk_m'].append(self.p_risk_m)
            history['p_risk_w'].append(self.p_risk_w)
            history['p_risk'].append(self.p_risk)
            
            # 检查收敛
            if (abs(self.X_H - old_X_H) < tol and
                abs(self.X_M - old_X_M) < tol and
                abs(self.X_L - old_X_L) < tol and
                abs(self.p_risk_m - old_p_risk_m) < tol and
                abs(self.p_risk_w - old_p_risk_w) < tol and
                abs(self.p_risk - old_p_risk) < tol):
                converged = True
                break
        
        return {
            'X_H': self.X_H,
            'X_M': self.X_M,
            'X_L': self.X_L,
            'p_risk_m': self.p_risk_m,
            'p_risk_w': self.p_risk_w,
            'p_risk': self.p_risk,
            'converged': converged,
            'iterations': i + 1,
            'removal_ratios': removal_ratios,
            'history': history
        } 