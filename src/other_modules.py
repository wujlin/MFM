import math
from collections import Counter
from scipy.stats import entropy
import numpy as np

# Standard library imports
from collections import Counter
import math
from typing import List, Dict, Optional, Union, Any, Tuple

# Third-party library imports
import numpy as np
from scipy.stats import entropy
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import entropy
from collections import Counter

from src.modules import detect_change_point, adaptive_regulatory_field, sigmoid_scale



class NoMainstreamNode:
    """完全重新定义的无主流媒体节点"""
    def __init__(self, name, node_type, sentiment=None, risk=None, intensity=None):
        self.name = name
        self.type = node_type
        self.sentiment = sentiment
        self.risk = risk
        self.intensity = intensity
        self.influencers = []
        
    def add_influencer(self, influencer):
        self.influencers.append(influencer)
        
    def update(self, params):
        """自定义的状态更新机制"""
        if self.type == 'o_people':
            self._update_people(params)
        elif self.type == 'w_media':
            self._update_wemedia(params)
            
    def _update_people(self, params):
        """仅考虑自媒体影响的用户情绪更新"""
        intensity = self.intensity
        risk_w = params['risk_w']
        norisk_w = params['norisk_w']
        theta = params['theta']
        sigma = params['sigma']
        zeta = params['zeta']
        
        sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
        for sentiment in ['H', 'M', 'L']:
            if sentiment not in sentiment_counter:
                sentiment_counter[sentiment] = 0
                
        # 只考虑自媒体的风险信息
        if self.sentiment == 'H':
            d1 = norisk_w - risk_w
            d2 = sentiment_counter.get('M', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('H', 0)
        elif self.sentiment == 'M':
            d1 = abs(norisk_w - risk_w)
            d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('M', 0)
        elif self.sentiment == 'L':
            d1 = risk_w - norisk_w
            d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('M', 0) - sentiment_counter.get('L', 0)
        
        intensity = intensity - ((np.tanh(theta * d1) + np.tanh(sigma * d2)) + 2) / 4
        d = risk_w - norisk_w
        Pw = (1 + np.tanh(d)) / 2
        
        if intensity > 1:
            intensity = 1
        elif intensity < 0:
            intensity = 0
            
        # 状态转换逻辑
        if intensity > np.random.random():
            if self.sentiment == 'M':
                if np.random.random() < zeta:
                    self.sentiment = 'H' if Pw > np.random.random() else 'L'
                else:
                    self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
            elif self.sentiment in ['H', 'L']:
                if np.random.random() < zeta:
                    self.sentiment = 'M'
                else:
                    self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
                    
        self.intensity = intensity
        
    def _update_wemedia(self, params):
        """自媒体的风险状态更新"""
        if np.random.random() < params['beta']:
            sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
            n_high = sentiment_counter.get('H', 0)
            n_low = sentiment_counter.get('L', 0)
            n_middle = sentiment_counter.get('M', 0)
            
            if n_middle > max(n_high, n_low):
                self.risk = 'NR'
            elif n_high > max(n_middle, n_low):
                d = 2 * n_high - n_middle - n_low
                Pu = 1 - math.exp(-params['miu'] * d)
                self.risk = 'NR' if Pu > np.random.random() else self.risk
            elif n_low > max(n_middle, n_high):
                d = 2 * n_low - n_middle - n_high
                Pu = 1 - math.exp(-params['miu'] * d)
                self.risk = 'R' if Pu > np.random.random() else self.risk

class NoMainstreamModel:
    """完全重新定义的无主流媒体模型"""
    def __init__(self, original_network):
        """
        基于原始网络构建无主流媒体模型
        
        Args:
            original_network: 原始CSDAG网络实例
        """
        self.nodes = {}
        self.t = 0
        self.history = []
        
        # 复制除主流媒体外的所有节点
        for name, node in original_network.nodes.items():
            if node.type == 'm_media':
                continue
            
            if node.type == 'o_people':
                new_node = NoMainstreamNode(
                    name=name,
                    node_type=node.type,
                    sentiment=node.sentiment,
                    intensity=node.intensity
                )
            elif node.type == 'w_media':
                new_node = NoMainstreamNode(
                    name=name,
                    node_type=node.type,
                    risk=node.risk
                )
            self.nodes[name] = new_node
            
        # 复制连接关系，排除主流媒体节点
        for name, node in original_network.nodes.items():
            if name in self.nodes:  # 确保节点存在
                for influencer in node.influencers:
                    if influencer.type != 'm_media':
                        self.nodes[name].add_influencer(self.nodes[influencer.name])
    
    def simulate_step(self, alpha, beta, theta, sigma, zeta, miu, g_m):
        """执行一步模拟"""
        self.t += 1
        
        # 计算当前风险分布
        risk_counts = {'w_media': Counter()}
        for node in self.nodes.values():
            if node.type == 'w_media':
                risk_counts['w_media'][node.risk] += 1
                
        total_w = sum(risk_counts['w_media'].values())
        risk_w = risk_counts['w_media'].get('R', 0) / total_w if total_w > 0 else 0
        norisk_w = risk_counts['w_media'].get('NR', 0) / total_w if total_w > 0 else 0
        
        # 构建更新参数
        params = {
            'beta': beta,
            'theta': theta,
            'sigma': sigma,
            'zeta': zeta,
            'miu': miu,
            'risk_w': risk_w,
            'norisk_w': norisk_w
        }
        
        # 随机更新顺序
        update_order = list(self.nodes.keys())
        np.random.shuffle(update_order)
        
        # 更新所有节点
        for name in update_order:
            self.nodes[name].update(params)
            
        # 记录状态分布
        current_distribution = self.calculate_state_proportions()
        self.history.append(current_distribution)
    
    def simulate_steps(self, steps: int = 91,
                      alpha: float = 0.5,
                      beta: float = 0.3,
                      theta: float = 0.4,
                      sigma: float = 0.2,
                      zeta: float = 0.1,
                      miu: float = 0.05,
                      g_m: float = 10.0) -> List[Dict]:
        """执行多步模拟"""
        ini_distribution = self.calculate_state_proportions()
        self.history = [ini_distribution]
        self.t = 0
        
        for _ in range(steps):
            self.simulate_step(
                alpha=alpha,
                beta=beta,
                theta=theta,
                sigma=sigma,
                zeta=zeta,
                miu=miu,
                g_m=g_m
            )
            
        return self.history
    
    def calculate_state_proportions(self) -> Dict[str, Dict[str, float]]:
        """计算各类节点的状态分布"""
        states = {
            'm_media': Counter({'R': 0, 'NR': 0}),  # 保持接口一致性
            'w_media': Counter({'R': 0, 'NR': 0}),
            'o_people': Counter({'H': 0, 'M': 0, 'L': 0})
        }
        
        # 统计各类节点的状态
        for node in self.nodes.values():
            if node.type == 'w_media':
                states['w_media'][node.risk] += 1
            elif node.type == 'o_people':
                states['o_people'][node.sentiment] += 1
                
        # 计算比例
        proportions = {}
        for type_ in states:
            total = sum(states[type_].values())
            if total > 0:
                proportions[type_] = {
                    state: count / total 
                    for state, count in states[type_].items()
                }
            else:
                proportions[type_] = {
                    state: 0 for state in states[type_]
                }
                
        return proportions
    
    
class NoWeMediaNode:
    """完全重新定义的无自媒体节点"""
    def __init__(self, name, node_type, sentiment=None, risk=None, intensity=None):
        self.name = name
        self.type = node_type
        self.sentiment = sentiment
        self.risk = risk
        self.intensity = intensity
        self.influencers = []
        
    def add_influencer(self, influencer):
        self.influencers.append(influencer)
        
    def update(self, params):
        """自定义的状态更新机制"""
        if self.type == 'o_people':
            self._update_people(params)
        elif self.type == 'm_media':
            self._update_mainstream(params)
            
    def _update_people(self, params):
        """仅考虑主流媒体影响的用户情绪更新"""
        intensity = self.intensity
        risk_m = params['risk_m']
        norisk_m = params['norisk_m']
        theta = params['theta']
        sigma = params['sigma']
        zeta = params['zeta']
        
        sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
        for sentiment in ['H', 'M', 'L']:
            if sentiment not in sentiment_counter:
                sentiment_counter[sentiment] = 0
                
        # 只考虑主流媒体的风险信息
        if self.sentiment == 'H':
            d1 = norisk_m - risk_m
            d2 = sentiment_counter.get('M', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('H', 0)
        elif self.sentiment == 'M':
            d1 = abs(norisk_m - risk_m)
            d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('M', 0)
        elif self.sentiment == 'L':
            d1 = risk_m - norisk_m
            d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('M', 0) - sentiment_counter.get('L', 0)
        
        intensity = intensity - ((np.tanh(theta * d1) + np.tanh(sigma * d2)) + 2) / 4
        d = risk_m - norisk_m
        Pw = (1 + np.tanh(d)) / 2
        
        if intensity > 1:
            intensity = 1
        elif intensity < 0:
            intensity = 0
            
        # 状态转换逻辑
        if intensity > np.random.random():
            if self.sentiment == 'M':
                if np.random.random() < zeta:
                    self.sentiment = 'H' if Pw > np.random.random() else 'L'
                else:
                    self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
            elif self.sentiment in ['H', 'L']:
                if np.random.random() < zeta:
                    self.sentiment = 'M'
                else:
                    self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
                    
        self.intensity = intensity
        
    def _update_mainstream(self, params):
        """主流媒体的风险状态更新"""
        if np.random.random() < params['alpha']:
            sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
            n_high = sentiment_counter.get('H', 0)
            n_low = sentiment_counter.get('L', 0)
            n_middle = sentiment_counter.get('M', 0)
            
            if n_middle > max(n_high, n_low):
                self.risk = 'NR'
            elif n_high > max(n_middle, n_low):
                d = 2 * n_high - n_middle - n_low
                g_m_scaled = sigmoid_scale(params['g_m'])
                Pu = 1 - math.exp(-g_m_scaled * d)
                self.risk = 'NR' if Pu > np.random.random() else self.risk
            elif n_low > max(n_middle, n_high):
                d = 2 * n_low - n_middle - n_high
                g_m_scaled = sigmoid_scale(params['g_m'])
                Pu = 1 - math.exp(-g_m_scaled * d)
                self.risk = 'R' if Pu > np.random.random() else self.risk

class NoWeMediaModel:
    """完全重新定义的无自媒体模型"""
    def __init__(self, original_network):
        self.nodes = {}
        self.t = 0
        self.history = []
        
        # 复制除自媒体外的所有节点
        for name, node in original_network.nodes.items():
            if node.type == 'w_media':
                continue
            
            if node.type == 'o_people':
                new_node = NoWeMediaNode(
                    name=name,
                    node_type=node.type,
                    sentiment=node.sentiment,
                    intensity=node.intensity
                )
            elif node.type == 'm_media':
                new_node = NoWeMediaNode(
                    name=name,
                    node_type=node.type,
                    risk=node.risk
                )
            self.nodes[name] = new_node
            
        # 复制连接关系，排除自媒体节点
        for name, node in original_network.nodes.items():
            if name in self.nodes:
                for influencer in node.influencers:
                    if influencer.type != 'w_media':
                        self.nodes[name].add_influencer(self.nodes[influencer.name])

    def simulate_step(self, alpha, beta, theta, sigma, zeta, miu, g_m_base):
        """执行一步模拟"""
        self.t += 1
        
        # 计算调控场强度
        change_rate = detect_change_point(self.history)
        g_m = adaptive_regulatory_field(g_m_base, change_rate, self.history)
        
        # 计算当前风险分布
        risk_counts = {'m_media': Counter()}
        for node in self.nodes.values():
            if node.type == 'm_media':
                risk_counts['m_media'][node.risk] += 1
                
        total_m = sum(risk_counts['m_media'].values())
        risk_m = risk_counts['m_media'].get('R', 0) / total_m if total_m > 0 else 0
        norisk_m = risk_counts['m_media'].get('NR', 0) / total_m if total_m > 0 else 0
        
        # 构建更新参数
        params = {
            'alpha': alpha,
            'theta': theta,
            'sigma': sigma,
            'zeta': zeta,
            'g_m': g_m,
            'risk_m': risk_m,
            'norisk_m': norisk_m
        }
        
        # 随机更新顺序
        update_order = list(self.nodes.keys())
        np.random.shuffle(update_order)
        
        # 更新所有节点
        for name in update_order:
            self.nodes[name].update(params)
            
        # 记录状态分布
        current_distribution = self.calculate_state_proportions()
        self.history.append(current_distribution)
    
    def simulate_steps(self, steps: int = 91,
                      alpha: float = 0.5,
                      beta: float = 0.3,
                      theta: float = 0.4,
                      sigma: float = 0.2,
                      zeta: float = 0.1,
                      miu: float = 0.05,
                      g_m: float = 10.0) -> List[Dict]:
        """执行多步模拟"""
        ini_distribution = self.calculate_state_proportions()
        self.history = [ini_distribution]
        self.t = 0
        
        for _ in range(steps):
            self.simulate_step(
                alpha=alpha,
                beta=beta,
                theta=theta,
                sigma=sigma,
                zeta=zeta,
                miu=miu,
                g_m_base=g_m
            )
            
        return self.history
    
    def calculate_state_proportions(self) -> Dict[str, Dict[str, float]]:
        """计算各类节点的状态分布"""
        states = {
            'm_media': Counter({'R': 0, 'NR': 0}),
            'w_media': Counter({'R': 0, 'NR': 0}),  # 保持接口一致性
            'o_people': Counter({'H': 0, 'M': 0, 'L': 0})
        }
        
        for node in self.nodes.values():
            if node.type == 'm_media':
                states['m_media'][node.risk] += 1
            elif node.type == 'o_people':
                states['o_people'][node.sentiment] += 1
                
        proportions = {}
        for type_ in states:
            total = sum(states[type_].values())
            if total > 0:
                proportions[type_] = {
                    state: count / total 
                    for state, count in states[type_].items()
                }
            else:
                proportions[type_] = {
                    state: 0 for state in states[type_]
                }
                
        return proportions

class HomophilyOnlyNode:
    """完全重新定义的仅同质性影响节点"""
    def __init__(self, name, node_type, sentiment=None, intensity=None):
        self.name = name
        self.type = node_type
        self.sentiment = sentiment
        self.intensity = intensity
        self.influencers = []
        
    def add_influencer(self, influencer):
        self.influencers.append(influencer)
        
    def update(self, params):
        """仅考虑同质性影响的状态更新"""
        if self.type != 'o_people':
            return
            
        intensity = self.intensity
        sigma = params['sigma']
        zeta = params['zeta']
        
        # 统计邻居情绪状态
        sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
        for sentiment in ['H', 'M', 'L']:
            if sentiment not in sentiment_counter:
                sentiment_counter[sentiment] = 0
                
        # 计算同质性影响
        if self.sentiment == 'H':
            d2 = sentiment_counter.get('M', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('H', 0)
        elif self.sentiment == 'M':
            d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('M', 0)
        elif self.sentiment == 'L':
            d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('M', 0) - sentiment_counter.get('L', 0)

        # 更新intensity，只考虑同质性影响
        intensity = intensity - (np.tanh(sigma * d2) + 1) / 2
        
        if intensity > 1:
            intensity = 1
        elif intensity < 0:
            intensity = 0
            
        # 状态转换逻辑
        if intensity > np.random.random():
            if self.sentiment == 'M':
                if np.random.random() < zeta:
                    # 基于邻居节点的主导情绪状态
                    if sentiment_counter:
                        self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
                else:
                    if sentiment_counter:
                        self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
            elif self.sentiment in ['H', 'L']:
                if np.random.random() < zeta:
                    self.sentiment = 'M'
                else:
                    if sentiment_counter:
                        self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
                        
        self.intensity = intensity

class HomophilyOnlyModel:
    """完全重新定义的仅同质性影响模型"""
    def __init__(self, original_network):
        """
        基于原始网络构建仅同质性影响模型
        
        Args:
            original_network: 原始CSDAG网络实例
        """
        self.nodes = {}
        self.t = 0
        self.history = []
        
        # 只复制普通用户节点
        for name, node in original_network.nodes.items():
            if node.type == 'o_people':
                new_node = HomophilyOnlyNode(
                    name=name,
                    node_type=node.type,
                    sentiment=node.sentiment,
                    intensity=node.intensity
                )
                self.nodes[name] = new_node
            
        # 只复制用户节点之间的连接关系
        for name, node in original_network.nodes.items():
            if name in self.nodes:
                for influencer in node.influencers:
                    if influencer.type == 'o_people':
                        self.nodes[name].add_influencer(self.nodes[influencer.name])
    
    def simulate_step(self, alpha, beta, theta, sigma, zeta, miu, g_m):
        """执行一步模拟"""
        self.t += 1
        
        # 构建更新参数，只保留必要的参数
        params = {
            'sigma': sigma,  # 同质性影响系数
            'zeta': zeta    # 记忆衰减系数
        }
        
        # 随机更新顺序
        update_order = list(self.nodes.keys())
        np.random.shuffle(update_order)
        
        # 更新所有节点
        for name in update_order:
            self.nodes[name].update(params)
            
        # 记录状态分布
        current_distribution = self.calculate_state_proportions()
        self.history.append(current_distribution)
    
    def simulate_steps(self, steps: int = 91,
                      alpha: float = 0.5,
                      beta: float = 0.3,
                      theta: float = 0.4,
                      sigma: float = 0.2,
                      zeta: float = 0.1,
                      miu: float = 0.05,
                      g_m: float = 10.0) -> List[Dict]:
        """执行多步模拟"""
        ini_distribution = self.calculate_state_proportions()
        self.history = [ini_distribution]
        self.t = 0
        
        for _ in range(steps):
            self.simulate_step(
                alpha=alpha,
                beta=beta,
                theta=theta,
                sigma=sigma,
                zeta=zeta,
                miu=miu,
                g_m=g_m
            )
            
        return self.history
    
    def calculate_state_proportions(self) -> Dict[str, Dict[str, float]]:
        """计算各类节点的状态分布"""
        states = {
            'm_media': Counter({'R': 0, 'NR': 0}),  # 保持接口一致性
            'w_media': Counter({'R': 0, 'NR': 0}),  # 保持接口一致性
            'o_people': Counter({'H': 0, 'M': 0, 'L': 0})
        }
        
        # 只统计用户节点的情绪状态
        for node in self.nodes.values():
            states['o_people'][node.sentiment] += 1
                
        # 计算比例
        proportions = {
            'm_media': {'R': 0, 'NR': 0},  # 媒体节点的比例始终为0
            'w_media': {'R': 0, 'NR': 0},
            'o_people': {}
        }
        
        total_people = sum(states['o_people'].values())
        if total_people > 0:
            proportions['o_people'] = {
                state: count / total_people 
                for state, count in states['o_people'].items()
            }
        else:
            proportions['o_people'] = {
                state: 0 for state in ['H', 'M', 'L']
            }
                
        return proportions
    


def calculate_weekly_metrics(history: List[Dict]) -> Dict[str, Dict[int, float]]:
    """
    从模拟历史中计算每周的指标值
    
    Args:
        history: 模拟历史数据，格式为 List[Dict]，每个 Dict 包含各子系统的状态分布
        
    Returns:
        Dict[str, Dict[int, float]]: 每周的指标值，格式为 {指标: {周数: 值}}
    """
    # 初始化结果字典
    result = {
        'w_risk_p': {},
        'm_risk_p': {},
        'sentiment_high_p': {},
        'sentiment_middle_p': {},
        'sentiment_low_p': {}
    }
    
    # 计算每周的指标值
    for week in range(14):
        day = week * 7
        if day < len(history):
            state = history[day]
            result['w_risk_p'][week] = state['w_media']['R']
            result['m_risk_p'][week] = state['m_media']['R']
            result['sentiment_high_p'][week] = state['o_people']['H']
            result['sentiment_middle_p'][week] = state['o_people']['M']
            result['sentiment_low_p'][week] = state['o_people']['L']
    
    return result

def calculate_average_metrics(results: List[Dict[str, Dict[int, float]]]) -> Dict[str, Dict[int, float]]:
    """
    计算多个结果的均值
    
    Args:
        results: 多个模拟结果，格式为 List[Dict[str, Dict[int, float]]]
        
    Returns:
        Dict[str, Dict[int, float]]: 平均后的指标值
    """
    # 初始化平均结果字典
    avg_result = {
        'w_risk_p': {},
        'm_risk_p': {},
        'sentiment_high_p': {},
        'sentiment_middle_p': {},
        'sentiment_low_p': {}
    }
    
    # 计算每周的平均值
    for week in range(14):
        if all(week in r['w_risk_p'] for r in results):
            for metric in avg_result.keys():
                avg_result[metric][week] = np.mean([r[metric][week] for r in results])
    
    return avg_result

def process_history(history: List[Dict]) -> Dict[str, Dict[int, float]]:
    """
    处理模拟历史数据，返回每周的指标值
    
    Args:
        history: 模拟历史数据，格式为 List[Dict]
        
    Returns:
        Dict[str, Dict[int, float]]: 每周的指标值
    """
    # 计算每周的指标值
    result = calculate_weekly_metrics(history)
    
    # 将结果放入列表中（为了兼容 calculate_average_metrics 的输入格式）
    results = [result]
    
    # 计算平均结果
    avg_result = calculate_average_metrics(results)
    
    return avg_result


def calculate_error_detailed(simulated: Dict[str, Dict[int, float]], 
                            empirical: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
    """
    详细计算模拟结果与经验数据之间的误差
    
    Args:
        simulated: 模拟结果字典，格式为 {指标: {周数: 值}}
        empirical: 经验数据字典，格式为 {指标: {周数: 值}}
        
    Returns:
        Dict: 包含详细误差信息的字典，包括：
            - total_nrmse: 总体归一化均方根误差
            - metric_errors: 各指标的NRMSE
            - detailed_errors: 每周的详细误差
            - missing_data: 缺失数据统计
    """
    # 初始化结果字典
    result = {
        'total_rmse': 0.0,
        'metric_errors': {},
        'detailed_errors': {},
        'missing_data': {
            'total_missing': 0,
            'missing_by_metric': Counter()
        }
    }
    
    # 定义指标顺序
    metrics = ['w_risk_p', 'm_risk_p', 'sentiment_high_p', 
              'sentiment_middle_p', 'sentiment_low_p']
    
    # 计算每个指标的误差
    for metric in metrics:
        errors = []
        detailed_errors = {}
        emp_values = empirical.get(metric, {})
        sim_values = simulated.get(metric, {})
        
        # 计算每周误差
        for week in emp_values:
            if week in sim_values:
                sim_value = sim_values[week]
                emp_value = emp_values[week]
                error = (sim_value - emp_value) ** 2
                errors.append(error)
                detailed_errors[week] = {
                    'simulated': sim_value,
                    'empirical': emp_value,
                    'squared_error': error
                }
            else:
                result['missing_data']['total_missing'] += 1
                result['missing_data']['missing_by_metric'][metric] += 1
        
        # 计算当前指标的RMSE
        if errors:
            mse = np.mean(errors)
            rmse = np.sqrt(mse)
            # 归一化处理
            # emp_range = max(emp_values.values()) - min(emp_values.values())
            # nrmse = rmse / emp_range if emp_range > 0 else 0.0
            result['metric_errors'][metric] = rmse
            result['detailed_errors'][metric] = detailed_errors
        else:
            result['metric_errors'][metric] = float('inf')
            result['detailed_errors'][metric] = {}
    
    # 计算总体RMSE
    # 选取 sentiment_high_p ,sentiment_middle_p , sentiment_low_p 进行计算
    valid_errors = [result['metric_errors']['sentiment_high_p'], result['metric_errors']['sentiment_middle_p'], result['metric_errors']['sentiment_low_p']]
    if valid_errors:
        result['total_rmse'] = np.mean(valid_errors)
    
    return result