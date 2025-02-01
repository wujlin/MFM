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

def calculate_entropy(state_distribution: dict) -> float:
    """
    计算系统状态的信息熵，处理嵌套字典结构
    
    Args:
        state_distribution: 包含各状态分布的嵌套字典
        
    Returns:
        float: 信息熵值
    """
    # 分别计算每个子系统的熵
    entropies = []
    
    for subsystem in state_distribution.values():
        values = np.array(list(subsystem.values()))
        # 确保概率和为1
        probs = values / np.sum(values)
        subsystem_entropy = entropy(probs)
        entropies.append(subsystem_entropy)
    
    # 返回总体熵（各子系统熵的平均值）
    return np.mean(entropies)

def detect_change_point(history: List[dict], window_size: int = 5) -> float:
    """
    计算系统状态变化率
    
    Args:
        history: 历史状态分布列表
        window_size: 滑动窗口大小
    
    Returns:
        float: 状态变化率
    """
    if len(history) < window_size + 1:
        return 0.0
        
    # 计算当前和历史熵值
    current_entropy = calculate_entropy(history[-1])
    past_entropy = calculate_entropy(history[-window_size-1])
    
    # 计算熵变化率
    entropy_rate = (current_entropy - past_entropy) / window_size
    return abs(entropy_rate)

def adaptive_regulatory_field(g_m_base: float, 
                            change_rate: float,
                            history: List[dict],
                            sensitivity: float = 8.0) -> float:
    """
    基于系统状态变化计算自适应调控场强度
    
    Args:
        g_m_base: 基础调控场强度
        change_rate: 当前系统状态变化率
        history: 历史状态分布
        sensitivity: 灵敏度参数
    
    Returns:
        float: 当前时间步的调控场强度
    """
    if len(history) < 5:
        return 0.0
        
    # 计算动态阈值：使用历史变化率的标准差
    past_rates = [detect_change_point(history[:i+1]) for i in range(len(history)-1)]
    threshold = np.std(past_rates) * 2  # 使用2倍标准差作为动态阈值
    
    # 只有当变化率超过动态阈值时才激活调控
    if abs(change_rate) < threshold:
        return 0.0
        
    # 使用sigmoid函数将变化率映射到[0,1]
    activation = 1 / (1 + np.exp(-sensitivity * (abs(change_rate) - threshold)))
    return g_m_base * activation


def sigmoid_scale(g_m, k=0.1):
    """
    将g_m映射到(0,1)区间
    k: 控制曲线陡度的参数
    """
    return 1 / (1 + np.exp(-k * g_m))


class Node:
    def __init__(self, name, node_type, risk=None, sentiment=None, intensity=None):
        self.name = name
        self.type = node_type
        self.risk = risk
        self.sentiment = sentiment
        self.intensity = intensity
        self.influencers = []
        # self.sigma = sigma

    def add_influencer(self, influencer):
        self.influencers.append(influencer)

    def calculate_n_major(self):
        # 仅适用于 o_people 类型的节点
        sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
        if sentiment_counter:
            return max(sentiment_counter.values())  # 返回最多的情绪类别的数量
        return 0

    def calculate_influencers_sentiment_distribution(self):
        sentiment_counts = Counter(inf.sentiment for inf in self.influencers if inf.sentiment)
        n_high = sentiment_counts.get('H', 0)
        n_low = sentiment_counts.get('L', 0)
        n_middle = sentiment_counts.get('M', 0)
        return n_high, n_low, n_middle
    
    def calculate_sentiment_distribution(self):
        """
        计算影响者的情感分布
        Returns:
            tuple: (n_high, n_low, n_middle) 各情感状态的数量
        """
        sentiment_counter = Counter(inf.sentiment for inf in self.influencers if inf.sentiment)
        n_high = sentiment_counter.get('H', 0)
        n_low = sentiment_counter.get('L', 0)
        n_middle = sentiment_counter.get('M', 0)
        return n_high, n_low, n_middle
    
    def update_m_media(self, params):
        if self.type == 'm_media':
            alpha = params['alpha']
            # n_high, n_low, n_middle = self.calculate_influencers_sentiment_distribution() # 计算情绪分布
            d = params['d']
            n_high_global = params['n_high']
            n_low_global = params['n_low']
            n_middle_global = params['n_middle']
            sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
            # 添加默认情绪类别为 0
            for sentiment in ['H', 'M', 'L']:
                if sentiment not in sentiment_counter:
                    sentiment_counter[sentiment] = 0
            
            n_high_local = sentiment_counter.get('H', 0)
            n_low_local = sentiment_counter.get('L', 0)
            n_middle_local = sentiment_counter.get('M', 0)


            n_high = n_high_local
            n_low = n_low_local
            n_middle = n_middle_local

            if n_middle > n_high and n_middle > n_low:
                pass

            elif n_high > n_middle and n_high > n_low:
                d = 2 * n_high - n_middle - n_low
                Pu = 1 - math.exp(-alpha * d)
                if Pu > np.random.random():
                    self.risk = 'NR'

            elif n_low > n_middle and n_low > n_high:
                d = 2 * n_low - n_middle - n_high
                Pu = 1 - math.exp(-alpha * d)
                if Pu > np.random.random():
                    self.risk = 'R'
                # else:
                #     self.risk = 'NR'
            
            else:   # 三者相等
                d = n_middle
                Pu = 1 - math.exp(-alpha * d)
                if Pu > np.random.random():
                    self.risk = 'NR'



    def update_m_media_gv(self, params):
        """更新主流媒体节点状态，包含自适应调控场效应"""
        if self.type == 'm_media':
            alpha = params['alpha']
            g_m = params['g_m']  # 使用计算得到的自适应调控场强度
            
            sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
            for sentiment in ['H', 'M', 'L']:
                if sentiment not in sentiment_counter:
                    sentiment_counter[sentiment] = 0
            
            n_high = sentiment_counter.get('H', 0)
            n_low = sentiment_counter.get('L', 0)
            n_middle = sentiment_counter.get('M', 0)

            if n_middle > n_high and n_middle > n_low:
                Pu = sigmoid_scale(g_m)
                if Pu > np.random.random():
                    self.risk = 'NR'
            elif n_high > n_middle and n_high > n_low:
                d = 2 * n_high - n_middle - n_low
                Pu = min(1, ((1 - math.exp(-alpha * d)) + sigmoid_scale(g_m))/2)
                if Pu > np.random.random():
                    self.risk = 'NR'
            elif n_low > n_middle and n_low > n_high:
                d = 2 * n_low - n_middle - n_high
                Pu = min(1, ((1 - math.exp(-alpha * d)) + sigmoid_scale(g_m))/2)
                if Pu > np.random.random():
                    self.risk = 'R'
            else:
                d = n_middle
                Pu = min(1, ((1 - math.exp(-alpha * d)) + sigmoid_scale(g_m))/2)
                if Pu > np.random.random():
                    self.risk = 'NR'
        
    def update_w_media(self, params: Dict[str, Any]) -> None:
        """
        Update we-media node's risk perception.
        """
        if self.type == 'w_media':

            beta = params['beta']
            n_high, n_low, n_middle = self.calculate_sentiment_distribution() # 这里使用了全局的分布
            
            # 计算情感熵
            total_sentiments = n_high + n_low + n_middle
            sentiment_probs = ([n_high/total_sentiments, n_low/total_sentiments, n_middle/total_sentiments]
                            if total_sentiments > 0 else [1/3, 1/3, 1/3])
            sentiment_entropy = entropy(sentiment_probs)

            risk_global_w = params['risk_w']
            norisk_global_w = params['norisk_w']
            risk_global_m = params['risk_m']
            norisk_global_m = params['norisk_m']
            risk_global = risk_global_m + risk_global_w
            norisk_global = norisk_global_m + norisk_global_w
            Pv_2 = (np.tanh(norisk_global - risk_global) + 1) / 2

            # 更新：结合情感分布和风险分布

            Pv_1 = 1 - math.exp(-beta * sentiment_entropy)
            if Pv_1*Pv_2 > np.random.random():
                self.risk = 'NR'
            else:
                self.risk = 'R'


    
    def update_w_media_gv(self, params: Dict[str, Any]) -> None:
        """
        Update we-media node state with government influence.
        
        Args:
            params: Dictionary containing simulation parameters
            government_effect_w: Government influence factor on we-media
        """
        if self.type == 'w_media':

            beta = params['beta']
            n_high, n_low, n_middle = self.calculate_sentiment_distribution()
            
            total_sentiments = sum([n_high, n_low, n_middle])
            sentiment_probs = ([n_high/total_sentiments, n_low/total_sentiments, n_middle/total_sentiments]
                            if total_sentiments > 0 else [1/3, 1/3, 1/3])
            sentiment_entropy = entropy(sentiment_probs)

            risk_global_w = params['risk_w']
            norisk_global_w = params['norisk_w']
            risk_global_m = params['risk_m']
            norisk_global_m = params['norisk_m']
            risk_global = risk_global_m + risk_global_w
            norisk_global = norisk_global_m + norisk_global_w
            Pv_2 = (np.tanh(norisk_global - risk_global) + 1) / 2


            Pv_1 = 1 - math.exp(-beta * sentiment_entropy)
            if Pv_1*Pv_2 > np.random.random():
                self.risk = 'NR'
            else:
                self.risk = 'R'


    def update_o_people(self, params):
        if self.type == 'o_people':
            intensity = self.intensity
            # 这里的r应该规范化
            risk_m = params['risk_m']
            risk_w = params['risk_w']
            norisk_m = params['norisk_m']
            norisk_w = params['norisk_w']
            theta = params['theta']
            sigma = params['sigma']
            zeta = params['zeta']
            miu = params['miu']
            risk_global = risk_m + risk_w
            norisk_global = norisk_m + norisk_w
            risk_counter = Counter([inf.risk for inf in self.influencers if inf.risk])
            # 添加默认风险类别为 0
            for risk in ['R', 'NR']:
                if risk not in risk_counter:
                    risk_counter[risk] = 0


            risk = risk_global
            norisk = norisk_global

            sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
            # 添加默认情绪类别为 0
            for sentiment in ['H', 'M', 'L']:
                if sentiment not in sentiment_counter:
                    sentiment_counter[sentiment] = 0

            if self.sentiment == 'H':
                d1 = norisk - risk
                d2 = sentiment_counter.get('M', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('H', 0)
            elif self.sentiment == 'M':
                # 绝对值
                d1 = abs(norisk - risk)  # 这里选用了绝对值，是因为我们认为风险和非风险的差距越大，越容易产生情绪的变化
                d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('M', 0)
            elif self.sentiment == 'L':
                d1 = risk - norisk
                d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('M', 0) - sentiment_counter.get('L', 0)
            
            # intensity越小，越容易产生情绪的变化
            intensity = intensity - ((np.tanh(theta * d1) + np.tanh(sigma * d2)) + 2) / 4 # 这里我们添加了risk信息机制还有homophily机制

            d = risk - norisk
            Pw = (1 + np.tanh(d)) / 2

            if intensity > 1:
                intensity = 1
            elif intensity < 0:
                intensity = 0

            # 判断所有 sentiment_counter 的值是否相等
            def all_values_equal(d):
                values = list(d.values())
                return values.count(values[0]) == len(values)
            
            # 修改后的代码
            if intensity > np.random.random():
                if self.sentiment == 'M':
                    if np.random.random() < zeta:
                        if Pw > np.random.random():
                            self.sentiment = 'H'
                        else:
                            self.sentiment = 'L'
                        # else:
                        #     self.sentiment = np.random.choice(['H', 'L'])
                    else:
                        if all_values_equal(sentiment_counter):
                            # self.sentiment = 'L'  # 如果 sentiment_counter 中所有值相等，保留现有的 sentiment
                            # self.sentiment = np.random.choice(['H', 'L'])
                            if Pw > np.random.random():
                                self.sentiment = 'H'
                            else:
                                self.sentiment = 'L'
                            # else:   
                            #     self.sentiment = np.random.choice(['H', 'L'])
                        else:
                            self.sentiment = max(sentiment_counter, key=sentiment_counter.get)

                elif self.sentiment == 'H':
                    if np.random.random() < zeta:
                        self.sentiment = 'M'
                    else:
                        if all_values_equal(sentiment_counter):
                            # self.sentiment = np.random.choice(['M', 'L'])
                            if Pw > np.random.random():
                                self.sentiment = 'H'
                            else:
                                self.sentiment = 'L'
                            # else:   
                            #     self.sentiment = np.random.choice(['H', 'L'])
                        else:
                            self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
                            
                elif self.sentiment == 'L':
                    if np.random.random() < zeta:
                        self.sentiment = 'M'
                    else:
                        if all_values_equal(sentiment_counter):
                            # self.sentiment = np.random.choice(['H', 'M'])
                            if Pw > np.random.random():
                                self.sentiment = 'H'
                            else:
                                self.sentiment = 'L'
                            # else:   
                            #     self.sentiment = np.random.choice(['H', 'L'])
                        else:
                            self.sentiment = max(sentiment_counter, key=sentiment_counter.get)

    
    def update_o_people_gv(self, params):
        # print('update_o_people_gv')
        if self.type == 'o_people':
            intensity = self.intensity
            # 这里的r应该规范化
            risk_m = params['risk_m']
            risk_w = params['risk_w']
            norisk_m = params['norisk_m']
            norisk_w = params['norisk_w']
            theta = params['theta']
            sigma = params['sigma']
            zeta = params['zeta']
            miu = params['miu']

            risk_global = risk_m + risk_w
            norisk_global = norisk_m + norisk_w
            risk_counter = Counter([inf.risk for inf in self.influencers if inf.risk])
            # 添加默认风险类别为 0
            for risk in ['R', 'NR']:
                if risk not in risk_counter:
                    risk_counter[risk] = 0


            risk = risk_global
            norisk = norisk_global

            sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
            # 添加默认情绪类别为 0
            for sentiment in ['H', 'M', 'L']:
                if sentiment not in sentiment_counter:
                    sentiment_counter[sentiment] = 0

            if self.sentiment == 'H':
                d1 = norisk - risk
                d2 = sentiment_counter.get('M', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('H', 0)
            elif self.sentiment == 'M':
                # 绝对值
                d1 = abs(norisk - risk)
                d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('M', 0)
            elif self.sentiment == 'L':
                d1 = risk - norisk
                d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('M', 0) - sentiment_counter.get('L', 0)
            
            intensity = intensity - ((np.tanh(theta * d1) + np.tanh(sigma * d2)) + 2) / 4 # 这里我们添加了risk信息机制还有homophily机制

            d = risk - norisk
            # Pw = 1 - math.exp(-miu * d)
            Pw = (1 + np.tanh(d)) / 2
            if intensity > 1:
                intensity = 1
            elif intensity < 0:
                intensity = 0
            
            # 判断所有 sentiment_counter 的值是否相等
            def all_values_equal(d):
                values = list(d.values())
                return values.count(values[0]) == len(values)
            
            # 修改后的代码
            if intensity > np.random.random():
                if self.sentiment == 'M':
                    if np.random.random() < zeta:
                        if Pw > np.random.random():
                            self.sentiment = 'H'
                        else:
                            self.sentiment = 'L'
                        # else:
                        #     self.sentiment = np.random.choice(['H', 'L'])
                    else:
                        if all_values_equal(sentiment_counter):
                            # self.sentiment = 'L'  # 如果 sentiment_counter 中所有值相等，保留现有的 sentiment
                            # self.sentiment = np.random.choice(['H', 'L'])
                            if Pw > np.random.random():
                                self.sentiment = 'H'
                            else:
                                self.sentiment = 'L'
                            # else:   
                            #     self.sentiment = np.random.choice(['H', 'L'])
                        else:
                            self.sentiment = max(sentiment_counter, key=sentiment_counter.get)

                elif self.sentiment == 'H':
                    if np.random.random() < zeta:
                        self.sentiment = 'M'
                    else:
                        if all_values_equal(sentiment_counter):
                            # self.sentiment = np.random.choice(['M', 'L'])
                            if Pw > np.random.random():
                                self.sentiment = 'H'
                            else:
                                self.sentiment = 'L'
                            # else:   
                            #     self.sentiment = np.random.choice(['H', 'L'])
                        else:
                            self.sentiment = max(sentiment_counter, key=sentiment_counter.get)
                            
                elif self.sentiment == 'L':
                    if np.random.random() < zeta:
                        self.sentiment = 'M'
                    else:
                        if all_values_equal(sentiment_counter):
                            # self.sentiment = np.random.choice(['H', 'M'])
                            if Pw > np.random.random():
                                self.sentiment = 'H'
                            else:
                                self.sentiment = 'L'
                            # else:   
                            #     self.sentiment = np.random.choice(['H', 'L'])
                        else:
                            self.sentiment = max(sentiment_counter, key=sentiment_counter.get)


    def forward(self, params):
        self.update_m_media(params)
        self.update_w_media(params)
        self.update_o_people(params)

    def forward_gv(self, params):
        """带调控场的更新方法"""
        self.update_m_media_gv(params)
        self.update_w_media_gv(params)
        self.update_o_people_gv(params)

class Network:
    def __init__(self):
        self.nodes = {}
        self.t = 0
        self.history = []

    def add_node(self, node):
        self.nodes[node.name] = node

    def calculate_sentiment_distribution(self):
        sentiment_counts = Counter(node.sentiment for node in self.nodes.values() if node.type == 'o_people')
        return sentiment_counts

    def simulate_step(self, alpha, beta, theta, sigma, zeta, miu, g_m):
        self.t += 1
        sentiment_counts = self.calculate_sentiment_distribution()
        property = self.calculate_state_proportions()
        risk_m = property['m_media']['R']
        risk_w = property['w_media']['R']
        norisk_m = property['m_media']['NR']
        norisk_w = property['w_media']['NR']
        d = (sentiment_counts['H'] + sentiment_counts['L'] - sentiment_counts['M']) / (sentiment_counts['M'] + 1)
        params = {
            'alpha': alpha, 
            'beta': beta, 
            'theta': theta, 
            'sigma': sigma,
            'zeta': zeta, 
            'miu': miu,
            'g_m': g_m,
            'd': d,
            't': self.t,
            'n_high': sentiment_counts['H'], 
            'n_low': sentiment_counts['L'], 
            'n_middle': sentiment_counts['M'],
            'risk_m': risk_m,
            'risk_w': risk_w,
            'norisk_m': norisk_m,
            'norisk_w': norisk_w
        }
        for node in self.nodes.values():
            node.forward(params)
    
    def simulate_step_gv(self, alpha, beta, theta, sigma, zeta, miu, g_m_base):
        """带自适应调控场的单步模拟"""
        self.t += 1
        
        # 计算当前状态分布
        current_state = self.calculate_state_proportions()
        self.history.append(current_state)
        
        # 计算状态变化率
        change_rate = detect_change_point(self.history)
        
        # 计算自适应调控场强度
        g_m = adaptive_regulatory_field(
            g_m_base=g_m_base,
            change_rate=change_rate,
            history=self.history,
            sensitivity=8.0
        )
        
        # 更新其他参数
        sentiment_counts = self.calculate_sentiment_distribution()
        property = current_state
        risk_m = property['m_media']['R']
        risk_w = property['w_media']['R']
        norisk_m = property['m_media']['NR']
        norisk_w = property['w_media']['NR']
        
        d = (sentiment_counts['H'] + sentiment_counts['L'] - 
            sentiment_counts['M']) / (sentiment_counts['M'] + 1)
        
        params = {
            'alpha': alpha,
            'beta': beta,
            'theta': theta,
            'sigma': sigma,
            'zeta': zeta,
            'miu': miu,
            'g_m': g_m,
            'd': d,
            't': self.t,
            'n_high': sentiment_counts['H'],
            'n_low': sentiment_counts['L'],
            'n_middle': sentiment_counts['M'],
            'risk_m': risk_m,
            'risk_w': risk_w,
            'norisk_m': norisk_m,
            'norisk_w': norisk_w
        }
        
        for node in self.nodes.values():
            node.forward_gv(params)

    def simulate_steps(self, steps: int = 91, 
                    alpha: float = 0.5,
                    beta: float = 0.3,
                    theta: float = 0.4,
                    sigma: float = 0.2,
                    zeta: float = 0.1,
                    miu: float = 0.05,
                    g_m: float = 10.0) -> List[Dict]:
        """执行多步模拟，使用自适应调控场
        
        Args:
            steps: 模拟总时长，默认为91（与经验数据匹配）
            alpha: 主流媒体响应系数
            beta: 自媒体响应系数
            theta: 情感转换系数
            sigma: 社交影响系数
            zeta: 记忆衰减系数
            miu: 噪声系数
            g_m: 基础调控场强度
            
        Returns:
            List[Dict]: 模拟历史记录
        """
        # 初始化历史记录
        ini_distribution = self.calculate_state_proportions()
        self.history = [ini_distribution]
        self.t = 0  # 重置时间步
        
        # 执行模拟
        for _ in range(steps):
            self.simulate_step_gv(
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
        """计算各类节点的状态分布
        
        Returns:
            Dict[str, Dict[str, float]]: 各类节点的状态分布比例
        """
        # 初始化状态计数器
        states = {
            'm_media': Counter({'R': 0, 'NR': 0}),
            'w_media': Counter({'R': 0, 'NR': 0}),
            'o_people': Counter({'H': 0, 'M': 0, 'L': 0})
        }
        
        # 统计各类节点的状态
        for node in self.nodes.values():
            if node.type in ['m_media', 'w_media']:
                states[node.type][node.risk] += 1
            elif node.type == 'o_people':
                states[node.type][node.sentiment] += 1
                
        # 计算占比
        proportions = {
            type_: {
                state: count / sum(states[type_].values()) 
                for state, count in states[type_].items()
            } 
            for type_ in states
        }
        
        return proportions


