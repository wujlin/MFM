import math
from collections import Counter
from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
from scipy.stats import entropy
from dataclasses import dataclass, field

# ================ 工具函数 ================
def calculate_entropy(state_distribution: dict) -> float:
    """计算系统状态的信息熵"""
    entropies = []
    for subsystem in state_distribution.values():
        values = np.array(list(subsystem.values()))
        probs = values / np.sum(values)
        subsystem_entropy = entropy(probs)
        entropies.append(subsystem_entropy)
    return np.mean(entropies)

def detect_change_point(history: List[dict], window_size: int = 5) -> float:
    """检测系统状态变化点"""
    if len(history) < window_size + 1:
        return 0.0
    current_entropy = calculate_entropy(history[-1])
    past_entropy = calculate_entropy(history[-window_size-1])
    entropy_rate = (current_entropy - past_entropy) / window_size
    return abs(entropy_rate)

def adaptive_regulatory_field(g_m_base: float, 
                          change_rate: float,
                          history: List[dict],
                          sensitivity: float = 8.0) -> float:
    """计算自适应调控场强度"""
    if len(history) < 5:
        return 0.0
    past_rates = [detect_change_point(history[:i+1]) for i in range(len(history)-1)]
    threshold = np.std(past_rates) * 2
    if abs(change_rate) < threshold:
        return 0.0
    activation = 1 / (1 + np.exp(-sensitivity * (abs(change_rate) - threshold)))
    return g_m_base * activation

# ================ 节点类 ================
class Node:
    """网络节点基类"""
    def __init__(self, name: str, type: str, risk: str = None, 
                 sentiment: str = None, intensity: float = None):
        self.name = name
        self.type = type
        self.risk = risk
        self.sentiment = sentiment
        self.intensity = intensity
        self.influencer_names = []  # 存储影响者的名称而不是对象
        
    def __getstate__(self):
        """自定义序列化状态"""
        state = self.__dict__.copy()
        # 只保存影响者的名称
        state['influencer_names'] = [inf.name for inf in getattr(self, 'influencers', [])]
        # 删除influencers属性，避免循环引用
        state.pop('influencers', None)
        return state
    
    def __setstate__(self, state):
        """自定义反序列化状态"""
        self.__dict__.update(state)
        # influencers将在Network类中重建
        self.influencers = []

    def add_influencer(self, node):
        """添加影响者"""
        if not hasattr(self, 'influencers'):
            self.influencers = []
        self.influencers.append(node)
        self.influencer_names.append(node.name)

    # -------- 原始动力学模型 --------
    def update_m_media(self, params: Dict):
        """主流媒体节点更新规则"""
        if self.type == 'm_media':
            alpha = params['alpha']
            d = params['d']
            if self.risk == 'R':
                if np.random.random() < alpha * (1 - d):
                    self.risk = 'NR'
            else:  # NR
                if np.random.random() < alpha * (1 + d):
                    self.risk = 'R'

    def update_w_media(self, params: Dict):
        """自媒体节点更新规则"""
        if self.type == 'w_media':
            beta = params['beta']
            d = params['d']
            if self.risk == 'R':
                if np.random.random() < beta * (1 - d):
                    self.risk = 'NR'
            else:  # NR
                if np.random.random() < beta * (1 + d):
                    self.risk = 'R'

    def update_o_people(self, params: Dict):
        """普通用户节点更新规则"""
        if self.type == 'o_people':
            # 获取参数
            intensity = self.intensity
            risk_m = params['risk_m']
            risk_w = params['risk_w']
            norisk_m = params['norisk_m']
            norisk_w = params['norisk_w']
            theta = params['theta']
            sigma = params['sigma']
            
            # 计算全局和局部风险
            risk_global = risk_m + risk_w
            norisk_global = norisk_m + norisk_w
            
            # 统计邻居状态
            risk_counter = Counter([inf.risk for inf in self.influencers if inf.risk])
            sentiment_counter = Counter([inf.sentiment for inf in self.influencers if inf.sentiment])
            
            # 更新情绪强度
            if self.sentiment == 'H':
                d1 = norisk_global - risk_global
                d2 = sentiment_counter.get('M', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('H', 0)
            elif self.sentiment == 'M':
                d1 = abs(norisk_global - risk_global)
                d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('L', 0) - sentiment_counter.get('M', 0)
            else:  # L
                d1 = risk_global - norisk_global
                d2 = sentiment_counter.get('H', 0) + sentiment_counter.get('M', 0) - sentiment_counter.get('L', 0)
            
            # 更新情绪强度
            intensity = intensity - ((np.tanh(theta * d1) + np.tanh(sigma * d2)) + 2) / 4
            intensity = np.clip(intensity, 0, 1)
            
            # 情绪状态转换
            if intensity > np.random.random():
                if self.sentiment == 'M':
                    # 转换为H或L的概率与风险差值有关
                    d = risk_global - norisk_global
                    Pw = (1 + np.tanh(d)) / 2
                    self.sentiment = 'H' if np.random.random() < Pw else 'L'
                else:  # H或L
                    self.sentiment = 'M'
            
            self.intensity = intensity

    # -------- 带调控场的动力学模型 --------
    def update_m_media_gv(self, params: Dict):
        """带调控场的主流媒体更新"""
        if self.type == 'm_media':
            g_m = params['g_m']  # 调控场强度
            self.update_m_media(params)  # 原始更新
            # 调控场效应
            if g_m > 0 and self.risk == 'R':
                if np.random.random() < g_m:
                    self.risk = 'NR'

    def update_w_media_gv(self, params: Dict):
        """带调控场的自媒体更新"""
        if self.type == 'w_media':
            g_m = params['g_m']  # 调控场强度
            self.update_w_media(params)  # 原始更新
            # 调控场效应
            if g_m > 0 and self.risk == 'R':
                if np.random.random() < g_m:
                    self.risk = 'NR'

    def update_o_people_gv(self, params: Dict):
        """带调控场的普通用户更新"""
        if self.type == 'o_people':
            g_m = params['g_m']  # 调控场强度
            self.update_o_people(params)  # 原始更新
            # 调控场效应：增加转向中性情绪的概率
            if g_m > 0 and self.sentiment != 'M':
                if np.random.random() < g_m:
                    self.sentiment = 'M'

    def forward(self, params: Dict):
        """原始动力学模型前向传播"""
        self.update_m_media(params)
        self.update_w_media(params)
        self.update_o_people(params)

    def forward_gv(self, params: Dict):
        """带调控场的动力学模型前向传播"""
        self.update_m_media_gv(params)
        self.update_w_media_gv(params)
        self.update_o_people_gv(params)

# ================ 网络类 ================
class Network:
    """网络基类"""
    def __init__(self):
        self.nodes = {}
        self.t = 0
        self.history = []
    
    def __getstate__(self):
        """自定义序列化状态"""
        state = self.__dict__.copy()
        # 转换节点为可序列化格式
        nodes_data = {}
        for name, node in self.nodes.items():
            nodes_data[name] = {
                'name': node.name,
                'type': node.type,
                'risk': node.risk,
                'sentiment': node.sentiment,
                'intensity': node.intensity,
                'influencer_names': node.influencer_names
            }
        state['nodes_data'] = nodes_data
        # 删除原始nodes字典
        del state['nodes']
        return state
    
    def __setstate__(self, state):
        """自定义反序列化状态"""
        self.__dict__.update(state)
        # 重建节点
        self.nodes = {}
        for name, data in state['nodes_data'].items():
            node = Node(
                name=data['name'],
                type=data['type'],
                risk=data['risk'],
                sentiment=data['sentiment'],
                intensity=data['intensity']
            )
            self.nodes[name] = node
        
        # 重建节点间的影响关系
        for name, data in state['nodes_data'].items():
            node = self.nodes[name]
            for inf_name in data['influencer_names']:
                node.add_influencer(self.nodes[inf_name])
        
        # 删除临时数据
        del self.nodes_data

    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.name] = node

    def calculate_sentiment_distribution(self) -> Counter:
        """计算情绪分布"""
        return Counter(node.sentiment for node in self.nodes.values() 
                      if node.type == 'o_people')

    def calculate_state_proportions(self) -> Dict[str, Dict[str, float]]:
        """计算状态分布"""
        states = {
            'm_media': Counter({'R': 0, 'NR': 0}),
            'w_media': Counter({'R': 0, 'NR': 0}),
            'o_people': Counter({'H': 0, 'M': 0, 'L': 0})
        }
        
        for node in self.nodes.values():
            if node.type in ['m_media', 'w_media']:
                states[node.type][node.risk] += 1
            elif node.type == 'o_people':
                states[node.type][node.sentiment] += 1
                
        return {type_: {state: count / sum(states[type_].values()) 
                       for state, count in states[type_].items()} 
                for type_ in states}

    def simulate_step(self, **params):
        """执行单步模拟"""
        self.t += 1
        sentiment_counts = self.calculate_sentiment_distribution()
        property = self.calculate_state_proportions()
        
        # 更新参数
        params.update({
            't': self.t,
            'risk_m': property['m_media']['R'],
            'risk_w': property['w_media']['R'],
            'norisk_m': property['m_media']['NR'],
            'norisk_w': property['w_media']['NR'],
            'd': (sentiment_counts['H'] + sentiment_counts['L'] - 
                 sentiment_counts['M']) / (sentiment_counts['M'] + 1),
            'n_high': sentiment_counts['H'],
            'n_low': sentiment_counts['L'],
            'n_middle': sentiment_counts['M']
        })
        
        for node in self.nodes.values():
            node.forward(params)

    def simulate_steps(self, steps: int = 91, **params) -> List[Dict]:
        """执行多步模拟
        
        Args:
            steps: 模拟总时长，默认为91（与经验数据匹配）
            params: 必须包含以下参数：
                - alpha: 主流媒体响应系数
                - beta: 自媒体响应系数
                - theta: 情感转换系数
                - sigma: 社交影响系数
                - zeta: 记忆衰减系数
                - miu: 噪声系数
                - g_m: 基础调控场强度（在旧代码中为g_m_base）
        
        Returns:
            List[Dict]: 模拟历史记录
        """
        self.history = [self.calculate_state_proportions()]
        self.t = 0
        
        # 确保参数名称一致性
        if 'g_m' in params and 'g_m_base' not in params:
            params['g_m_base'] = params['g_m']
        
        for _ in range(steps):
            self.simulate_step_gv(**params)
        
        return self.history

    def simulate_step_gv(self, **params):
        """执行带调控场的单步模拟"""
        self.t += 1
        current_state = self.calculate_state_proportions()
        self.history.append(current_state)
        
        # 计算调控场强度
        change_rate = detect_change_point(self.history)
        
        # 确保参数一致性
        g_m_base = params.get('g_m_base', params.get('g_m', 0.0))
        
        g_m = adaptive_regulatory_field(
            g_m_base=g_m_base,
            change_rate=change_rate,
            history=self.history
        )
        
        # 更新参数
        params['g_m'] = g_m
        self.simulate_step(**params)