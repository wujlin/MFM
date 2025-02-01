import numpy as np
from typing import List, Dict
from collections import Counter
from copy import deepcopy

class VoterNode:
    def __init__(self, name, node_type, state=None, influence_strength=1.0):
        self.name = name
        self.type = node_type
        # 分别存储risk和sentiment
        if node_type in ['m_media', 'w_media']:
            self.risk = state
            self.sentiment = None
        else:  # o_people
            self.sentiment = state
            self.risk = None
        self.influence_strength = influence_strength
        self.influencers = []
        
    def add_influencer(self, influencer):
        self.influencers.append(influencer)
        
    def update(self, temperature=1.0):
        if not self.influencers:
            return
            
        if self.type == 'o_people':
            # 统计邻居节点的状态
            sentiment_counter = Counter()
            for inf in self.influencers:
                if hasattr(inf, 'sentiment') and inf.sentiment:
                    sentiment_counter[inf.sentiment] += inf.influence_strength
                elif hasattr(inf, 'risk') and inf.risk:
                    # 媒体的风险状态映射到情绪状态
                    mapped_state = 'H' if inf.risk == 'NR' else 'L'
                    sentiment_counter[mapped_state] += inf.influence_strength
            
            # 确保所有可能的状态都有计数
            for state in ['H', 'M', 'L']:
                if state not in sentiment_counter:
                    sentiment_counter[state] = 0
                    
            # 根据当前状态和邻居状态决定转换
            if self.sentiment == 'H':
                # 从H状态可以转换到M或保持H
                if sentiment_counter['M'] > max(sentiment_counter['H'], sentiment_counter['L']):
                    self.sentiment = 'M'
            elif self.sentiment == 'M':
                # 从M状态可以转换到H、L或保持M
                max_count = max(sentiment_counter.values())
                max_states = [s for s, c in sentiment_counter.items() if c == max_count]
                if len(max_states) == 1:
                    self.sentiment = max_states[0]
            elif self.sentiment == 'L':
                # 从L状态可以转换到M或保持L
                if sentiment_counter['M'] > max(sentiment_counter['H'], sentiment_counter['L']):
                    self.sentiment = 'M'
                    
        else:  # m_media or w_media
            # 统计邻居节点的风险状态
            risk_counter = Counter()
            for inf in self.influencers:
                if hasattr(inf, 'risk') and inf.risk:
                    risk_counter[inf.risk] += inf.influence_strength
                elif hasattr(inf, 'sentiment') and inf.sentiment:
                    # 情绪状态映射到风险状态
                    mapped_state = 'NR' if inf.sentiment == 'H' else 'R'
                    risk_counter[mapped_state] += inf.influence_strength
            
            # 确保所有可能的状态都有计数
            for state in ['R', 'NR']:
                if state not in risk_counter:
                    risk_counter[state] = 0
                    
            # 采用多数规则
            if risk_counter['R'] > risk_counter['NR']:
                self.risk = 'R'
            elif risk_counter['NR'] > risk_counter['R']:
                self.risk = 'NR'


class EnhancedVoterModel:
    def __init__(self, original_network, temperature=1.0):
        """
        增强版Voter Model
        
        Args:
            original_network: 原始CSDAG网络实例
            temperature: 温度参数，控制状态转换的随机性
        """
        self.nodes = {}
        self.t = 0
        self.history = []
        self.temperature = temperature
        
        # 设置不同类型节点的基础影响力
        influence_strengths = {
            'm_media': 2.0,    # 主流媒体影响力较大
            'w_media': 1.5,    # 自媒体影响力中等
            'o_people': 1.0    # 普通用户基准影响力
        }
        
        # 初始化节点
        for name, node in original_network.nodes.items():
            if node.type == 'o_people':
                state = node.sentiment
            else:
                state = node.risk
            
            voter_node = VoterNode(
                name=name,
                node_type=node.type,
                state=state,
                influence_strength=influence_strengths[node.type]
            )
            self.nodes[name] = voter_node
            
        # 复制网络连接
        for name, node in original_network.nodes.items():
            for influencer in node.influencers:
                self.nodes[name].add_influencer(self.nodes[influencer.name])
    
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
    
    def simulate_step(self):
        """执行一步模拟"""
        self.t += 1
        
        # 随机更新顺序
        update_order = list(self.nodes.keys())
        np.random.shuffle(update_order)
        
        # 更新所有节点
        for name in update_order:
            self.nodes[name].update(self.temperature)
            
        # 记录状态分布
        current_distribution = self.calculate_state_proportions()
        self.history.append(current_distribution)
    
    def simulate_steps(self, steps: int = 91) -> List[Dict]:
        """
        执行多步模拟
        
        Args:
            steps: 模拟总时长
            
        Returns:
            List[Dict]: 模拟历史记录
        """
        # 初始化历史记录
        ini_distribution = self.calculate_state_proportions()
        self.history = [ini_distribution]
        self.t = 0
        
        # 执行模拟
        for _ in range(steps):
            self.simulate_step()
            
        return self.history