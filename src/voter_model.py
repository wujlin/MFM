import numpy as np
from typing import List, Dict
from collections import Counter
from copy import deepcopy

class VoterNode:
    def __init__(self, name, node_type, state=None):
        self.name = name
        self.type = node_type
        self.state = state  # 对于media节点是{'R', 'NR'}，对于people节点是{'H', 'M', 'L'}
        self.influencers = []
        
    def add_influencer(self, influencer):
        self.influencers.append(influencer)
        
    def update(self):
        if not self.influencers:
            return
            
        # 随机选择一个邻居并模仿其状态
        neighbor = np.random.choice(self.influencers)
        if self.type == 'o_people':
            self.state = neighbor.sentiment if hasattr(neighbor, 'sentiment') else neighbor.state
        else:  # m_media or w_media
            self.state = neighbor.risk if hasattr(neighbor, 'risk') else neighbor.state

class VoterModel:
    def __init__(self, original_network):
        """
        基于原始网络构建Voter Model
        
        Args:
            original_network: 原始CSDAG网络实例
        """
        self.nodes = {}
        self.t = 0
        self.history = []
        
        # 将原始网络转换为Voter Model网络
        for name, node in original_network.nodes.items():
            if node.type == 'o_people':
                state = node.sentiment
            else:
                state = node.risk
            
            voter_node = VoterNode(name, node.type, state)
            self.nodes[name] = voter_node
            
        # 复制网络连接
        for name, node in original_network.nodes.items():
            for influencer in node.influencers:
                self.nodes[name].add_influencer(self.nodes[influencer.name])
    
    def simulate_step(self):
        """执行一步Voter Model更新"""
        self.t += 1
        
        # 随机更新顺序
        update_order = list(self.nodes.keys())
        np.random.shuffle(update_order)
        
        # 保存当前状态以实现同步更新
        current_states = {name: deepcopy(node.state) for name, node in self.nodes.items()}
        
        # 更新所有节点
        for name in update_order:
            self.nodes[name].update()
            
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
    
    def calculate_state_proportions(self) -> Dict[str, Dict[str, float]]:
        """计算各类节点的状态分布"""
        states = {
            'm_media': Counter({'R': 0, 'NR': 0}),
            'w_media': Counter({'R': 0, 'NR': 0}),
            'o_people': Counter({'H': 0, 'M': 0, 'L': 0})
        }
        
        for node in self.nodes.values():
            if node.type in ['m_media', 'w_media']:
                states[node.type][node.state] += 1
            elif node.type == 'o_people':
                states[node.type][node.state] += 1
                
        proportions = {
            type_: {
                state: count / sum(states[type_].values()) 
                for state, count in states[type_].items()
            } 
            for type_ in states
        }
        
        return proportions