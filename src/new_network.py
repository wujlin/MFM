import networkx as nx
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
from src.new_modules import Node, Network  # 更新导入路径

class TheoreticalNetwork(Network):
    """理论网络模型，继承新的Network类"""
    
    def __init__(self, 
                network_type: str,
                n_mainstream: int = 10,
                n_wemedia: int = 100,
                n_people: int = 1000,
                p_rewire: float = 0.1,
                m_ba: int = 3):
        super().__init__()
        self.network_type = network_type
        self.n_mainstream = n_mainstream
        self.n_wemedia = n_wemedia
        self.n_people = n_people
        self.p_rewire = p_rewire
        self.m_ba = m_ba
        
        # 经验数据中的状态分布
        self.p_w_media = [0.683, 0.317]
        self.p_m_media = [0.651, 0.349]
        self.p_o_people = [0.318, 0.313, 0.369]
        self.p_sentiment_intensity = [0.975, 0.961, 0.960]
        
        # 生成网络
        self._generate_network()
    
    def __getstate__(self):
        """自定义序列化状态"""
        state = super().__getstate__()
        # 添加TheoreticalNetwork特有的属性
        state.update({
            'network_type': self.network_type,
            'n_mainstream': self.n_mainstream,
            'n_wemedia': self.n_wemedia,
            'n_people': self.n_people,
            'p_rewire': self.p_rewire,
            'm_ba': self.m_ba,
            'p_w_media': self.p_w_media,
            'p_m_media': self.p_m_media,
            'p_o_people': self.p_o_people,
            'p_sentiment_intensity': self.p_sentiment_intensity
        })
        return state
        
    def _generate_network(self):
        """生成三层理论网络结构"""
        # 生成各层网络
        m_media_net = self._create_layer_network(self.n_mainstream)
        w_media_net = self._create_layer_network(self.n_wemedia)
        people_net = self._create_layer_network(self.n_people)
        
        # 创建节点并初始化状态
        self._create_nodes(m_media_net, w_media_net, people_net)
        
        # 建立层间连接
        self._create_interlayer_edges()
        
    def _create_layer_network(self, n_nodes: int) -> nx.Graph:
        """根据网络类型创建单层网络"""
        if self.network_type == 'random':
            return nx.erdos_renyi_graph(n_nodes, self.p_rewire)
        elif self.network_type == 'ws':
            k = int(np.log2(n_nodes))  # 确保连通性
            return nx.watts_strogatz_graph(n_nodes, k, self.p_rewire)
        elif self.network_type == 'ba':
            m0 = min(self.m_ba + 1, n_nodes)  # 初始完全图的节点数
            return nx.barabasi_albert_graph(n_nodes, self.m_ba, m0)
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
            
    def _create_nodes(self, m_media_net: nx.Graph, w_media_net: nx.Graph, 
                     people_net: nx.Graph):
        """创建节点并建立层内连接"""
        # 创建主流媒体节点
        n_risk_m = int(self.n_mainstream * self.p_m_media[0])
        for i in range(self.n_mainstream):
            risk = 'R' if i < n_risk_m else 'NR'
            node = Node(f'm_media_{i}', 'm_media', risk=risk)
            self.add_node(node)
            
        # 创建自媒体节点
        n_risk_w = int(self.n_wemedia * self.p_w_media[0])
        for i in range(self.n_wemedia):
            risk = 'R' if i < n_risk_w else 'NR'
            node = Node(f'w_media_{i}', 'w_media', risk=risk)
            self.add_node(node)
            
        # 创建普通用户节点
        n_high = int(self.n_people * self.p_o_people[0])
        n_middle = int(self.n_people * self.p_o_people[1])
        n_low = self.n_people - n_high - n_middle
        
        node_count = 0
        # 创建高唤醒状态节点
        for i in range(n_high):
            node = Node(f'o_people_{node_count}', 'o_people', 
                       sentiment='H', 
                       intensity=self.p_sentiment_intensity[0])
            self.add_node(node)
            node_count += 1
            
        # 创建中等唤醒状态节点
        for i in range(n_middle):
            node = Node(f'o_people_{node_count}', 'o_people', 
                       sentiment='M', 
                       intensity=self.p_sentiment_intensity[1])
            self.add_node(node)
            node_count += 1
            
        # 创建低唤醒状态节点
        for i in range(n_low):
            node = Node(f'o_people_{node_count}', 'o_people', 
                       sentiment='L', 
                       intensity=self.p_sentiment_intensity[2])
            self.add_node(node)
            node_count += 1
            
        # 建立层内连接
        self._add_intralayer_edges(m_media_net, 'm_media')
        self._add_intralayer_edges(w_media_net, 'w_media')
        self._add_intralayer_edges(people_net, 'o_people')
        
    def _add_intralayer_edges(self, network: nx.Graph, layer_type: str):
        """建立层内节点连接"""
        for edge in network.edges():
            node1 = self.nodes[f'{layer_type}_{edge[0]}']
            node2 = self.nodes[f'{layer_type}_{edge[1]}']
            node1.add_influencer(node2)
            node2.add_influencer(node1)
            
    def _create_interlayer_edges(self):
        """建立层间连接"""
        # 主流媒体到用户的连接
        for i in range(self.n_people):
            person = self.nodes[f'o_people_{i}']
            # 随机连接到主流媒体
            n_connections = np.random.poisson(2)  # 平均连接2个主流媒体
            media_indices = np.random.choice(self.n_mainstream, 
                                          size=min(n_connections, self.n_mainstream),
                                          replace=False)
            for idx in media_indices:
                media = self.nodes[f'm_media_{idx}']
                person.add_influencer(media)
                
        # 自媒体到用户的连接
        for i in range(self.n_people):
            person = self.nodes[f'o_people_{i}']
            # 随机连接到自媒体
            n_connections = np.random.poisson(5)  # 平均连接5个自媒体
            media_indices = np.random.choice(self.n_wemedia,
                                          size=min(n_connections, self.n_wemedia),
                                          replace=False)
            for idx in media_indices:
                media = self.nodes[f'w_media_{idx}']
                person.add_influencer(media)