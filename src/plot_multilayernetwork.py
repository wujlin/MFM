import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



def prepare_sampled_network(network, w_media_ratio=0.1, o_people_ratio=0.1, min_degree=2, random_seed=42):
    """
    创建采样后的网络数据，优先保留有足够连接的节点
    
    Parameters:
    -----------
    network : Network
        原始网络
    w_media_ratio : float
        自媒体采样比例
    o_people_ratio : float
        普通用户采样比例
    min_degree : int
        最小连接度阈值
    random_seed : int
        随机种子
    """
    np.random.seed(random_seed)
    
    # 创建三个层的图和层间边列表
    layers = [nx.Graph() for _ in range(3)]
    interlayer_edges = []  # 初始化层间边列表
    
    # 按类型分类节点
    nodes_by_type = defaultdict(list)
    for node_name, node in network.nodes.items():
        nodes_by_type[node.type].append(node)
    
    # 计算每个节点的同类型连接数
    def count_same_type_connections(node):
        return sum(1 for inf in node.influencers if inf.type == node.type)
    
    # 筛选并排序节点（按连接数）
    m_media_nodes = nodes_by_type['m_media']  # 保留所有主流媒体
    
    # 自媒体节点筛选
    w_media_with_connections = [(node, count_same_type_connections(node)) 
                              for node in nodes_by_type['w_media']]
    w_media_with_connections.sort(key=lambda x: x[1], reverse=True)
    w_media_filtered = [node for node, conn in w_media_with_connections if conn >= min_degree]
    
    # 普通用户节点筛选
    o_people_with_connections = [(node, count_same_type_connections(node)) 
                               for node in nodes_by_type['o_people']]
    o_people_with_connections.sort(key=lambda x: x[1], reverse=True)
    o_people_filtered = [node for node, conn in o_people_with_connections if conn >= min_degree]
    
    # 采样节点
    w_media_sample_size = int(len(nodes_by_type['w_media']) * w_media_ratio)
    o_people_sample_size = int(len(nodes_by_type['o_people']) * o_people_ratio)
    
    w_media_nodes = np.random.choice(w_media_filtered, 
                                   size=min(len(w_media_filtered), w_media_sample_size),
                                   replace=False)
    o_people_nodes = np.random.choice(o_people_filtered,
                                    size=min(len(o_people_filtered), o_people_sample_size),
                                    replace=False)
    
    # 创建采样节点集合
    sampled_nodes = {node.name: node for node in [*m_media_nodes, *w_media_nodes, *o_people_nodes]}
    
    # 处理节点和边
    layer_mapping = {'m_media': 0, 'w_media': 1, 'o_people': 2}
    
    # 添加节点和边
    for node in sampled_nodes.values():
        layer_idx = layer_mapping[node.type]
        
        # 添加节点及其属性
        layers[layer_idx].add_node(node.name, 
                                 risk=node.risk,
                                 sentiment=node.sentiment,
                                 type=node.type)
        
        # 处理层内边和层间边
        for influencer in node.influencers:
            if influencer.name in sampled_nodes:
                if influencer.type == node.type:
                    # 同层边
                    layers[layer_idx].add_edge(influencer.name, node.name)
                else:
                    # 层间边
                    influencer_layer = layer_mapping[influencer.type]
                    interlayer_edges.append((influencer_layer, influencer.name,
                                          layer_idx, node.name))
    
    # 打印网络统计信息
    print("\nNetwork Statistics:")
    print("-" * 50)
    for i, layer_name in enumerate(['Mainstream Media', 'We Media', 'Public']):
        print(f"\n{layer_name} Layer:")
        print(f"Nodes: {layers[i].number_of_nodes()}")
        print(f"Edges: {layers[i].number_of_edges()}")
        if layers[i].number_of_nodes() > 0:
            print(f"Average Degree: {2 * layers[i].number_of_edges() / layers[i].number_of_nodes():.2f}")
    
    print(f"\nInterlayer Connections: {len(interlayer_edges)}")
    
    return layers, interlayer_edges


import numpy as np
from math import comb
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

def create_optimized_layout(layer, scale=1.0):
    """
    优化的布局算法，使用环形布局提高层次区分度
    """
    if len(layer) == 0:
        return {}
    
    # 计算节点重要性
    centrality = nx.betweenness_centrality(layer)
    degree = dict(layer.degree())
    
    # 结合圆形布局和spring布局
    circular_pos = nx.circular_layout(layer, scale=scale)
    spring_pos = nx.spring_layout(
        layer,
        k=30/np.sqrt(len(layer)),
        iterations=150,
        scale=scale,
        weight=None  # 不考虑边权重
    )
    
    # 混合两种布局
    positions = np.array(list(spring_pos.values()))
    circular_positions = np.array(list(circular_pos.values()))
    
    # 根据节点重要性调整布局混合比例
    mix_ratios = np.array([
        0.7 if (centrality[node] > np.mean(list(centrality.values())) or 
                degree[node] > np.mean(list(degree.values()))) 
        else 0.3 
        for node in spring_pos.keys()
    ])
    
    # 使用不同的混合比例
    positions = (mix_ratios[:, np.newaxis] * positions + 
                (1 - mix_ratios[:, np.newaxis]) * circular_positions)
    
    # 添加控制的随机扰动
    angles = 2 * np.pi * np.random.random(len(positions))
    radii = np.array([
        1.0 if (centrality[node] > np.mean(list(centrality.values())) or 
                degree[node] > np.mean(list(degree.values())))
        else 2.0
        for node in spring_pos.keys()
    ]) * np.random.random(len(positions))
    
    displacement = np.column_stack((
        radii * np.cos(angles),
        radii * np.sin(angles)
    ))
    positions += displacement
    
    return {node: pos for node, pos in zip(spring_pos.keys(), positions)}

def visualize_multilayer_network(layers, interlayer_edges):
    """
    优化的多层网络可视化，增强层次感
    """
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 55  # 设置基础字体大小
    fig = plt.figure(figsize=(28, 24))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置背景色
    ax.set_facecolor('#F5F5F5')
    fig.patch.set_facecolor('#F5F5F5')
    
    # 配色方案
    colors = {
        'risk': {
            'mainstream': {
                'R': '#FF474C',
                'NR': '#4DBBD5'
            },
            'wemedia': {
                'R': '#9B59B6',
                'NR': '#E67E22'
            }
        },
        'sentiment': {
            'H': '#00A087',
            'M': '#3C5488',
            'L': '#F39B7F'
        }
    }
    
    # 调整边的颜色为更浅的灰色
    edge_colors = {
        'internal': '#00000030',    # 更浅的灰色
        'interlayer': '#00000025'   # 更浅的灰色
    }

    # 统一使用浅灰色作为层的底色
    layer_color = '#80808015'  # 统一的浅灰色
    
    base_spacing = 2000
    layer_spacings = [
        base_spacing * 0.8,          # 第一层到第二层
        base_spacing * 1.2     # 第二层到第三层
    ]
    layout_scale = 40
    
    # 预计算布局和重要性指标
    layouts = []
    node_importance = []
    for layer in layers:
        if len(layer) == 0:
            layouts.append({})
            node_importance.append({})
            continue
            
        layout = create_optimized_layout(layer, scale=layout_scale)
        centrality = nx.betweenness_centrality(layer)
        degree = dict(layer.degree())
        
        importance = {
            node: (centrality[node] + degree[node]/max(dict(layer.degree()).values()))/2 
            for node in layer.nodes()
        }
        
        layouts.append(layout)
        node_importance.append(importance)

    # 修改层级位置的计算
    layer_positions = [0]  # 第一层的位置
    for spacing in layer_spacings:
        layer_positions.append(layer_positions[-1] - spacing)

    # 绘制圆形边框
    for i in range(3):
        z = layer_positions[i]
        
        if len(layouts[i]) > 0:
            positions = np.array(list(layouts[i].values()))
            center = np.mean(positions, axis=0)
            
            # 计算到中心点的最大距离作为半径
            distances = np.sqrt(np.sum((positions - center) ** 2, axis=1))
            radius = np.max(distances) * 1.1  # 稍微扩大一点以包含所有点
            
            # 生成圆形边框的点
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = center[0] + radius * np.cos(theta)
            circle_y = center[1] + radius * np.sin(theta)
            circle_z = np.full_like(theta, z)
            
            # 绘制圆形边框
            ax.plot(circle_x, circle_y, circle_z,
                   color='gray', alpha=0.3, linewidth=1.5)
            
            # 创建圆形填充
            circle_points = np.column_stack([circle_x, circle_y])
            ax.add_collection3d(
                art3d.Poly3DCollection(
                    [np.column_stack([circle_points, np.full(len(circle_points), z)])],
                    facecolors=layer_color,
                    alpha=0.2
                )
            )
    
    # 修改层间连接的绘制
    for src_layer, src_node, dst_layer, dst_node in interlayer_edges:
        if (src_node in layouts[src_layer] and 
            dst_node in layouts[dst_layer]):
            src_pos = layouts[src_layer][src_node]
            dst_pos = layouts[dst_layer][dst_node]
            src_z = layer_positions[src_layer]
            dst_z = layer_positions[dst_layer]
            
            edge_importance = (node_importance[src_layer].get(src_node, 0) + 
                             node_importance[dst_layer].get(dst_node, 0)) / 2
            
            points = np.array([
                [src_pos[0], src_pos[1], src_z],
                [src_pos[0], src_pos[1], (src_z + dst_z)/2],
                [dst_pos[0], dst_pos[1], dst_z]
            ])
            
            curve = bezier_curve(points, nums=20)
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                   c=edge_colors['interlayer'],
                   linewidth=0.8 + edge_importance * 1.2,  # 减小线宽
                   alpha=0.25)  # 降低透明度
    
    # 修改层内边和节点的绘制
    for i, (layer, layout, importance) in enumerate(zip(layers, layouts, node_importance)):
        z_level = layer_positions[i]
        
        # 绘制层内边
        for edge in layer.edges():
            if edge[0] in layout and edge[1] in layout:
                pos1, pos2 = layout[edge[0]], layout[edge[1]]
                edge_imp = (importance[edge[0]] + importance[edge[1]]) / 2
                ax.plot([pos1[0], pos2[0]], 
                       [pos1[1], pos2[1]],
                       [z_level, z_level],
                       c=edge_colors['internal'],
                       linewidth=0.8 + edge_imp * 1.2,  # 减小线宽
                       alpha=0.25)  # 降低透明度
        
        # 绘制节点
        for node in layer.nodes():
            if node not in layout:
                continue
                
            pos = layout[node]
            node_imp = importance[node]
            
            if layer.nodes[node]['type'] == 'o_people':
                color = colors['sentiment'][layer.nodes[node]['sentiment']]
            else:
                if i == 0:
                    color = colors['risk']['mainstream'][layer.nodes[node]['risk']]
                else:
                    color = colors['risk']['wemedia'][layer.nodes[node]['risk']]
            
            base_size = 100 + 80 * np.sqrt(layer.degree(node))
            node_size = base_size * (1 + node_imp * 2)
            
            edge_width = 1.0 + node_imp * 2 if node_imp > np.mean(list(importance.values())) else 0.8
            edge_color = 'black' if node_imp > np.mean(list(importance.values())) else 'white'
            
            ax.scatter(pos[0], pos[1], z_level,
                      c=color, s=node_size,
                      edgecolors=edge_color, linewidth=edge_width,
                      alpha=0.9)
        
    
    # 调整视角和相机距离
    # ax.view_init(elev=20, azim=45)
    # ax.dist = 6
    ax.set_axis_off()
    
    # 修改文本标签的位置和大小（使用 Times New Roman）
    layer_names = ['Mainstream Media', 'We Media', 'Public']
    for i, name in enumerate(layer_names):
        ax.text2D(0.80, 0.65 - i * 0.25, name,
                 fontsize=40,
                 ha='left',
                 va='center',
                 weight='bold',
                 family='Times New Roman',
                 transform=ax.transAxes)
        
    # # 创建图例句柄
    # legend_elements = [
    #     # 主流媒体风险状态
    #     plt.scatter([], [], c='#FF474C', marker='o', s=200, label='Mainstream Media - Risk'),
    #     plt.scatter([], [], c='#4DBBD5', marker='o', s=200, label='Mainstream Media - No Risk'),
        
    #     # 自媒体风险状态
    #     plt.scatter([], [], c='#9B59B6', marker='o', s=200, label='We Media - Risk'),
    #     plt.scatter([], [], c='#E67E22', marker='o', s=200, label='We Media - No Risk'),
        
    #     # 公众情感状态
    #     plt.scatter([], [], c='#00A087', marker='o', s=200, label='Public - High Arousal'),
    #     plt.scatter([], [], c='#3C5488', marker='o', s=200, label='Public - Middle Arousal'),
    #     plt.scatter([], [], c='#F39B7F', marker='o', s=200, label='Public - Low Arousal')
    # ]
    
    # # 添加图例，使用更紧凑的布局
    # ax.legend(handles=legend_elements,
    #          loc='upper center',
    #          bbox_to_anchor=(0.5, -0.02),  # 减小间距
    #          ncol=4,  # 增加列数使图例更紧凑
    #          fontsize=14,  # 减小字体
    #          frameon=True,
    #          fancybox=True,
    #          shadow=True,
    #          prop={'family': 'Times New Roman'},
    #          columnspacing=1.0,  # 减小列间距
    #          handletextpad=0.5,  # 减小图标和文本之间的间距
    #          bbox_transform=ax.transAxes)

    # # 调整布局，进一步减少底部空间
    # plt.subplots_adjust(left=0, right=1, bottom=0.08, top=1)  # 减小bottom值
    
    # 调整视角
    ax.view_init(elev=25, azim=45)  # 稍微调高视角
    ax.dist = 7  # 减小相机距离使图形更大
    
    # 移除所有边距
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # 保存设置
    plt.savefig('graph/multilayer_network.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.2)  # 减小边距
    plt.show()

def bezier_curve(points, nums=20):
    """
    生成贝塞尔曲线点
    """
    t = np.linspace(0, 1, nums)
    n = len(points) - 1
    curve = np.zeros((nums, 3))
    
    for i in range(n + 1):
        curve += np.outer(
            comb(n, i) * (1 - t)**(n - i) * t**i,
            points[i]
        )
    
    return curve