"""
全局配置参数文件，用于统一管理模型参数设置
"""

import numpy as np

# 求解器参数
MAX_ITER = 1000      # 最大迭代次数
CONVERGENCE_TOL = 1e-6  # 收敛容差

# 基于数值稳定性测试结果的精度配置
# 测试表明：1e-6到1e-14精度差异极小(相对误差<0.1%)，可根据计算需求选择

# 不同精度需求的预设配置
PRECISION_CONFIGS = {
    'fast': {           # 快速计算模式 - 适合大规模扫描
        'tol': 1e-6,
        'max_iter': 300,
        'description': '快速模式，精度足够且计算最快'
    },
    'standard': {       # 标准精度模式 - 日常分析
        'tol': 1e-8, 
        'max_iter': 600,
        'description': '标准精度，平衡准确性与效率'
    },
    'high': {          # 高精度模式 - 精细分析
        'tol': 1e-10,
        'max_iter': 1000,
        'description': '高精度模式，用于需要极高准确性的场合'
    },
    'ultra': {         # 超高精度模式 - 特殊研究
        'tol': 1e-12,
        'max_iter': 2000,
        'description': '超高精度，主要用于数值稳定性研究'
    }
}

# 推荐配置：基于数值稳定性测试，对大多数应用fast模式已足够
RECOMMENDED_MODE = 'fast'

# 网络参数默认值
DEFAULT_N = 2000  # 网络大小
DEFAULT_K = 20    # 平均度
DEFAULT_BETA = 1.0  # 逆温度

# 敏感性分析参数
DEFAULT_SAMPLE_SIZE = 100  # 蒙特卡洛模拟样本量
DEFAULT_TIME_STEPS = 500   # 模拟总步数

# 跳变检测参数
JUMP_ZSCORE_THRESHOLD = 4.0  # 跳变检测Z分数阈值

# 其他配置参数可以在此添加
# ... 

# 在config.py中添加
BOUNDARY_FRACTION = 0.10  # 边界排除比例，减小以捕获接近边界的峰值

# ================= 缓存路径配置 =================
# 统一的缓存路径管理，解决"屎山代码"中的路径混乱问题
CACHE_CONFIG = {
    'poisson': 'scan_results/poisson_law_cache',
    'powerlaw': 'scan_results/power_law_cache', 
    'original': 'scan_results/original_sim_cache'
}

def get_cache_dir(distribution_type='poisson'):
    """统一的缓存目录获取函数"""
    return CACHE_CONFIG.get(distribution_type, CACHE_CONFIG['poisson'])

def get_cache_subdirs(distribution_type='poisson'):
    """获取缓存子目录"""
    base_dir = get_cache_dir(distribution_type)
    return {
        'base': base_dir,
        'data': f"{base_dir}/correlation_data",
        'analysis': f"{base_dir}/analysis_results"
    }

# ================= 参数过滤配置 =================
# 统一的参数过滤逻辑，避免在多个文件中重复定义
RELEVANT_PARAMS = {
    'poisson': {'phi', 'theta', 'kappa', 'N', 'steps', 'rho'},
    'powerlaw': {'phi', 'theta', 'kappa', 'N', 'steps', 'gamma_pref', 'k_min_pref', 'max_k', 'rho'},
    'original': {'phi', 'theta', 'kappa', 'N', 'steps', 'original_like_dist', 'rho'}
}

def filter_params(params_dict, distribution_type='poisson'):
    """统一的参数过滤函数"""
    relevant = RELEVANT_PARAMS.get(distribution_type, RELEVANT_PARAMS['poisson'])
    return {k: v for k, v in params_dict.items() if k in relevant}

def params_match(params1, params2, distribution_type='poisson'):
    """检查两个参数字典是否匹配（忽略无关参数）"""
    filtered1 = filter_params(params1, distribution_type)
    filtered2 = filter_params(params2, distribution_type)
    return filtered1 == filtered2 