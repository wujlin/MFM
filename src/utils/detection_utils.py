import numpy as np
from src.config import JUMP_ZSCORE_THRESHOLD

def detect_jumps_improved(r_values, x_values, abs_threshold=None):
    """
    改进的跳变检测函数 - 去除绝对阈值依赖，基于相对变化和统计显著性
    
    Args:
        r_values: 控制参数数组
        x_values: 状态变量数组
        abs_threshold: 保持接口兼容性，但不再使用（已弃用）
        
    Returns:
        jumps: 跳变大小列表
        r_jumps: 跳变位置列表
        has_significant_jump: 是否存在显著跳变
        z_score: 最大跳变的z分数
    """
    if len(r_values) != len(x_values):
        raise ValueError("r_values和x_values长度必须相同")
    
    if len(r_values) < 3:
        return [], [], False, 0.0
    
    # 计算导数（考虑步长）
    dx_dr = np.gradient(x_values, r_values)
    abs_deriv = np.abs(dx_dr)
    
    # 计算统计量
    mean_deriv = np.mean(abs_deriv)
    std_deriv = np.std(abs_deriv)
    
    if std_deriv == 0:
        return [], [], False, 0.0
    
    # 找出局部极大值作为候选跳变点
    peaks = []
    for i in range(1, len(abs_deriv)-1):
        if abs_deriv[i] > abs_deriv[i-1] and abs_deriv[i] > abs_deriv[i+1]:
            peaks.append((i, abs_deriv[i]))
    
    if not peaks:
        return [], [], False, 0.0
    
    # 按跳变大小排序
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # 提取跳变信息
    jumps = []
    r_jumps = []
    
    for idx, jump_size in peaks[:5]:  # 考虑最大的5个候选跳变
        jumps.append(jump_size)
        r_jumps.append(r_values[idx])
    
    # 计算z分数判断统计显著性
    max_jump = jumps[0] if jumps else 0
    z_score = (max_jump - mean_deriv) / std_deriv
    
    # 序参量的总体变化幅度
    x_range = np.max(x_values) - np.min(x_values)
    
    # 1. 统计显著性判断（主要标准）
    statistical_significant = z_score > 4
    
    # 2. 相对跳变大小判断（替代绝对阈值）
    # 基于序参量变化范围的相对跳变
    if x_range > 0:
        # 计算步长的平均值
        avg_step = np.mean(np.diff(r_values))
        # 将导数转换为相对于序参量范围的变化率
        relative_jump_rate = max_jump * avg_step / x_range
        # 相对跳变显著性：单步变化超过总变化的15%
        relative_significant = relative_jump_rate > 0.15
    else:
        relative_significant = False
    
    # 3. 跳变导致的实际状态变化显著性
    significant_state_change = False
    if len(jumps) > 0 and len(r_jumps) > 0:
        jump_idx = np.where(np.abs(r_values - r_jumps[0]) < 1e-10)[0]
        if len(jump_idx) > 0:
            idx = jump_idx[0]
            # 确保有足够的点来估计跳变前后的状态
            window_size = min(3, idx, len(x_values) - idx - 1)
            if window_size > 0:
                # 计算跳变前后的平均状态值
                pre_jump_avg = np.mean(x_values[max(0, idx-window_size):idx])
                post_jump_avg = np.mean(x_values[idx+1:min(len(x_values), idx+1+window_size)])
                state_change = abs(post_jump_avg - pre_jump_avg)
                
                # 状态变化相对于总变化范围的比例
                if x_range > 0:
                    relative_state_change = state_change / x_range
                    # 要求状态变化至少占总变化的10%
                    significant_state_change = relative_state_change > 0.10
    
    # 4. 导数峰值的孤立性检验
    # 真正的跳变应该在导数分布中显著突出
    derivative_isolation = False
    if len(jumps) > 1:
        # 最大跳变与第二大跳变的比值
        jump_ratio = jumps[0] / jumps[1] if jumps[1] > 0 else float('inf')
        # 如果最大跳变明显大于其他跳变，说明是孤立的峰值
        derivative_isolation = jump_ratio > 2.0
    elif len(jumps) == 1:
        # 只有一个候选跳变，检查它相对于平均导数的突出程度
        derivative_isolation = max_jump > 3 * mean_deriv
    
    # 综合判断跳变显著性
    # 必须满足统计显著性，同时满足以下至少两个条件：
    # - 相对跳变显著
    # - 状态变化显著  
    # - 导数峰值孤立
    secondary_criteria = sum([
        relative_significant,
        significant_state_change,
        derivative_isolation
    ])
    
    has_significant_jump = statistical_significant and secondary_criteria >= 2
    
    # 输出诊断信息（去除绝对阈值相关信息）
    # print(f"跳变分析详情: z_score={z_score:.2f}, max_jump={max_jump:.4f}, x_range={x_range:.4f}")
    # print(f"统计显著={statistical_significant}, 相对显著={relative_significant}, " +
    #       f"状态变化显著={significant_state_change}, 导数孤立={derivative_isolation}")
    # print(f"最终判断: 显著跳变={has_significant_jump}")
    
    return jumps, r_jumps, has_significant_jump, z_score 