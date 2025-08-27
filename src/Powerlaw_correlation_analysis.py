#!/usr/bin/env python3
"""
幂律分布扫描结果统计分析
分析对称性、拟合质量等指标，并生成可视化
"""

import sys
sys.path.append('.')

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== 📁 缓存路径配置区域 ====================
# 使用统一配置管理缓存路径

from src.config import get_cache_subdirs
cache_dirs = get_cache_subdirs('powerlaw')
cache_dir = cache_dirs['base']

print(f"🔴 幂律分布分析 - 缓存目录: {cache_dir}")

# 动态设置缓存路径到analyze_symmetry模块
import src.analyze_symmetry as aps

# 重新配置模块的缓存路径
aps.CACHE_DIR = cache_dir
aps.DATA_CACHE_DIR = cache_dirs['data']
aps.ANALYSIS_CACHE_DIR = cache_dirs['analysis']

# 确保目录存在
os.makedirs(aps.DATA_CACHE_DIR, exist_ok=True)
os.makedirs(aps.ANALYSIS_CACHE_DIR, exist_ok=True)

print(f"   关联长度数据: {aps.DATA_CACHE_DIR}")
print(f"   分析结果: {aps.ANALYSIS_CACHE_DIR}")
print("=" * 60)

# 导入分析函数（在路径配置之后）
from src.analyze_symmetry import (
    analyze_power_law_symmetry,  # 直接调用绘图函数
    list_cached_results,
    load_correlation_data,  # 加载关联长度数据
    robust_pickle_load,  # 统一的pickle加载函数
    collect_all_scan_results,  # 统一的扫描结果收集函数
    detect_multiple_peaks,      # 统一的多峰检测函数
    analyze_peak_quality,       # 统一的峰值质量分析函数
    filter_valid_results,       # 统一的结果过滤函数
    analyze_results_statistics  # 统一的结果统计函数
)

# 直接使用统一的函数，不再重复定义

def robust_pickle_load(file_path):
    """
    更robust的pickle加载，强力处理numpy版本兼容性问题
    """
    import sys
    import types
    
    # 第一次尝试：直接加载
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if "numpy._core" in str(e):
            # 第二次尝试：创建numpy._core的完整mock
            try:
                import numpy
                
                # 创建_core模块作为numpy.core的别名
                if not hasattr(numpy, '_core'):
                    numpy._core = numpy.core
                
                # 注入到sys.modules中
                if 'numpy._core' not in sys.modules:
                    sys.modules['numpy._core'] = numpy.core
                
                # 确保所有必要的子模块都存在
                core_submodules = ['multiarray', 'umath', 'numeric', '_internal']
                for submod in core_submodules:
                    if hasattr(numpy.core, submod):
                        if not hasattr(numpy._core, submod):
                            setattr(numpy._core, submod, getattr(numpy.core, submod))
                        
                        # 同时注入sys.modules
                        sys_key = f'numpy._core.{submod}'
                        if sys_key not in sys.modules:
                            sys.modules[sys_key] = getattr(numpy.core, submod)
                
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
            except Exception as e2:
                # 第三次尝试：更激进的monkey patching
                try:
                    # 创建一个完整的mock _core模块
                    _core_mock = types.ModuleType('numpy._core')
                    
                    # 复制numpy.core的所有属性到_core_mock
                    for attr_name in dir(numpy.core):
                        if not attr_name.startswith('__'):
                            attr_value = getattr(numpy.core, attr_name)
                            setattr(_core_mock, attr_name, attr_value)
                    
                    # 确保numpy._core存在
                    numpy._core = _core_mock
                    sys.modules['numpy._core'] = _core_mock
                    
                    # 处理可能的子模块引用
                    for attr_name in dir(numpy.core):
                        if hasattr(numpy.core, attr_name) and not attr_name.startswith('__'):
                            attr_value = getattr(numpy.core, attr_name)
                            if hasattr(attr_value, '__module__') and 'numpy.core' in str(attr_value.__module__):
                                sys_key = f'numpy._core.{attr_name}'
                                if sys_key not in sys.modules:
                                    sys.modules[sys_key] = attr_value
                    
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                        
                except Exception as e3:
                    # 最后尝试：pickle protocol兼容性
                    try:
                        # 尝试用不同的pickle protocol
                        with open(file_path, 'rb') as f:
                            # 先读取文件内容
                            content = f.read()
                        
                        # 尝试修复内容中的模块引用
                        content_str = content.decode('latin1') if isinstance(content, bytes) else str(content)
                        if 'numpy._core' in content_str:
                            content_str = content_str.replace('numpy._core', 'numpy.core')
                            content = content_str.encode('latin1')
                        
                        # 重新加载
                        import io
                        content_io = io.BytesIO(content)
                        return pickle.load(content_io)
                        
                    except:
                        print(f"   🔧 强力修复失败: {file_path}")
                        raise e
        else:
            raise e

def collect_all_scan_results(distribution_filter='powerlaw'):
    """
    收集扫描结果 - 专门针对幂律分布优化
    包括老格式缓存和新分层缓存，使用robust加载机制
    
    参数:
    - distribution_filter: 分布类型过滤器
      - 'powerlaw': 只收集幂律分布结果 (gamma_pref!=None) [默认]
      - 'poisson': 只收集泊松分布结果 (gamma_pref=None)
      - None: 收集所有分布类型的结果
    
    返回:
    - results_list: 结果列表，每个元素包含完整的分析结果
    """
    
    if distribution_filter == 'powerlaw':
        print("🔴 收集幂律分布扫描结果...")
        print("   🔴 幂律分布关键参数: γ (幂律指数), k_min (最小截断度), k_max (最大截断度)")
        print("   📊 模型参数: φ (影响阈值), θ (感知阈值)")
    elif distribution_filter == 'poisson':
        print("🔵 收集泊松分布扫描结果...")
        print("   主要参数: κ (kappa), φ (phi), θ (theta)")
    elif distribution_filter is None:
        print("📊 收集所有分布类型的扫描结果...")
    else:
        print(f"📊 收集 {distribution_filter} 分布扫描结果...")
    print("=" * 50)
    
    results_list = []
    failed_files = []
    poisson_count = 0
    powerlaw_count = 0
    
    # === 1. 扫描老格式缓存 (直接在CACHE_DIR中) ===
    print("🔍 扫描老格式缓存...")
    old_cache_files = [f for f in os.listdir(aps.CACHE_DIR) if f.endswith('.pkl') and f.startswith('result_')]
    
    for cache_file in old_cache_files:
        cache_path = os.path.join(aps.CACHE_DIR, cache_file)
        try:
            result = robust_pickle_load(cache_path)
            
            # 识别分布类型
            network_params = result.get('network_params', {})
            gamma_pref = network_params.get('gamma_pref')
            
            is_poisson = (gamma_pref is None)
            is_powerlaw = (gamma_pref is not None)
            
            # 根据过滤条件判断是否收集此结果
            should_collect = False
            
            if distribution_filter is None:
                should_collect = True
            elif distribution_filter == 'poisson' and is_poisson:
                should_collect = True
            elif distribution_filter == 'powerlaw' and is_powerlaw:
                should_collect = True
                
            if should_collect:
                # 确保有必要的字段
                if all(key in result for key in ['phi', 'theta', 'asymmetry']):
                    result['source'] = 'old_cache'
                    result['cache_file'] = cache_file
                    result['distribution_type'] = 'poisson' if is_poisson else 'powerlaw'
                    
                    # 为幂律分布添加特有参数标识
                    if is_powerlaw:
                        result['k_min_pref'] = network_params.get('k_min_pref', 'unknown')
                        result['max_k'] = network_params.get('max_k', 'unknown')
                        result['gamma_pref'] = gamma_pref
                    
                    # 尝试从对应的correlation_data中提取xi_peak (老格式缓存也需要)
                    try:
                        phi = result['phi']
                        theta = result['theta'] 
                        kappa = result.get('kappa', 120)
                        correlation_data = load_correlation_data(
                            phi=phi, theta=theta, kappa=kappa,
                            gamma_pref=gamma_pref if is_powerlaw else None, 
                            k_min_pref=network_params.get('k_min_pref', 1) if is_powerlaw else 1,
                            max_k=network_params.get('max_k', 200) if is_powerlaw else 200,
                            r_range=(0.1, 0.9), peak_search_points=100
                        )
                        if correlation_data and 'xi_peak' in correlation_data:
                            result['xi_peak'] = correlation_data['xi_peak']
                    except:
                        pass
                    
                    results_list.append(result)
                    
                    if is_poisson:
                        poisson_count += 1
                    else:
                        powerlaw_count += 1
                    
        except Exception as e:
            failed_files.append(cache_file)
            # 只显示前3个失败的详细错误，避免刷屏
            if len(failed_files) <= 3:
                print(f"⚠️ 无法加载 {cache_file}: {e}")
    
    print(f"   老格式缓存: 泊松{poisson_count}个 + 幂律{powerlaw_count}个 = 总计{poisson_count + powerlaw_count}个")
    
    # === 2. 扫描新分层缓存 (ANALYSIS_CACHE_DIR) ===
    print("🔍 扫描新分层缓存...")
    layered_success = 0
    layered_failed = 0
    layered_poisson = 0
    layered_powerlaw = 0
    
    if os.path.exists(aps.ANALYSIS_CACHE_DIR):
        analysis_files = [f for f in os.listdir(aps.ANALYSIS_CACHE_DIR) if f.endswith('.pkl')]
        
        for analysis_file in analysis_files:
            analysis_path = os.path.join(aps.ANALYSIS_CACHE_DIR, analysis_file)
            try:
                cache_data = robust_pickle_load(analysis_path)
                
                result = cache_data.get('analysis_result', {})
                
                # 识别分布类型
                network_params = result.get('network_params', {})
                gamma_pref = network_params.get('gamma_pref')
                
                is_poisson = (gamma_pref is None)
                is_powerlaw = (gamma_pref is not None)
                
                # 根据过滤条件判断是否收集此结果
                should_collect = False
                
                if distribution_filter is None:
                    should_collect = True
                elif distribution_filter == 'poisson' and is_poisson:
                    should_collect = True
                elif distribution_filter == 'powerlaw' and is_powerlaw:
                    should_collect = True
                
                if should_collect:
                    if all(key in result for key in ['phi', 'theta', 'asymmetry']):
                        result['source'] = 'layered_cache'
                        result['cache_file'] = analysis_file
                        result['analysis_params'] = cache_data.get('analysis_parameters', {})
                        result['distribution_type'] = 'poisson' if is_poisson else 'powerlaw'
                        
                        # 为幂律分布添加特有参数标识
                        if is_powerlaw:
                            result['k_min_pref'] = network_params.get('k_min_pref', 'unknown')
                            result['max_k'] = network_params.get('max_k', 'unknown')
                            result['gamma_pref'] = gamma_pref
                        
                        # 尝试从对应的correlation_data中提取xi_peak
                        try:
                            phi = result['phi']
                            theta = result['theta'] 
                            kappa = result.get('kappa', 120)  # 幂律分布可能没有kappa
                            correlation_data = load_correlation_data(
                                phi=phi, theta=theta, kappa=kappa,
                                gamma_pref=gamma_pref, 
                                k_min_pref=network_params.get('k_min_pref', 1),
                                max_k=network_params.get('max_k', 200),
                                r_range=(0.1, 0.9), peak_search_points=100
                            )
                            if correlation_data and 'xi_peak' in correlation_data:
                                result['xi_peak'] = correlation_data['xi_peak']
                        except:
                            pass
                        
                        results_list.append(result)
                        layered_success += 1
                        
                        if is_poisson:
                            layered_poisson += 1
                        else:
                            layered_powerlaw += 1
                        
            except Exception as e:
                layered_failed += 1
                failed_files.append(analysis_file)
                # 只显示前3个失败的详细错误
                if layered_failed <= 3:
                    print(f"⚠️ 无法加载 {analysis_file}: {e}")
    
    print(f"   新分层缓存: 泊松{layered_poisson}个 + 幂律{layered_powerlaw}个 = 总计{layered_success}个")
    
    if failed_files:
        print(f"   ⚠️ 总共 {len(failed_files)} 个文件加载失败 (通常是numpy版本兼容性问题)")
        if len(failed_files) > 6:
            print(f"      显示了前6个错误，其余 {len(failed_files)-6} 个类似")
    
    # 最终统计
    total_poisson = poisson_count + layered_poisson
    total_powerlaw = powerlaw_count + layered_powerlaw
    total_all = len(results_list)
    
    print(f"\n📈 收集结果统计:")
    print(f"   🔵 泊松分布: {total_poisson} 个")
    print(f"   🔴 幂律分布: {total_powerlaw} 个")
    print(f"   ✅ 总计: {total_all} 个结果")
    
    if distribution_filter:
        filter_names = {'poisson': '泊松分布', 'powerlaw': '幂律分布'}
        filter_name = filter_names.get(distribution_filter, distribution_filter)
        print(f"   🎯 过滤器: 仅收集{filter_name}结果")
    
    return results_list

def collect_all_powerlaw_results():
    """
    主要函数：收集所有幂律分布结果
    """
    return collect_all_scan_results(distribution_filter='powerlaw')

def collect_all_poisson_results():
    """
    兼容性函数：收集所有泊松分布结果
    """
    return collect_all_scan_results(distribution_filter='poisson')

def plot_from_cache_data(case):
    """
    从缓存数据绘制专业的幂律分析图表，向analyze_power_law_symmetry看齐
    
    参数:
    - case: 包含参数的字典 {'phi', 'theta', 'kappa', 'gamma_pref', 'k_min_pref', 'max_k'}
    
    返回:
    - True: 成功从缓存绘图
    - False: 缓存数据不足，需要重新计算
    """
    
    phi = case['phi']
    theta = case['theta'] 
    kappa = int(case.get('kappa', 120))  # 幂律分布可能没有kappa或使用默认值
    gamma_pref = case.get('gamma_pref')
    k_min_pref = case.get('k_min_pref', 1)
    max_k = case.get('max_k', 200)
    
    # 尝试加载关联长度数据 (第1层缓存)
    correlation_data = load_correlation_data(
        phi=phi, theta=theta, kappa=kappa,
        gamma_pref=gamma_pref,  # 幂律分布
        k_min_pref=k_min_pref,
        max_k=max_k,
        use_original_like_dist=case.get('use_original_like_dist', False),
        r_range=(0.1, 0.9), peak_search_points=100
    )
    
    if correlation_data is None:
        return False
    
    # 提取绘图数据
    r_values = correlation_data.get('r_values')
    xi_values = correlation_data.get('xi_values') 
    r_peak = correlation_data.get('r_peak')
    xi_peak = correlation_data.get('xi_peak')
    
    if r_values is None or xi_values is None:
        return False
    
    print(f"    📂 使用缓存数据: {len(r_values)}个数据点, r_peak={r_peak:.4f}, ξ_max={xi_peak:.2f}")
    
    # 🔍 多峰检测和质量分析 (使用更敏感的阈值0.55)
    is_multi_peak, peak_info = detect_multiple_peaks(r_values, xi_values, r_peak, prominence_threshold=0.55)
    quality_score, quality_info = analyze_peak_quality(r_values, xi_values, r_peak, window_size=0.05)
    
    # 输出峰值质量信息
    if is_multi_peak:
        problem_type = peak_info.get('problem_type', 'unknown')
        n_secondary = len(peak_info.get('secondary_peaks', []))
        print(f"    ⚠️ 多峰检测: {problem_type} (总峰数: {peak_info['n_peaks']}, 次峰数: {n_secondary})")
        
        # 显示次峰详细信息
        if n_secondary > 0 and 'secondary_peak_details' in peak_info:
            secondary_details = peak_info['secondary_peak_details']
            if secondary_details:
                detail_strs = []
                for detail in secondary_details:
                    pos = detail['position']
                    height = detail['height']
                    ratio = detail['ratio_to_main']
                    detail_strs.append(f"r={pos:.3f}(ξ={height:.1f}, {ratio:.1%})")
                print(f"    📍 次峰详情: {detail_strs}")
                threshold = peak_info.get('detection_threshold', 0.6)
                print(f"    🔍 检测阈值: {threshold:.1%} (降低可检测更多次峰)")
    else:
        print(f"    ✅ 单峰检测: 无显著次峰 (总峰数: {peak_info['n_peaks']}, 类型: {peak_info.get('problem_type', 'unknown')})")
    
    print(f"    📊 峰值质量: {quality_score:.3f} (单调性: L={quality_info['left_monotonic']}, R={quality_info['right_monotonic']}, 平滑性: {peak_info.get('local_smoothness', 'unknown')})")
    
    # 专业的幂律分析参数（参考analyze_power_law_symmetry）
    delta_r = 0.15  # 分析窗口半径
    exclusion_radius = 0.015  # 排除临界点附近的半径
    n_points = 30  # 每侧拟合点数
    xi_bounds = (1, 1000)  # 关联长度过滤范围
    
    # 生成左右两侧的分析点
    r_left = np.linspace(r_peak - delta_r, r_peak - exclusion_radius, n_points)
    r_right = np.linspace(r_peak + exclusion_radius, r_peak + delta_r, n_points)
    
    # 确保r值在合理范围内
    r_left = r_left[r_left > 0.05]
    r_right = r_right[r_right < 0.95]
    
    # 从原始数据中插值得到分析点的关联长度
    from scipy.interpolate import interp1d
    
    try:
        # 创建插值函数
        interp_func = interp1d(r_values, xi_values, kind='cubic', fill_value='extrapolate')
        
        # 插值得到分析点的关联长度
        xi_left = interp_func(r_left)
        xi_right = interp_func(r_right)
        
        # 过滤有效数据
        valid_left = ~np.isnan(xi_left) & (xi_left > 0)
        valid_right = ~np.isnan(xi_right) & (xi_right > 0)
        
        r_left_valid = r_left[valid_left]
        xi_left_valid = xi_left[valid_left]
        r_right_valid = r_right[valid_right]
        xi_right_valid = xi_right[valid_right]
        
        # 计算距离临界点的距离
        dr_left = np.abs(r_left_valid - r_peak)
        dr_right = np.abs(r_right_valid - r_peak)
        
        # 幂律拟合函数
        def fit_power_law(dr, xi):
            from scipy import stats
            xi_min, xi_max = xi_bounds
            mask = (xi < xi_max) & (xi > xi_min) & (dr > 0)
            if np.sum(mask) < 3:
                return None, None, None
            
            log_dr = np.log10(dr[mask])
            log_xi = np.log10(xi[mask])
            
            valid_log_mask = np.isfinite(log_dr) & np.isfinite(log_xi)
            if np.sum(valid_log_mask) < 3:
                return None, None, None
                
            log_dr = log_dr[valid_log_mask]
            log_xi = log_xi[valid_log_mask]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_dr, log_xi)
            nu = -slope  # 临界指数
            
            return nu, r_value**2, len(log_dr)
        
        # 拟合左右两侧
        nu_left, r2_left, n_left = fit_power_law(dr_left, xi_left_valid)
        nu_right, r2_right, n_right = fit_power_law(dr_right, xi_right_valid)
        
        # 安全的格式化输出
        left_nu_str = f"{nu_left:.3f}" if nu_left is not None else "N/A"
        left_r2_str = f"{r2_left:.3f}" if r2_left is not None else "N/A"
        right_nu_str = f"{nu_right:.3f}" if nu_right is not None else "N/A"
        right_r2_str = f"{r2_right:.3f}" if r2_right is not None else "N/A"
        
        print(f"    🔬 幂律拟合: 左ν={left_nu_str}(R²={left_r2_str}), 右ν={right_nu_str}(R²={right_r2_str})")
        
    except Exception as e:
        print(f"    ⚠️ 幂律分析失败: {e}")
        nu_left = nu_right = r2_left = r2_right = None
        dr_left = dr_right = np.array([])
        xi_left_valid = xi_right_valid = np.array([])
    
    # 绘制专业的3个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 左图: 完整关联长度曲线
    axes[0].plot(r_values, xi_values, 'b-o', markersize=4, alpha=0.7, label='Correlation length')
    axes[0].axvline(r_peak, color='red', linestyle='--', alpha=0.8, label=f'r_peak={r_peak:.4f}')
    
    # 标出排除区域
    axes[0].axvspan(r_peak - exclusion_radius, r_peak + exclusion_radius, 
                    alpha=0.2, color='red', label=f'Exclusion (±{exclusion_radius:.3f})')
    
    # 标注次峰位置
    secondary_peaks = peak_info.get('secondary_peaks', [])
    if is_multi_peak and len(secondary_peaks) > 0:
        for peak_idx in secondary_peaks:
            # 找到这个峰值在数组中的位置
            if peak_idx < len(r_values):
                r_secondary = r_values[peak_idx]
                xi_secondary = xi_values[peak_idx]
                axes[0].axvline(r_secondary, color='orange', linestyle=':', alpha=0.8)
                axes[0].plot(r_secondary, xi_secondary, 'o', color='orange', markersize=8, alpha=0.8)
        axes[0].axvline(np.nan, color='orange', linestyle=':', label=f'Secondary peaks ({len(secondary_peaks)})')
    
    axes[0].set_xlabel('Removal Ratio r', fontweight='bold')
    axes[0].set_ylabel('Correlation Length ξ', fontweight='bold')
    
    # 标题包含多峰信息
    if is_multi_peak:
        title = f'Correlation Length Curve (Multi-peak ⚠️)'
    else:
        title = f'Correlation Length Curve (Single-peak ✓)'
    axes[0].set_title(title, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 中图: 线性尺度拟合分析
    if len(dr_left) > 0:
        axes[1].plot(dr_left, xi_left_valid, 'bo', label='Left side', markersize=6, alpha=0.8)
    if len(dr_right) > 0:
        axes[1].plot(dr_right, xi_right_valid, 'ro', label='Right side', markersize=6, alpha=0.8)
    
    axes[1].set_xlabel('|r - r_peak|', fontweight='bold')
    axes[1].set_ylabel('Correlation Length ξ', fontweight='bold')
    axes[1].set_title('Linear Scale Analysis', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 右图: 对数尺度幂律拟合
    if len(dr_left) > 0:
        axes[2].loglog(dr_left, xi_left_valid, 'bo', label='Left side', markersize=6, alpha=0.8)
    if len(dr_right) > 0:
        axes[2].loglog(dr_right, xi_right_valid, 'ro', label='Right side', markersize=6, alpha=0.8)
    
    # 添加拟合线
    if nu_left is not None and r2_left is not None and len(dr_left) > 0:
        dr_fit_range = np.logspace(np.log10(dr_left.min()), np.log10(dr_left.max()), 50)
        xi_fit_left = dr_fit_range**(-nu_left) * (xi_left_valid[0] / dr_left[0]**(-nu_left))
        axes[2].loglog(dr_fit_range, xi_fit_left, 'b--', alpha=0.8, linewidth=2,
                      label=f'Left: ν={nu_left:.3f} (R²={r2_left:.3f})')
    
    if nu_right is not None and r2_right is not None and len(dr_right) > 0:
        dr_fit_range = np.logspace(np.log10(dr_right.min()), np.log10(dr_right.max()), 50)
        xi_fit_right = dr_fit_range**(-nu_right) * (xi_right_valid[0] / dr_right[0]**(-nu_right))
        axes[2].loglog(dr_fit_range, xi_fit_right, 'r--', alpha=0.8, linewidth=2,
                      label=f'Right: ν={nu_right:.3f} (R²={r2_right:.3f})')
    
    axes[2].set_xlabel('|r - r_peak|', fontweight='bold')
    axes[2].set_ylabel('Correlation Length ξ', fontweight='bold')
    axes[2].set_title('Power-law Scaling', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出最终结果总结
    peak_status = "多峰⚠️" if is_multi_peak else "单峰✓"
    quality_status = "高质量✓" if quality_score > 0.6 else "低质量⚠️"
    
    # 显示幂律分布参数
    if gamma_pref is not None:
        print(f"    ✅ 幂律分布分析图表已生成 - φ={phi:.3f}, θ={theta:.3f}, γ={gamma_pref:.2f}, k∈[{k_min_pref},{max_k}]")
    else:
        print(f"    ✅ 专业分析图表已生成 - φ={phi:.3f}, θ={theta:.3f}, κ={kappa}")
    print(f"    📈 综合评估: {peak_status}, {quality_status} (质量分: {quality_score:.3f})")
    return True

def detect_multiple_peaks(r_values, xi_values, main_peak_r, prominence_threshold=0.6):
    """
    极简双峰检测 - 只检测明显的双峰结构
    
    原则：
    1. 不使用关联长度阈值
    2. 只接受恰好2个峰的情况
    3. 基于简单的局部最大值检测
    
    参数:
    - r_values: removal ratio数组
    - xi_values: 关联长度数组  
    - main_peak_r: 主峰位置
    - prominence_threshold: 未使用，保持接口兼容
    
    返回:
    - is_double_peak: 是否为双峰
    - peak_info: 峰值信息字典
    """
    import numpy as np
    
    # === 简单的局部最大值检测 ===
    peaks = []
    peak_heights = []
    
    # 寻找局部最大值：比左右邻居都高
    for i in range(1, len(xi_values) - 1):
        if xi_values[i] > xi_values[i-1] and xi_values[i] > xi_values[i+1]:
            peaks.append(i)
            peak_heights.append(xi_values[i])
    
    peaks = np.array(peaks)
    peak_heights = np.array(peak_heights)
    
    # === 严格的双峰判断 + 噪声过滤 ===
    is_double_peak = False
    
    if len(peaks) == 2:
        # 检查次峰是否为平台噪声
        peak_heights_sorted = sorted(peak_heights, reverse=True)
        main_peak_height = peak_heights_sorted[0]
        secondary_peak_height = peak_heights_sorted[1]
        
        # 过滤条件：次峰不能是平台噪声（ξ=1.0附近）
        if secondary_peak_height > 1.2:  # 次峰必须大于1.5
            is_double_peak = True
    
    # 构建简洁信息
    peak_info = {
        'n_peaks': len(peaks),
        'is_double_peak': is_double_peak,
        'peak_positions': r_values[peaks].tolist() if len(peaks) > 0 else [],
        'peak_heights': peak_heights.tolist() if len(peaks) > 0 else [],
        'method': 'simple_local_maxima_with_noise_filter',
        'filtered_reason': 'secondary_peak_too_low' if len(peaks) == 2 and not is_double_peak else None
    }
    
    return is_double_peak, peak_info

def analyze_peak_quality(r_values, xi_values, main_peak_r, window_size=0.05):
    """
    分析主峰周围的质量（平滑度、单调性）
    
    参数:
    - r_values, xi_values: 数据
    - main_peak_r: 主峰位置  
    - window_size: 分析窗口大小
    
    返回:
    - quality_score: 质量评分 (0-1, 越高越好)
    - quality_info: 详细质量信息
    """
    
    # 找到主峰周围的窗口
    main_peak_idx = np.argmin(np.abs(r_values - main_peak_r))
    
    # 定义左右窗口
    left_mask = (r_values >= main_peak_r - window_size) & (r_values < main_peak_r)
    right_mask = (r_values > main_peak_r) & (r_values <= main_peak_r + window_size)
    
    quality_info = {
        'left_monotonic': False,
        'right_monotonic': False,  
        'left_smoothness': 0.0,
        'right_smoothness': 0.0,
        'peak_sharpness': 0.0
    }
    
    quality_score = 0.0
    
    if np.sum(left_mask) > 2 and np.sum(right_mask) > 2:
        # 检查左侧单调性（应该单调递增到峰值）
        left_xi = xi_values[left_mask]
        left_r = r_values[left_mask]
        if len(left_xi) > 1:
            left_diffs = np.diff(left_xi)
            increasing_ratio = np.sum(left_diffs > 0) / len(left_diffs)
            quality_info['left_monotonic'] = increasing_ratio > 0.7
            
        # 检查右侧单调性（应该单调递减离开峰值）  
        right_xi = xi_values[right_mask]
        right_r = r_values[right_mask]
        if len(right_xi) > 1:
            right_diffs = np.diff(right_xi)
            decreasing_ratio = np.sum(right_diffs < 0) / len(right_diffs)
            quality_info['right_monotonic'] = decreasing_ratio > 0.7
            
        # 计算平滑度（二阶导数的变化）
        if len(left_xi) > 2:
            left_second_diff = np.diff(left_xi, n=2)
            quality_info['left_smoothness'] = 1.0 / (1.0 + np.std(left_second_diff))
            
        if len(right_xi) > 2:
            right_second_diff = np.diff(right_xi, n=2)
            quality_info['right_smoothness'] = 1.0 / (1.0 + np.std(right_second_diff))
        
        # 峰值尖锐度（峰值相对于两侧的突出程度）
        peak_height = xi_values[main_peak_idx]
        left_avg = np.mean(left_xi) if len(left_xi) > 0 else peak_height
        right_avg = np.mean(right_xi) if len(right_xi) > 0 else peak_height
        side_avg = (left_avg + right_avg) / 2
        if side_avg > 0:
            quality_info['peak_sharpness'] = (peak_height - side_avg) / peak_height
        
        # 综合质量评分
        quality_score = (
            0.3 * (1.0 if quality_info['left_monotonic'] else 0.0) +
            0.3 * (1.0 if quality_info['right_monotonic'] else 0.0) +
            0.2 * min(quality_info['left_smoothness'], 1.0) +
            0.2 * min(quality_info['right_smoothness'], 1.0)
        )
    
    return quality_score, quality_info

def filter_valid_results(results_list):
    """
    过滤出有效的结果，排除不合理的数据
    
    参数:
    - results_list: 原始结果列表
    
    返回:
    - filtered_list: 过滤后的结果列表
    - filter_stats: 过滤统计信息
    """
    
    print("\n🔍 数据质量筛选...")
    print("=" * 50)
    
    original_count = len(results_list)
    filtered_list = []
    filter_reasons = {
        'missing_fields': 0,
        'infinite_asymmetry': 0,
        'negative_r2': 0,
        'unreasonable_nu': 0,
        'invalid_peak': 0,
        'parameter_constraint': 0
    }
    
    for result in results_list:
        # 1. 检查必要字段
        required_fields = ['phi', 'theta', 'kappa', 'asymmetry', 'nu_avg', 'r2_left', 'r2_right']
        if not all(result.get(field) is not None for field in required_fields):
            filter_reasons['missing_fields'] += 1
            continue
        
        # 2. 检查不对称性是否为无穷大或NaN
        asymmetry = result.get('asymmetry')
        if asymmetry is None or np.isinf(asymmetry) or np.isnan(asymmetry):
            filter_reasons['infinite_asymmetry'] += 1
            continue
        
        # 3. 检查R²是否合理
        r2_left = result.get('r2_left')
        r2_right = result.get('r2_right')
        if r2_left is None or r2_right is None or r2_left < 0 or r2_right < 0:
            filter_reasons['negative_r2'] += 1
            continue
        
        # 4. 检查临界指数是否合理 (通常在0.1-2.0之间)
        nu_avg = result.get('nu_avg')
        if nu_avg is None or nu_avg <= 0 or nu_avg > 3.0:
            filter_reasons['unreasonable_nu'] += 1
            continue
        
        # 5. 检查峰值位置是否合理
        r_peak = result.get('r_peak')
        if r_peak is None or r_peak <= 0 or r_peak >= 1:
            filter_reasons['invalid_peak'] += 1
            continue
        
        # 6. 检查参数约束 (theta > phi)
        phi = result.get('phi')
        theta = result.get('theta')
        if phi is None or theta is None or theta <= phi:
            filter_reasons['parameter_constraint'] += 1
            continue
        
        # 通过所有检查
        filtered_list.append(result)
    
    filtered_count = len(filtered_list)
    removed_count = original_count - filtered_count
    
    print(f"📊 筛选结果:")
    print(f"   原始样本数: {original_count}")
    print(f"   有效样本数: {filtered_count}")
    print(f"   移除样本数: {removed_count} ({removed_count/original_count*100:.1f}%)")
    
    if removed_count > 0:
        print(f"\n❌ 移除原因统计:")
        for reason, count in filter_reasons.items():
            if count > 0:
                reason_names = {
                    'missing_fields': '缺少必要字段',
                    'infinite_asymmetry': '不对称性为无穷大/NaN',
                    'negative_r2': 'R²为负数/缺失',
                    'unreasonable_nu': '临界指数不合理',
                    'invalid_peak': '峰值位置不合理',
                    'parameter_constraint': '参数约束违反'
                }
                print(f"   {reason_names.get(reason, reason)}: {count}")
    
    return filtered_list, filter_reasons

def analyze_results_statistics(results_list):
    """
    统计分析收集到的结果
    
    参数:
    - results_list: 结果列表
    
    返回:
    - stats_df: 统计结果DataFrame
    """
    
    print("\n📈 统计分析...")
    print("=" * 50)
    
    # 先进行数据质量筛选
    filtered_results, filter_stats = filter_valid_results(results_list)
    
    # 转换为DataFrame便于分析
    data_rows = []
    
    for result in filtered_results:
        row = {
            'phi': result.get('phi'),
            'theta': result.get('theta'), 
            'asymmetry': result.get('asymmetry'),
            'nu_avg': result.get('nu_avg'),
            'nu_left': result.get('nu_left'),
            'nu_right': result.get('nu_right'),
            'r2_left': result.get('r2_left'),
            'r2_right': result.get('r2_right'),
            'r_peak': result.get('r_peak'),
            'xi_peak': result.get('xi_peak'),  # 添加关联长度峰值
            'source': result.get('source'),
            'cache_file': result.get('cache_file'),
            'distribution_type': result.get('distribution_type', 'unknown')
        }
        
        # 添加分布特有的参数
        if result.get('distribution_type') == 'poisson':
            row['kappa'] = result.get('kappa')
            row['k_min_pref'] = None
            row['max_k'] = None
            row['gamma_pref'] = None
        elif result.get('distribution_type') == 'powerlaw':
            row['kappa'] = result.get('kappa', 120)  # 幂律分布可能有固定的kappa
            row['k_min_pref'] = result.get('k_min_pref')
            row['max_k'] = result.get('max_k')
            row['gamma_pref'] = result.get('gamma_pref')
        else:
            # 兼容老数据
            row['kappa'] = result.get('kappa')
            row['k_min_pref'] = result.get('k_min_pref')
            row['max_k'] = result.get('max_k')
            row['gamma_pref'] = result.get('gamma_pref')
        
        # 计算平均R²
        if row['r2_left'] is not None and row['r2_right'] is not None:
            row['r2_avg'] = (row['r2_left'] + row['r2_right']) / 2
        else:
            row['r2_avg'] = None
            
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # 基本统计
    print(f"\n📊 有效数据统计:")
    print(f"   有效样本数: {len(df)}")
    print(f"   有效asymmetry: {df['asymmetry'].notna().sum()}")
    print(f"   有效R²: {df['r2_avg'].notna().sum()}")
    print(f"   有效关联长度峰值: {df['r_peak'].notna().sum()}")
    
    if len(df) > 0:
        # 对称性统计
        valid_asymmetry = df['asymmetry'].dropna()
        if len(valid_asymmetry) > 0:
            print(f"\n🎯 对称性统计:")
            print(f"   最小不对称性: {valid_asymmetry.min():.1%}")
            print(f"   最大不对称性: {valid_asymmetry.max():.1%}")
            print(f"   平均不对称性: {valid_asymmetry.mean():.1%}")
            print(f"   中位不对称性: {valid_asymmetry.median():.1%}")
            
            # 对称性分布
            excellent = (valid_asymmetry < 0.05).sum()
            good = ((valid_asymmetry >= 0.05) & (valid_asymmetry < 0.10)).sum()
            moderate = ((valid_asymmetry >= 0.10) & (valid_asymmetry < 0.20)).sum()
            poor = (valid_asymmetry >= 0.20).sum()
            
            print(f"   对称性分布:")
            print(f"     优秀 (<5%): {excellent} ({excellent/len(valid_asymmetry)*100:.1f}%)")
            print(f"     良好 (5%-10%): {good} ({good/len(valid_asymmetry)*100:.1f}%)")
            print(f"     中等 (10%-20%): {moderate} ({moderate/len(valid_asymmetry)*100:.1f}%)")
            print(f"     较差 (≥20%): {poor} ({poor/len(valid_asymmetry)*100:.1f}%)")
        
        # R²统计
        valid_r2 = df['r2_avg'].dropna()
        if len(valid_r2) > 0:
            print(f"\n📐 拟合质量统计:")
            print(f"   最高平均R²: {valid_r2.max():.3f}")
            print(f"   最低平均R²: {valid_r2.min():.3f}")
            print(f"   平均R²: {valid_r2.mean():.3f}")
            print(f"   中位R²: {valid_r2.median():.3f}")
            
            # R²分布
            excellent_r2 = (valid_r2 >= 0.95).sum()
            good_r2 = ((valid_r2 >= 0.90) & (valid_r2 < 0.95)).sum()
            moderate_r2 = ((valid_r2 >= 0.80) & (valid_r2 < 0.90)).sum()
            poor_r2 = (valid_r2 < 0.80).sum()
            
            print(f"   拟合质量分布:")
            print(f"     优秀 (≥0.95): {excellent_r2} ({excellent_r2/len(valid_r2)*100:.1f}%)")
            print(f"     良好 (0.90-0.95): {good_r2} ({good_r2/len(valid_r2)*100:.1f}%)")
            print(f"     中等 (0.80-0.90): {moderate_r2} ({moderate_r2/len(valid_r2)*100:.1f}%)")
            print(f"     较差 (<0.80): {poor_r2} ({poor_r2/len(valid_r2)*100:.1f}%)")
        
        # 临界指数统计
        valid_nu = df['nu_avg'].dropna()
        if len(valid_nu) > 0:
            print(f"\n⚡ 临界指数统计:")
            print(f"   最大ν: {valid_nu.max():.3f}")
            print(f"   最小ν: {valid_nu.min():.3f}")
            print(f"   平均ν: {valid_nu.mean():.3f}")
            print(f"   中位ν: {valid_nu.median():.3f}")
            
            # 与理论值比较
            mean_field_close = ((valid_nu >= 0.4) & (valid_nu <= 0.6)).sum()
            ising_3d_close = ((valid_nu >= 0.55) & (valid_nu <= 0.70)).sum()
            ising_2d_close = ((valid_nu >= 0.9) & (valid_nu <= 1.1)).sum()
            
            print(f"   理论值接近度:")
            print(f"     接近平均场 (0.4-0.6): {mean_field_close} ({mean_field_close/len(valid_nu)*100:.1f}%)")
            print(f"     接近3D Ising (0.55-0.70): {ising_3d_close} ({ising_3d_close/len(valid_nu)*100:.1f}%)")
            print(f"     接近2D Ising (0.9-1.1): {ising_2d_close} ({ising_2d_close/len(valid_nu)*100:.1f}%)")
        
        # 关联长度峰值位置统计 (r_peak)
        valid_r_peak = df['r_peak'].dropna()
        if len(valid_r_peak) > 0:
            print(f"\n📍 关联长度峰值位置统计 (r_peak):")
            print(f"   最大位置: {valid_r_peak.max():.3f}")
            print(f"   最小位置: {valid_r_peak.min():.3f}")
            print(f"   平均位置: {valid_r_peak.mean():.3f}")
            print(f"   中位位置: {valid_r_peak.median():.3f}")
            
            # 峰值位置分布
            low_r = (valid_r_peak < 0.3).sum()
            mid_r = ((valid_r_peak >= 0.3) & (valid_r_peak < 0.7)).sum()
            high_r = (valid_r_peak >= 0.7).sum()
            
            print(f"   峰值位置分布:")
            print(f"     低removal区 (<0.3): {low_r} ({low_r/len(valid_r_peak)*100:.1f}%)")
            print(f"     中removal区 (0.3-0.7): {mid_r} ({mid_r/len(valid_r_peak)*100:.1f}%)")
            print(f"     高removal区 (≥0.7): {high_r} ({high_r/len(valid_r_peak)*100:.1f}%)")
        
        # 关联长度峰值大小统计 (xi_peak)
        valid_xi_peak = df['xi_peak'].dropna()
        if len(valid_xi_peak) > 0:
            print(f"\n🏔️ 关联长度峰值大小统计 (ξ_peak):")
            print(f"   最大关联长度: {valid_xi_peak.max():.2f}")
            print(f"   最小关联长度: {valid_xi_peak.min():.2f}")
            print(f"   平均关联长度: {valid_xi_peak.mean():.2f}")
            print(f"   中位关联长度: {valid_xi_peak.median():.2f}")
            
            # 关联长度量级分布
            small_xi = (valid_xi_peak < 10).sum()
            medium_xi = ((valid_xi_peak >= 10) & (valid_xi_peak < 100)).sum()
            large_xi = (valid_xi_peak >= 100).sum()
            
            print(f"   关联长度量级分布:")
            print(f"     小尺度 (<10): {small_xi} ({small_xi/len(valid_xi_peak)*100:.1f}%)")
            print(f"     中等尺度 (10-100): {medium_xi} ({medium_xi/len(valid_xi_peak)*100:.1f}%)")
            print(f"     大尺度 (≥100): {large_xi} ({large_xi/len(valid_xi_peak)*100:.1f}%)")
        else:
            print(f"\n🏔️ 关联长度峰值大小统计: 无有效数据")
        
        # 参数范围统计 - 按分布类型分组
        print(f"\n⚙️ 参数范围:")
        
        # 基本参数（两种分布都有）
        for param in ['phi', 'theta']:
            valid_param = df[param].dropna()
            if len(valid_param) > 0:
                print(f"   {param}: [{valid_param.min():.3f}, {valid_param.max():.3f}]")
        
        # 分布特有参数统计
        poisson_df = df[df['distribution_type'] == 'poisson']
        powerlaw_df = df[df['distribution_type'] == 'powerlaw']
        
        if len(poisson_df) > 0:
            print(f"\n   🔵 泊松分布特有参数:")
            kappa_poisson = poisson_df['kappa'].dropna()
            if len(kappa_poisson) > 0:
                print(f"      κ: [{kappa_poisson.min():.0f}, {kappa_poisson.max():.0f}]")
        
        if len(powerlaw_df) > 0:
            print(f"\n   🔴 幂律分布网络拓扑参数:")
            gamma_param = powerlaw_df['gamma_pref'].dropna()
            if len(gamma_param) > 0:
                print(f"      γ (幂律指数): [{gamma_param.min():.2f}, {gamma_param.max():.2f}]")
            
            for param in ['k_min_pref', 'max_k']:
                valid_param = powerlaw_df[param].dropna()
                if len(valid_param) > 0:
                    param_names = {'k_min_pref': 'k_min (最小截断度)', 'max_k': 'k_max (最大截断度)'}
                    param_name = param_names.get(param, param)
                    print(f"      {param_name}: [{valid_param.min():.0f}, {valid_param.max():.0f}]")
    
    return df

def find_best_cases(df, top_n=5, min_xi_peak=5.0, min_r2=0.85):
    """
    找出最佳case
    
    参数:
    - df: 结果DataFrame
    - top_n: 返回前几个最佳结果
    - min_xi_peak: 最小关联长度峰值阈值，低于此值被筛除
    - min_r2: 最小拟合质量阈值，左右两侧R²都需≥此值
    
    返回:
    - best_symmetry: 对称性最好的case
    - high_quality: 高质量拟合的case (R²≥0.95)
    """
    
    print(f"\n🏆 寻找最佳案例...")
    print("=" * 50)
    
    # 先进行拟合质量前置筛选
    r2_mask = (df['r2_left'] >= min_r2) & (df['r2_right'] >= min_r2)
    quality_filtered_df = df[r2_mask].copy()
    
    # 统计总体情况
    total_cases = len(df)
    quality_cases = len(quality_filtered_df)
    valid_xi_cases = len(quality_filtered_df.dropna(subset=['xi_peak']))
    xi_valid_df = quality_filtered_df.dropna(subset=['xi_peak'])
    strong_divergence_cases = len(xi_valid_df[xi_valid_df['xi_peak'] >= min_xi_peak])
    
    print(f"📊 数据概览:")
    print(f"   总case数: {total_cases}")
    print(f"   高质量拟合 (R²≥{min_r2}): {quality_cases}")
    print(f"   有ξ_peak数据: {valid_xi_cases}")
    print(f"   强发散case (ξ≥{min_xi_peak}): {strong_divergence_cases}")
    print(f"   弱发散被筛除: {valid_xi_cases - strong_divergence_cases}")
    
    # === 1. 对称性最好的case (拟合质量 + 关联长度双重筛选) ===
    xi_filtered_df = quality_filtered_df.dropna(subset=['asymmetry', 'xi_peak'])
    xi_filtered_df = xi_filtered_df[xi_filtered_df['xi_peak'] >= min_xi_peak]
    
    if len(xi_filtered_df) > 0:
        best_symmetry = xi_filtered_df.nsmallest(top_n, 'asymmetry')
        
        print(f"\n🎯 对称性最佳 (预筛选: R²≥{min_r2}, ξ≥{min_xi_peak}) - {len(best_symmetry)} 个case:")
    else:
        # 如果强发散case不够，使用所有高质量case但给出警告
        valid_df = quality_filtered_df.dropna(subset=['asymmetry'])
        best_symmetry = valid_df.nsmallest(top_n, 'asymmetry')
        
        print(f"\n⚠️ 对称性最佳 (R²≥{min_r2}, 无强发散case) - {len(best_symmetry)} 个case:")
    
    print("-" * 60)
    print(f"{'Rank':<4} {'φ':<6} {'θ':<6} {'κ':<4} {'不对称性':<10} {'ν_avg':<7} {'R²_avg':<7} {'ξ_peak':<8}")
    print("-" * 60)
    
    for i, (idx, row) in enumerate(best_symmetry.iterrows()):
        xi_peak = row.get('xi_peak')
        xi_str = f"{xi_peak:.2f}" if xi_peak is not None else "N/A"
        marker = "⚠️" if xi_peak is not None and xi_peak < min_xi_peak else ""
        
        print(f"{i+1:<4} {row['phi']:<6.3f} {row['theta']:<6.3f} {row['kappa']:<4.0f} "
              f"{row['asymmetry']:<10.1%} {row['nu_avg']:<7.3f} "
              f"{row['r2_avg']:<7.3f} {xi_str:<8} {marker}")
    
    # === 2. 极高质量拟合的case (两侧R²都≥0.95 且关联长度≥阈值) ===
    # 在高质量拟合基础上，进一步筛选R²≥0.95的极高质量case
    ultra_high_quality_mask = (quality_filtered_df['r2_left'] >= 0.95) & (quality_filtered_df['r2_right'] >= 0.95)
    high_quality_raw = quality_filtered_df[ultra_high_quality_mask].copy()
    
    # 同样进行关联长度筛选
    high_quality = high_quality_raw.dropna(subset=['xi_peak'])
    high_quality = high_quality[high_quality['xi_peak'] >= min_xi_peak]
    
    if len(high_quality) > 0:
        # 按对称性排序
        high_quality = high_quality.sort_values('asymmetry')
        
        print(f"\n📐 极高质量拟合 (R²≥0.95, ξ≥{min_xi_peak}) 的 {len(high_quality)} 个case:")
        print("-" * 75)
        print(f"{'Rank':<4} {'φ':<6} {'θ':<6} {'κ':<4} {'不对称性':<10} {'R²_L':<6} {'R²_R':<6} {'ν_avg':<7} {'ξ_peak':<8}")
        print("-" * 75)
        
        for i, (idx, row) in enumerate(high_quality.iterrows()):
            xi_peak = row.get('xi_peak')
            xi_str = f"{xi_peak:.2f}" if xi_peak is not None else "N/A"
            
            print(f"{i+1:<4} {row['phi']:<6.3f} {row['theta']:<6.3f} {row['kappa']:<4.0f} "
                  f"{row['asymmetry']:<10.1%} {row['r2_left']:<6.3f} {row['r2_right']:<6.3f} "
                  f"{row['nu_avg']:<7.3f} {xi_str:<8}")
    else:
        # 显示统计信息
        raw_count = len(high_quality_raw)
        print(f"\n📐 极高质量拟合统计:")
        print(f"   基础筛选 (R²≥{min_r2}): {quality_cases}")
        print(f"   极高质量 (R²≥0.95): {raw_count}")
        print(f"   强发散筛选后 (ξ≥{min_xi_peak}): 0")
        print(f"   ⚠️ 所有极高质量case都是弱发散，建议调整参数范围")
    
    # === 3. 综合最佳 (对称性好且R²高，基于高质量拟合) ===
    valid_quality_df = quality_filtered_df.dropna(subset=['asymmetry', 'r2_avg'])
    if len(valid_quality_df) > 0:
        # 综合评分：归一化的对称性 + 归一化的R²质量
        asymmetry_norm = (valid_quality_df['asymmetry'] - valid_quality_df['asymmetry'].min()) / (valid_quality_df['asymmetry'].max() - valid_quality_df['asymmetry'].min())
        r2_norm = (valid_quality_df['r2_avg'] - valid_quality_df['r2_avg'].min()) / (valid_quality_df['r2_avg'].max() - valid_quality_df['r2_avg'].min())
        
        # 综合评分：对称性权重0.6，R²权重0.4 (对称性更重要)
        valid_quality_df = valid_quality_df.copy()
        valid_quality_df['composite_score'] = 0.6 * (1 - asymmetry_norm) + 0.4 * r2_norm
        
        best_composite = valid_quality_df.nlargest(min(top_n, len(valid_quality_df)), 'composite_score')
        
        print(f"\n🏅 综合最佳 (R²≥{min_r2}, 对称性0.6 + R²质量0.4) 的 {len(best_composite)} 个case:")
        print("-" * 70)
        print(f"{'Rank':<4} {'φ':<6} {'θ':<6} {'κ':<4} {'不对称性':<10} {'R²_avg':<7} {'综合分':<7}")
        print("-" * 70)
        
        for i, (idx, row) in enumerate(best_composite.iterrows()):
            asymmetry = row['asymmetry']
            r2_avg = row['r2_avg']
            score = row['composite_score']
            
            print(f"{i+1:<4} {row['phi']:<6.3f} {row['theta']:<6.3f} {row['kappa']:<4.0f} "
                  f"{asymmetry:<10.1%} {r2_avg:<7.3f} {score:<7.3f}")
    
    return best_symmetry, high_quality

def visualize_best_cases(df, max_plots=3):
    """
    可视化3种最佳case的详细分析图
    
    参数:
    - df: 完整的结果DataFrame
    - max_plots: 最多显示几个图
    """
    
    print(f"\n🎨 生成3种最佳case的可视化...")
    print("=" * 50)
    
    import matplotlib.pyplot as plt
    
    cases_to_plot = []
    
    # 1. 对称性最好的case
    if len(df) > 0:
        best_symmetry_case = df.loc[df['asymmetry'].idxmin()]
        xi_peak_value = best_symmetry_case.get('xi_peak')
        cases_to_plot.append({
            'type': '🎯 最佳对称性',
            'phi': best_symmetry_case['phi'],
            'theta': best_symmetry_case['theta'], 
            'kappa': best_symmetry_case['kappa'],
            'asymmetry': best_symmetry_case['asymmetry'],
            'r2_avg': best_symmetry_case['r2_avg'],
            'xi_peak': xi_peak_value if xi_peak_value is not None else 'N/A'
        })
    
    # 2. 拟合质量最好的case (R²最高)
    if len(df) > 0:
        best_r2_case = df.loc[df['r2_avg'].idxmax()]
        
        # 检查是否与对称性case相同
        if not (best_r2_case['phi'] == cases_to_plot[0]['phi'] and
                best_r2_case['theta'] == cases_to_plot[0]['theta'] and
                best_r2_case['kappa'] == cases_to_plot[0]['kappa']):
            
            xi_peak_value = best_r2_case.get('xi_peak')
            cases_to_plot.append({
                'type': '📐 最佳拟合质量',
                'phi': best_r2_case['phi'],
                'theta': best_r2_case['theta'],
                'kappa': best_r2_case['kappa'], 
                'asymmetry': best_r2_case['asymmetry'],
                'r2_avg': best_r2_case['r2_avg'],
                'xi_peak': xi_peak_value if xi_peak_value is not None else 'N/A'
            })
    
    # 3. 关联长度最大的case (ξ_peak最大)
    valid_xi_df = df.dropna(subset=['xi_peak'])
    if len(valid_xi_df) > 0:
        best_xi_case = valid_xi_df.loc[valid_xi_df['xi_peak'].idxmax()]
        
        # 检查是否与前面的case重复
        is_duplicate = False
        for existing_case in cases_to_plot:
            if (best_xi_case['phi'] == existing_case['phi'] and
                best_xi_case['theta'] == existing_case['theta'] and
                best_xi_case['kappa'] == existing_case['kappa']):
                is_duplicate = True
                break
        
        if not is_duplicate:
            cases_to_plot.append({
                'type': '🏔️ 最大关联长度',
                'phi': best_xi_case['phi'],
                'theta': best_xi_case['theta'],
                'kappa': best_xi_case['kappa'],
                'asymmetry': best_xi_case['asymmetry'],
                'r2_avg': best_xi_case['r2_avg'],
                'xi_peak': best_xi_case['xi_peak']
            })
    
    # 如果关联长度最大的case不存在，明确告知用户
    if len(valid_xi_df) == 0:
        print(f"⚠️ 警告：没有找到有效的关联长度数据，只能显示对称性和拟合质量最佳的case")
    
    # 限制绘图数量
    cases_to_plot = cases_to_plot[:max_plots]
    
    print(f"将展示 {len(cases_to_plot)} 个最佳case的详细分析图:")
    
    for i, case in enumerate(cases_to_plot):
        print(f"\n📊 绘制 {case['type']} case:")
        
        # 显示幂律分布的完整参数
        gamma_pref = case.get('gamma_pref')
        k_min = case.get('k_min_pref')
        k_max = case.get('max_k')
        
        if gamma_pref is not None and k_min is not None and k_max is not None:
            print(f"    参数: φ={case['phi']:.3f}, θ={case['theta']:.3f}, γ={gamma_pref:.1f}, k_min={k_min:.0f}, k_max={k_max:.0f}")
        else:
            print(f"    参数: φ={case['phi']:.3f}, θ={case['theta']:.3f}, κ={case['kappa']:.0f} (基础参数)")
        print(f"    对称性: {case['asymmetry']:.1%}")
        print(f"    拟合质量: R²={case['r2_avg']:.3f}")
        if case['xi_peak'] != 'N/A' and case['xi_peak'] is not None:
            print(f"    关联长度峰值: ξ={case['xi_peak']:.2f}")
        else:
            print(f"    关联长度峰值: 数据缺失")
        
        # 尝试从缓存数据绘图，如果没有则重新计算
        try:
            print(f"    🎨 正在生成幂律对称性分析图...")
            
            # 尝试从分层缓存加载关联长度数据
            success = plot_from_cache_data(case)
            
            if not success:
                print(f"    🔄 缓存数据不足，重新计算...")
                result = analyze_power_law_symmetry(
                    phi=case['phi'],
                    theta=case['theta'], 
                    kappa=int(case.get('kappa', 120)),
                    gamma_pref=case.get('gamma_pref'),  # 幂律分布参数
                    k_min_pref=case.get('k_min_pref', 1),
                    max_k=case.get('max_k', 200),
                    r_range=(0.1, 0.9)  # 扩展搜索范围
                )
                
                if result:
                    print(f"    ✅ 图表已生成并显示")
                else:
                    print(f"    ❌ 图表生成失败")
            
        except Exception as e:
            print(f"    ❌ 绘图错误: {e}")
    
    print(f"\n✨ 图表生成完成！共生成 {len(cases_to_plot)} 个分析图。")

def visualize_top_cases(df, sort_by='symmetry', top_n=10, max_plots=None, min_xi_peak=5.0, min_r2=0.85, 
                       filter_multi_peak=True, min_peak_quality=0.5):
    """
    可视化排序靠前的多个case
    
    参数:
    - df: 完整的结果DataFrame
    - sort_by: 排序标准 ('symmetry', 'composite', 'xi_peak')
    - top_n: 显示前几个case
    - max_plots: 最多绘制几个图 (None表示全部绘制)
    - min_xi_peak: 最小关联长度峰值阈值 (对symmetry和composite排序有效)
    - min_r2: 最小拟合质量阈值 (左右两侧R²都需≥此值)
    - filter_multi_peak: 是否过滤多峰case
    - min_peak_quality: 最小峰值质量阈值
    """
    
    print(f"\n🎨 可视化排序前{top_n}的case (按{sort_by}排序)...")
    print("=" * 80)
    
    # 所有排序都先进行拟合质量前置筛选
    r2_mask = (df['r2_left'] >= min_r2) & (df['r2_right'] >= min_r2)
    quality_filtered_df = df[r2_mask].copy()
    
    if len(quality_filtered_df) == 0:
        print(f"❌ 没有满足拟合质量要求 (R²≥{min_r2}) 的case")
        return
    
    print(f"📊 拟合质量预筛选: {len(df)} → {len(quality_filtered_df)} case (R²≥{min_r2})")
    
    # 定义基础筛选描述（所有排序方式都会用到）
    filter_desc = f"R²≥{min_r2}"
    if filter_multi_peak:
        filter_desc += ", 单峰"
    
    # 🔍 多峰检测和峰值质量筛选
    if filter_multi_peak:
        print(f"🔍 执行多峰检测和峰值质量分析...")
        
        # 检测每个case的多峰性质
        single_peak_indices = []
        multi_peak_count = 0
        low_quality_count = 0
        
        for idx, row in quality_filtered_df.iterrows():
            try:
                phi, theta = row['phi'], row['theta']
                kappa = int(row.get('kappa', 120))
                gamma_pref = row.get('gamma_pref')
                k_min_pref = row.get('k_min_pref', 1)
                max_k = row.get('max_k', 200)
                
                # 加载关联长度数据  
                correlation_data = load_correlation_data(
                    phi=phi, theta=theta, kappa=kappa,
                    gamma_pref=gamma_pref, 
                    k_min_pref=k_min_pref,
                    max_k=max_k,
                    use_original_like_dist=row.get('use_original_like_dist', False),
                    r_range=(0.1, 0.9), peak_search_points=100
                )
                
                if correlation_data:
                    r_values = correlation_data.get('r_values')
                    xi_values = correlation_data.get('xi_values')
                    r_peak = correlation_data.get('r_peak')
                    
                    if r_values is not None and xi_values is not None and r_peak is not None:
                        # 多峰检测 (使用更敏感的阈值0.55)
                        is_multi_peak, peak_info = detect_multiple_peaks(r_values, xi_values, r_peak, prominence_threshold=0.55)
                        quality_score, quality_info = analyze_peak_quality(r_values, xi_values, r_peak)
                        
                        # 筛选条件
                        if is_multi_peak:
                            multi_peak_count += 1
                        elif quality_score < min_peak_quality:
                            low_quality_count += 1
                        else:
                            single_peak_indices.append(idx)
                    else:
                        low_quality_count += 1
                else:
                    low_quality_count += 1
                    
            except Exception as e:
                low_quality_count += 1
                continue
        
        # 应用多峰过滤
        if len(single_peak_indices) > 0:
            peak_filtered_df = quality_filtered_df.loc[single_peak_indices].copy()
            print(f"📊 多峰过滤结果: {len(quality_filtered_df)} → {len(peak_filtered_df)} case")
            print(f"   ⚠️ 多峰case: {multi_peak_count}")
            print(f"   ⚠️ 低质量峰: {low_quality_count}")
            print(f"   ✅ 单峰高质量: {len(peak_filtered_df)}")
        else:
            peak_filtered_df = quality_filtered_df.copy()
            print(f"⚠️ 多峰过滤后无符合条件的case，使用原数据")
    else:
        peak_filtered_df = quality_filtered_df.copy()
        print(f"🔍 跳过多峰检测 (filter_multi_peak=False)")
    
    # 根据排序标准选择数据和排序方式
    if sort_by == 'symmetry':
        # 对称性排序：拟合质量 + 多峰过滤 + 关联长度三重筛选
        valid_df = peak_filtered_df.dropna(subset=['asymmetry', 'xi_peak'])
        xi_filtered_df = valid_df[valid_df['xi_peak'] >= min_xi_peak]
        
        if len(xi_filtered_df) >= top_n:
            sorted_df = xi_filtered_df.nsmallest(top_n, 'asymmetry')  # 越小越好
            sort_desc = f"对称性最佳 ({filter_desc}, ξ≥{min_xi_peak})"
        elif len(valid_df) >= top_n:
            # 如果强发散case不够，使用所有过滤后case但标注
            sorted_df = valid_df.nsmallest(top_n, 'asymmetry')
            sort_desc = f"对称性最佳 ({filter_desc}, 含弱发散⚠️)"
        else:
            # case总数都不够
            sorted_df = valid_df.nsmallest(len(valid_df), 'asymmetry')
            sort_desc = f"对称性最佳 ({filter_desc}, 仅{len(valid_df)}个case)"
        
    elif sort_by == 'composite':
        # 综合排序：拟合质量 + 多峰过滤 + 关联长度三重筛选
        valid_df = peak_filtered_df.dropna(subset=['asymmetry', 'r2_avg', 'xi_peak'])
        xi_filtered_df = valid_df[valid_df['xi_peak'] >= min_xi_peak]
        
        if len(xi_filtered_df) >= top_n:
            # 使用强发散case计算综合评分
            asymmetry_norm = (xi_filtered_df['asymmetry'] - xi_filtered_df['asymmetry'].min()) / (xi_filtered_df['asymmetry'].max() - xi_filtered_df['asymmetry'].min())
            r2_norm = (xi_filtered_df['r2_avg'] - xi_filtered_df['r2_avg'].min()) / (xi_filtered_df['r2_avg'].max() - xi_filtered_df['r2_avg'].min())
            
            xi_filtered_df = xi_filtered_df.copy()
            xi_filtered_df['composite_score'] = 0.6 * (1 - asymmetry_norm) + 0.4 * r2_norm
            sorted_df = xi_filtered_df.nlargest(top_n, 'composite_score')
            sort_desc = f"综合评分最佳 ({filter_desc}, ξ≥{min_xi_peak})"
        elif len(valid_df) > 0:
            # 如果强发散case不够，使用所有过滤后case但标注
            asymmetry_norm = (valid_df['asymmetry'] - valid_df['asymmetry'].min()) / (valid_df['asymmetry'].max() - valid_df['asymmetry'].min())
            r2_norm = (valid_df['r2_avg'] - valid_df['r2_avg'].min()) / (valid_df['r2_avg'].max() - valid_df['r2_avg'].min())
            
            valid_df = valid_df.copy()
            valid_df['composite_score'] = 0.6 * (1 - asymmetry_norm) + 0.4 * r2_norm
            sorted_df = valid_df.nlargest(top_n, 'composite_score')
            sort_desc = f"综合评分最佳 ({filter_desc}, 含弱发散⚠️)"
        else:
            print(f"❌ 没有足够的数据计算综合评分 (需要{filter_desc})")
            return
            
    elif sort_by == 'xi_peak':
        # 关联长度峰值排序：拟合质量 + 多峰过滤
        valid_df = peak_filtered_df.dropna(subset=['xi_peak'])
        if len(valid_df) >= top_n:
            sorted_df = valid_df.nlargest(top_n, 'xi_peak')  # 越大越好
            sort_desc = f"关联长度峰值最大 ({filter_desc})"
        else:
            sorted_df = valid_df.nlargest(len(valid_df), 'xi_peak')
            sort_desc = f"关联长度峰值最大 ({filter_desc}, 仅{len(valid_df)}个case)"
        
    else:
        if sort_by == 'quality':
            print(f"⚠️ 拟合质量排序已移除 - 现在是所有排序的前提条件 (R²≥{min_r2})")
            print(f"💡 建议使用: 'symmetry', 'xi_peak', 或 'composite'")
        else:
            print(f"❌ 未知的排序标准: {sort_by}")
            print(f"💡 可用排序: 'symmetry', 'xi_peak', 'composite'")
        return
    
    if len(sorted_df) == 0:
        print(f"❌ 没有找到有效的数据进行排序")
        return
    
    # 确定实际绘制的数量
    actual_plots = min(len(sorted_df), max_plots if max_plots else len(sorted_df))
    
    print(f"📊 {sort_desc} - 前{len(sorted_df)}个case:")
    print("-" * 115)
    print(f"{'Rank':<4} {'分布':<6} {'φ':<7} {'θ':<7} {'幂律网络参数':<18} {'不对称性':<10} {'R²_avg':<8} {'ξ_peak':<8} {'峰值类型':<8} {'描述':<20}")
    print("-" * 115)
    
    # 显示排序表格
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        asymmetry = row['asymmetry'] if not pd.isna(row['asymmetry']) else None
        r2_avg = row['r2_avg'] if not pd.isna(row['r2_avg']) else None
        xi_peak = row['xi_peak'] if not pd.isna(row['xi_peak']) else None
        
        asymmetry_str = f"{asymmetry:.1%}" if asymmetry is not None else "N/A"
        r2_str = f"{r2_avg:.3f}" if r2_avg is not None else "N/A"
        xi_str = f"{xi_peak:.2f}" if xi_peak is not None else "N/A"
        
        # 获取分布类型和网络参数信息
        distribution_type = row.get('distribution_type', 'unknown')
        dist_symbol = "🔵" if distribution_type == 'poisson' else "🔴" if distribution_type == 'powerlaw' else "❓"
        
        # 网络参数显示
        if distribution_type == 'poisson':
            kappa = row.get('kappa', 'N/A')
            network_params = f"κ={kappa:.0f}" if kappa != 'N/A' else "κ=N/A"
        elif distribution_type == 'powerlaw':
            gamma_pref = row.get('gamma_pref', 'N/A')
            k_min = row.get('k_min_pref', 'N/A')
            k_max = row.get('max_k', 'N/A')
            if gamma_pref != 'N/A' and k_min != 'N/A' and k_max != 'N/A':
                network_params = f"γ={gamma_pref:.1f},k_min={k_min:.0f},k_max={k_max:.0f}"
            else:
                network_params = "幂律参数缺失"
        else:
            network_params = "未知"
        
        # 检测峰值类型
        peak_type = "未知"
        try:
            phi, theta = row['phi'], row['theta']
            kappa = row.get('kappa', 120)
            gamma_pref = row.get('gamma_pref') if distribution_type == 'powerlaw' else None
            
            correlation_data = load_correlation_data(
                phi=phi, theta=theta, kappa=kappa,
                gamma_pref=gamma_pref, 
                k_min_pref=row.get('k_min_pref', 1) if distribution_type == 'powerlaw' else 1,
                max_k=row.get('max_k', 200) if distribution_type == 'powerlaw' else 200,
                r_range=(0.1, 0.9), peak_search_points=100
            )
            if correlation_data:
                r_values = correlation_data.get('r_values')
                xi_values = correlation_data.get('xi_values')
                r_peak = correlation_data.get('r_peak')
                if r_values is not None and xi_values is not None and r_peak is not None:
                    is_multi_peak, peak_info = detect_multiple_peaks(r_values, xi_values, r_peak)
                    peak_type = "多峰⚠️" if is_multi_peak else "单峰✓"
        except:
            pass
        
        # 添加描述标签
        desc_parts = []
        if asymmetry is not None and asymmetry < 0.05:
            desc_parts.append("优秀对称")
        if r2_avg is not None and r2_avg >= 0.95:
            desc_parts.append("高质量拟合")
        if xi_peak is not None and xi_peak > 10:
            desc_parts.append("强发散")
        elif xi_peak is not None and xi_peak < 3:
            desc_parts.append("弱发散⚠️")
            
        desc = ", ".join(desc_parts) if desc_parts else "普通"
        
        marker = "🏆" if i < 3 else f"{i+1:2d}"
        print(f"{marker:<4} {dist_symbol:<6} {row['phi']:<7.3f} {row['theta']:<7.3f} {network_params:<18} "
              f"{asymmetry_str:<10} {r2_str:<8} {xi_str:<8} {peak_type:<8} {desc:<20}")
    
    # 开始绘制图表
    print(f"\n🎨 开始绘制前{actual_plots}个case的详细分析图...")
    
    cases_plotted = 0
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        if cases_plotted >= actual_plots:
            break
            
        print(f"\n📊 绘制第{i+1}名 case:")
        
        # 显示幂律分布的完整参数
        gamma_pref = row.get('gamma_pref', 'N/A')
        k_min = row.get('k_min_pref', 'N/A')
        k_max = row.get('max_k', 'N/A')
        
        if gamma_pref != 'N/A' and k_min != 'N/A' and k_max != 'N/A':
            print(f"    参数: φ={row['phi']:.3f}, θ={row['theta']:.3f}, γ={gamma_pref:.1f}, k_min={k_min:.0f}, k_max={k_max:.0f}")
        else:
            print(f"    参数: φ={row['phi']:.3f}, θ={row['theta']:.3f}, κ={row['kappa']:.0f} (回退到基础参数)")
        
        asymmetry = row['asymmetry'] if not pd.isna(row['asymmetry']) else None
        r2_avg = row['r2_avg'] if not pd.isna(row['r2_avg']) else None
        xi_peak = row['xi_peak'] if not pd.isna(row['xi_peak']) else None
        
        if asymmetry is not None:
            print(f"    对称性: {asymmetry:.1%}")
        if r2_avg is not None:
            print(f"    拟合质量: R²={r2_avg:.3f}")
        if xi_peak is not None:
            print(f"    关联长度峰值: ξ={xi_peak:.2f}")
        
        # 构造case字典用于绘图 - 包含完整的幂律分布参数
        case = {
            'phi': row['phi'],
            'theta': row['theta'],
            'kappa': row['kappa'],
            'gamma_pref': row.get('gamma_pref'),  # 幂律指数
            'k_min_pref': row.get('k_min_pref', 1),  # 最小度数
            'max_k': row.get('max_k', 200),  # 最大度数
            'use_original_like_dist': row.get('use_original_like_dist', False),  # 是否使用原始分布
            'asymmetry': asymmetry,
            'r2_avg': r2_avg,
            'xi_peak': xi_peak
        }
        
        try:
            print(f"    🎨 正在生成幂律对称性分析图...")
            
            # 尝试从分层缓存加载关联长度数据
            success = plot_from_cache_data(case)
            
            if not success:
                print(f"    🔄 缓存数据不足，重新计算...")
                result = analyze_power_law_symmetry(
                    phi=case['phi'],
                    theta=case['theta'], 
                    kappa=int(case.get('kappa', 120)),
                    gamma_pref=case.get('gamma_pref'),  # 幂律分布参数
                    k_min_pref=case.get('k_min_pref', 1),
                    max_k=case.get('max_k', 200),
                    use_original_like_dist=case.get('use_original_like_dist', False),
                    r_range=(0.1, 0.9)
                )
                
                if result:
                    print(f"    ✅ 图表已生成并显示")
                else:
                    print(f"    ❌ 图表生成失败")
            
            cases_plotted += 1
            
        except Exception as e:
            print(f"    ❌ 绘图错误: {e}")
    
    print(f"\n✨ 图表生成完成！共生成 {cases_plotted} 个分析图。")

def show_detailed_case_info(df, case_type="best"):
    """
    展示详细的case信息
    
    参数:
    - df: 结果DataFrame
    - case_type: 展示类型 ("best", "high_quality", "comprehensive")
    """
    
    print(f"\n📋 详细案例信息 ({case_type}):")
    print("=" * 80)
    
    if case_type == "best" and len(df) > 0:
        # 显示对称性最好的前3个
        best_cases = df.nsmallest(3, 'asymmetry')
        
        for i, (idx, case) in enumerate(best_cases.iterrows()):
            print(f"\n🏆 排名 #{i+1} - 最佳对称性:")
            
            # 显示幂律分布的完整参数
            gamma_pref = case.get('gamma_pref')
            k_min = case.get('k_min_pref')
            k_max = case.get('max_k')
            
            if gamma_pref is not None and k_min is not None and k_max is not None:
                print(f"   参数: φ={case['phi']:.3f}, θ={case['theta']:.3f}, γ={gamma_pref:.1f}, k_min={k_min:.0f}, k_max={k_max:.0f}")
            else:
                print(f"   参数: φ={case['phi']:.3f}, θ={case['theta']:.3f}, κ={case['kappa']:.0f} (基础参数)")
            print(f"   对称性: {case['asymmetry']:.1%}")
            print(f"   临界指数: ν_avg={case['nu_avg']:.3f} (左:{case['nu_left']:.3f}, 右:{case['nu_right']:.3f})")
            print(f"   拟合质量: R²_avg={case['r2_avg']:.3f} (左:{case['r2_left']:.3f}, 右:{case['r2_right']:.3f})")
            print(f"   关联长度峰值: r_peak={case['r_peak']:.4f}")
            print(f"   数据来源: {case['source']}")
    
    elif case_type == "high_quality":
        # 显示R²≥0.95的case
        high_qual_mask = (df['r2_left'] >= 0.95) & (df['r2_right'] >= 0.95)
        high_qual_cases = df[high_qual_mask].sort_values('asymmetry')
        
        if len(high_qual_cases) > 0:
            print(f"\n📐 高质量拟合案例 (R²≥0.95):")
            for i, (idx, case) in enumerate(high_qual_cases.iterrows()):
                print(f"\n   案例 #{i+1}:")
                
                # 显示幂律分布的完整参数
                gamma_pref = case.get('gamma_pref')
                k_min = case.get('k_min_pref')
                k_max = case.get('max_k')
                
                if gamma_pref is not None and k_min is not None and k_max is not None:
                    print(f"     参数: φ={case['phi']:.3f}, θ={case['theta']:.3f}, γ={gamma_pref:.1f}, k_min={k_min:.0f}, k_max={k_max:.0f}")
                else:
                    print(f"     参数: φ={case['phi']:.3f}, θ={case['theta']:.3f}, κ={case['kappa']:.0f} (基础参数)")
                print(f"     对称性: {case['asymmetry']:.1%}")
                print(f"     R²: 左={case['r2_left']:.3f}, 右={case['r2_right']:.3f}")
                print(f"     ν: {case['nu_avg']:.3f}")
        else:
            print(f"\n📐 无高质量拟合案例 (R²≥0.95)")



if __name__ == "__main__":
    
    # 🚀 智能分布类型识别
    # 根据缓存目录名称建议分析类型
    cache_dir_lower = cache_dir.lower()
    suggested_filter = 'powerlaw'  # 默认为幂律分布
    
    if 'powerlaw' in cache_dir_lower or 'power_law' in cache_dir_lower:
        suggested_filter = 'powerlaw'
        print(f"🔍 根据缓存目录名称，分析：🔴 幂律分布")
    elif 'poisson' in cache_dir_lower:
        suggested_filter = 'poisson'
        print(f"🔍 根据缓存目录名称，切换分析：🔵 泊松分布")
    else:
        # 默认情况下分析幂律分布
        suggested_filter = 'powerlaw'
        print(f"🔍 默认分析类型：🔴 幂律分布 (如需泊松分布，请修改缓存目录名)")
    
    print("=" * 60)
    
    # 1. 收集所有结果
    results_list = collect_all_scan_results(distribution_filter=suggested_filter)
    
    if len(results_list) == 0:
        if suggested_filter == 'powerlaw':
            print(f"❌ 没有找到任何幂律分布扫描结果")
            print(f"💡 提示：确保缓存目录中有 gamma_pref ≠ None 的数据文件")
        elif suggested_filter == 'poisson':
            print(f"❌ 没有找到任何泊松分布扫描结果")
            print(f"💡 提示：确保缓存目录中有 gamma_pref = None 的数据文件")
        else:
            print("❌ 没有找到任何扫描结果")
        exit()
    
    # 2. 统计分析
    df = analyze_results_statistics(results_list)
    
    if len(df) == 0:
        print("❌ 过滤后没有有效的结果数据")
        exit()
    
    # 3. 找出最佳case (筛选：R²≥0.85 + 关联长度≥5.0)
    best_symmetry, high_quality = find_best_cases(df, top_n=5, min_xi_peak=5.0, min_r2=0.85)
    
    # 4. 展示详细案例信息
    show_detailed_case_info(df, "best")
    show_detailed_case_info(df, "high_quality")
    
    # 5. 可视化排序靠前的更多case
    print("\n" + "="*80)
    print("🔍 详细可视化分析 - 多个排序靠前的case")
    print("="*80)
    
    # 方式1: 按对称性排序 (预筛选: R²≥0.85 + 单峰 + ξ≥5.0)
    visualize_top_cases(df, sort_by='symmetry', top_n=10, max_plots=8, 
                       min_xi_peak=5.0, min_r2=0.85, filter_multi_peak=True, min_peak_quality=0.5)
    
    # 方式2: 按关联长度峰值排序 (预筛选: R²≥0.85 + 单峰)
    visualize_top_cases(df, sort_by='xi_peak', top_n=8, max_plots=8, 
                       min_r2=0.85, filter_multi_peak=True, min_peak_quality=0.5)
    

