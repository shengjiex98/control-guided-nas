import random

import numpy as np

# 假设每个网络的初始参数量（可根据实际数据调整）
P0 = 1e7 # 默认初始参数量为 1e7

# P_funcs: 参数量函数，假设线性下降，可根据实际数据替换
def P_linear(p_r):
    return P0 * (1 - p_r)

def dP_dp_linear(p_r):
    return -P0

# 定义每种方法的参数量函数 P(p_r) 和 MDRS 函数 D(p_r)
P_funcs = {
    'network_slimming': lambda p_r: 67318787.9 - 97799814.0 * p_r,
    'structured_1_0': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    # 为简化示例，其他方法的函数暂用占位符，实际应用中需补充完整
    'structured_1_1': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    'structured_2_0': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    'structured_2_1': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    'structured_inf_0': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    'structured_inf_1': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    'structured_negative_inf_0': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    'structured_negative_inf_1': lambda p_r: 68160672.0 - 68104400.0 * p_r,
    'prune_anything': lambda p_r: 67092136.0 - 106733440.0 * p_r,
}

dP_dp_funcs = {
    'network_slimming': lambda p_r: -97799814.0,
    'structured_1_0': lambda p_r: -68104400.0,
    # 为简化示例，其他方法的导数暂用占位符
    'structured_1_1': lambda p_r: -68104400.0,
    'structured_2_0': lambda p_r: -68104400.0,
    'structured_2_1': lambda p_r: -68104400.0,
    'structured_inf_0': lambda p_r: -68104400.0,
    'structured_inf_1': lambda p_r: -68104400.0,
    'structured_negative_inf_0': lambda p_r: -68104400.0,
    'structured_negative_inf_1': lambda p_r: -68104400.0,
    'prune_anything': lambda p_r: -106733440.0,
}

# D_funcs 和 dD_dp_funcs（部分示例，完整实现见下方）
# Structured_1_0
def D_structured_1_0(p_r):
    if 0 <= p_r <= 0.235:
        a, b, c = 0.004012454245483116, 39.677022437247714, 8.93644342649431
        return a * np.exp(b * p_r) + c
    elif 0.235 < p_r <= 0.336:
        a, b, c = -5148.598116544685, 5103.677448158252, -877.7943235594163
        return a * p_r**2 + b * p_r + c
    else:
        return 236.2691420757144

def dD_dp_structured_1_0(p_r):
    if 0 <= p_r <= 0.235:
        a, b = 0.004012454245483116, 39.677022437247714
        return a * b * np.exp(b * p_r)
    elif 0.235 < p_r <= 0.336:
        a, b = -5148.598116544685, 5103.677448158252
        return 2 * a * p_r + b
    else:
        return 0

# Structured_1_1
def D_structured_1_1(p_r):
    if 0 <= p_r <= 0.093:
        a, b, c = -2147.1964531326475, -0.0024037404775716195, 2155.9400272792795
        return a * np.exp(b * p_r) + c
    elif 0.093 < p_r <= 0.156:
        a, b, c = 0.00016909115545380866, 72.26539173954315, 47.09976389170748
        return a * np.exp(b * p_r) + c
    elif 0.156 < p_r <= 0.166:
        a, b, c = 12033.627144749233, 0.06521772030534947, -12040.936856273986
        return a * np.exp(b * p_r) + c
    elif 0.166 < p_r <= 0.299:
        a, b, c = 154.41413262953319, 0.006910489592845492, 81.76330336312246
        return a * np.exp(b * p_r) + c
    else:
        return 236.2691420757144

def dD_dp_structured_1_1(p_r):
    if 0 <= p_r <= 0.093:
        a, b = -2147.1964531326475, -0.0024037404775716195
        return a * b * np.exp(b * p_r)
    elif 0.093 < p_r <= 0.156:
        a, b = 0.00016909115545380866, 72.26539173954315
        return a * b * np.exp(b * p_r)
    elif 0.156 < p_r <= 0.166:
        a, b = 12033.627144749233, 0.06521772030534947
        return a * b * np.exp(b * p_r)
    elif 0.166 < p_r <= 0.299:
        a, b = 154.41413262953319, 0.006910489592845492
        return a * b * np.exp(b * p_r)
    else:
        return 0

# Structured_2_0
def D_structured_2_0(p_r):
    if 0 <= p_r <= 0.239:
        a, b, c = 0.013012932468751284, 34.6840575389048, 8.752619407421202
        return a * np.exp(b * p_r) + c
    elif 0.239 < p_r <= 0.336:
        a, b, c = -12300.042728930948, 9232.1975362963, -1465.0052397970644
        return a * p_r**2 + b * p_r + c
    else:
        return 236.2691420757144

def dD_dp_structured_2_0(p_r):
    if 0 <= p_r <= 0.239:
        a, b = 0.013012932468751284, 34.6840575389048
        return a * b * np.exp(b * p_r)
    elif 0.239 < p_r <= 0.336:
        a, b = -12300.042728930948, 9232.1975362963
        return 2 * a * p_r + b
    else:
        return 0

# Structured_2_1
def D_structured_2_1(p_r):
    if 0 <= p_r <= 0.093:
        a, b, c = -2901.867440894803, -0.0016686443922591583, 2910.574741117134
        return a * np.exp(b * p_r) + c
    elif 0.093 < p_r <= 0.156:
        a, b, c = 0.03318144859801242, 38.72284098424792, 45.808009379492994
        return a * np.exp(b * p_r) + c
    elif 0.156 < p_r <= 0.166:
        a, b, c = 1.2173302154175103e-80, 1117.9531981393463, 115.50983684335613
        return a * np.exp(b * p_r) + c
    elif 0.166 < p_r <= 0.299:
        a, b, c = -0.008230466030191426, -77.17851679350895, 236.26913452367305
        return a * np.exp(b * p_r) + c
    else:
        return 236.2691420757144

def dD_dp_structured_2_1(p_r):
    if 0 <= p_r <= 0.093:
        a, b = -2901.867440894803, -0.0016686443922591583
        return a * b * np.exp(b * p_r)
    elif 0.093 < p_r <= 0.156:
        a, b = 0.03318144859801242, 38.72284098424792
        return a * b * np.exp(b * p_r)
    elif 0.156 < p_r <= 0.166:
        a, b = 1.2173302154175103e-80, 1117.9531981393463
        return a * b * np.exp(b * p_r)
    elif 0.166 < p_r <= 0.299:
        a, b = -0.008230466030191426, -77.17851679350895
        return a * b * np.exp(b * p_r)
    else:
        return 0

# Structured_inf_0
def D_structured_inf_0(p_r):
    if 0 <= p_r <= 0.18:
        a, b, c = 0.0007377790395652278, 62.2273852823994, 8.741462289888544
        return a * np.exp(b * p_r) + c
    elif 0.18 < p_r <= 0.265:
        a, b, c = -39673.92440246951, 19888.639426341735, -2252.816445675234
        return a * p_r**2 + b * p_r + c
    else:
        return 236.2691420757144

def dD_dp_structured_inf_0(p_r):
    if 0 <= p_r <= 0.18:
        a, b = 0.0007377790395652278, 62.2273852823994
        return a * b * np.exp(b * p_r)
    elif 0.18 < p_r <= 0.265:
        a, b = -39673.92440246951, 19888.639426341735
        return 2 * a * p_r + b
    else:
        return 0

# Structured_inf_1
def D_structured_inf_1(p_r):
    if 0 <= p_r <= 0.093:
        a, b, c = 0.026706829840875915, 38.33969377567627, 8.739718802236375
        return a * np.exp(b * p_r) + c
    elif 0.093 < p_r <= 0.156:
        a, b, c = 0.21908729726468915, 33.916467835684216, 45.29108176041522
        return a * np.exp(b * p_r) + c
    elif 0.156 < p_r <= 0.166:
        a, b, c = 23467.891133479672, 0.05175923563598978, -23519.16121406394
        return a * np.exp(b * p_r) + c
    elif 0.166 < p_r <= 0.17:
        a, b, c = 8.736550136151425e-05, 1.398956687123554, 236.2690238543778
        return a * np.exp(b * p_r) + c
    else:
        return 236.2691420757144

def dD_dp_structured_inf_1(p_r):
    if 0 <= p_r <= 0.093:
        a, b = 0.026706829840875915, 38.33969377567627
        return a * b * np.exp(b * p_r)
    elif 0.093 < p_r <= 0.156:
        a, b = 0.21908729726468915, 33.916467835684216
        return a * b * np.exp(b * p_r)
    elif 0.156 < p_r <= 0.166:
        a, b = 23467.891133479672, 0.05175923563598978
        return a * b * np.exp(b * p_r)
    elif 0.166 < p_r <= 0.17:
        a, b = 8.736550136151425e-05, 1.398956687123554
        return a * b * np.exp(b * p_r)
    else:
        return 0

# Structured_negative_inf_0
def D_structured_negative_inf_0(p_r):
    if 0 <= p_r <= 0.084:
        a, b, c = 0.17354931323059922, 73.4250457428353, 8.515924485457418
        return a * np.exp(b * p_r) + c
    elif 0.084 < p_r <= 0.099:
        a, b, c = -149089.92746151632, 31348.418223923218, -1408.9024713070198
        return a * p_r**2 + b * p_r + c
    else:
        return 236.2691420757144

def dD_dp_structured_negative_inf_0(p_r):
    if 0 <= p_r <= 0.084:
        a, b = 0.17354931323059922, 73.4250457428353
        return a * b * np.exp(b * p_r)
    elif 0.084 < p_r <= 0.099:
        a, b = -149089.92746151632, 31348.418223923218
        return 2 * a * p_r + b
    else:
        return 0

# Structured_negative_inf_1
def D_structured_negative_inf_1(p_r):
    if 0 <= p_r <= 0.066:
        a, b, c = 0.006696930744137932, 117.17481968511699, 8.969219344926977
        return a * np.exp(b * p_r) + c
    elif 0.066 < p_r <= 0.121:
        a, b, c = -88769.69925767911, 20980.929719549575, -997.7760520973361
        return a * p_r**2 + b * p_r + c
    else:
        return 236.2691420757144

def dD_dp_structured_negative_inf_1(p_r):
    if 0 <= p_r <= 0.066:
        a, b = 0.006696930744137932, 117.17481968511699
        return a * b * np.exp(b * p_r)
    elif 0.066 < p_r <= 0.121:
        a, b = -88769.69925767911, 20980.929719549575
        return 2 * a * p_r + b
    else:
        return 0

# Prune_anything
def D_prune_anything(p_r):
    if 0 <= p_r <= 0.15:
        a, b, c = 0.02739058357489841, 49.7085371263208, 8.404474269134662
        return a * np.exp(b * p_r) + c
    elif 0.15 < p_r <= 0.187:
        a, b, c = 0.027272685211738486, 41.80456263395774, 51.747562492191925
        return a * np.exp(b * p_r) + c
    elif 0.187 < p_r <= 0.213:
        a, b, c = 58245.25482443198, 0.013258077092413981, -58174.02052806215
        return a * np.exp(b * p_r) + c
    else:
        return 236.2691420757144

def dD_dp_prune_anything(p_r):
    if 0 <= p_r <= 0.15:
        a, b = 0.02739058357489841, 49.7085371263208
        return a * b * np.exp(b * p_r)
    elif 0.15 < p_r <= 0.187:
        a, b = 0.027272685211738486, 41.80456263395774
        return a * b * np.exp(b * p_r)
    elif 0.187 < p_r <= 0.213:
        a, b = 58245.25482443198, 0.013258077092413981
        return a * b * np.exp(b * p_r)
    else:
        return 0

# Network_slimming
def D_network_slimming(p_r):
    if 0 <= p_r <= 0.271:
        a, b, c = 0.021459387936957112, 33.09769903210029, 8.580391980810203
        return a * np.exp(b * p_r) + c
    elif 0.271 < p_r <= 0.299:
        a, b, c = -79653.73703541361, 46629.72886634632, -6587.09013389186
        return a * p_r**2 + b * p_r + c
    else:
        return 236.2691420757144

def dD_dp_network_slimming(p_r):
    if 0 <= p_r <= 0.271:
        a, b = 0.021459387936957112, 33.09769903210029
        return a * b * np.exp(b * p_r)
    elif 0.271 < p_r <= 0.299:
        a, b = -79653.73703541361, 46629.72886634632
        return 2 * a * p_r + b
    else:
        return 0

# 定义 D_funcs 和 dD_dp_funcs 字典
D_funcs = {
    'structured_1_0': D_structured_1_0,
    'structured_1_1': D_structured_1_1,
    'structured_2_0': D_structured_2_0,
    'structured_2_1': D_structured_2_1,
    'structured_inf_0': D_structured_inf_0,
    'structured_inf_1': D_structured_inf_1,
    'structured_negative_inf_0': D_structured_negative_inf_0,
    'structured_negative_inf_1': D_structured_negative_inf_1,
    'prune_anything': D_prune_anything,
    'network_slimming': D_network_slimming
}

dD_dp_funcs = {
    'structured_1_0': dD_dp_structured_1_0,
    'structured_1_1': dD_dp_structured_1_1,
    'structured_2_0': dD_dp_structured_2_0,
    'structured_2_1': dD_dp_structured_2_1,
    'structured_inf_0': dD_dp_structured_inf_0,
    'structured_inf_1': dD_dp_structured_inf_1,
    'structured_negative_inf_0': dD_dp_structured_negative_inf_0,
    'structured_negative_inf_1': dD_dp_structured_negative_inf_1,
    'prune_anything': dD_dp_prune_anything,
    'network_slimming': dD_dp_network_slimming
}
# 可用的剪枝方法列表
methods = list(D_funcs.keys())

# 计算 H 值：H = dD/dp_r / |dP/dp_r|
def compute_H(p_r, method):
    dD_dp = dD_dp_funcs[method](p_r)
    dP_dp = dP_dp_funcs[method](p_r)
    if dP_dp == 0:
        return np.inf
    return dD_dp / abs(dP_dp)

# 启发式搜索算法
def heuristic_search(P, N, max_iter=100, step_size=0.01):
    # 初始化：P 个网络，初始剪枝率设为 0.05，随机选择方法
    prune_rates = [0.05] * P
    prune_methods = [random.choice(methods) for _ in range(P)]
    
    for iteration in range(max_iter):
        # 计算当前总参数量和总 MDRS
        total_params = sum(P_funcs[prune_methods[i]](prune_rates[i]) for i in range(P))
        total_mdrs = sum(D_funcs[prune_methods[i]](prune_rates[i]) for i in range(P))
        
        # 如果总参数量超过约束 N，增加剪枝率
        if total_params > N:
            # 计算每个网络的 H 值
            H_values = [compute_H(prune_rates[i], prune_methods[i]) for i in range(P)]
            min_H_idx = np.argmin(H_values)
            
            # 增加 H 值最小的网络的剪枝率
            new_p_r = prune_rates[min_H_idx] + step_size
            if new_p_r <= 1.0:  # 确保剪枝率不超过 1
                prune_rates[min_H_idx] = new_p_r
        
        # 如果总参数量未超限，尝试减小剪枝率
        elif total_params < N:
            H_values = [compute_H(prune_rates[i], prune_methods[i]) for i in range(P)]
            max_H_idx = np.argmax(H_values)
            
            # 减小 H 值最大的网络的剪枝率
            new_p_r = prune_rates[max_H_idx] - step_size
           
            # 找到 MDRS 贡献最大的网络
            #mdrs_contributions = [D_funcs[prune_methods[i]](prune_rates[i]) for i in range(P)]
            #max_mdrs_idx = np.argmax(mdrs_contributions)
            
            # 减小该网络的剪枝率
            #new_p_r = prune_rates[max_mdrs_idx] - step_size
            if new_p_r >= 0.0:  # 确保剪枝率不小于 0
                prune_rates[max_H_idx] = new_p_r
        
        # 尝试切换剪枝方法
        for i in range(P):
            current_method = prune_methods[i]
            current_p_r = prune_rates[i]
            current_mdrs = D_funcs[current_method](current_p_r)
            current_params = P_funcs[current_method](current_p_r)
            
            # 随机尝试另一种方法
            new_method = random.choice(methods)
            new_mdrs = D_funcs[new_method](current_p_r)
            new_params = P_funcs[new_method](current_p_r)
            
            # 如果新方法在参数量约束下 MDRS 更低，则切换
            if new_params <= N / P and new_mdrs < current_mdrs:
                prune_methods[i] = new_method
        
        # 检查终止条件（此处简化为固定迭代次数）
        if False:
            break

    total_params = sum(P_funcs[prune_methods[i]](prune_rates[i]) for i in range(P))
    total_mdrs = sum(D_funcs[prune_methods[i]](prune_rates[i]) for i in range(P))
    
    return prune_methods, prune_rates, total_mdrs, total_params

# 主函数示例
if __name__ == "__main__":
    P = 2  # 网络数量
    N = 13 * P0  # 参数量约束（假设为初始总参数量的 2/3）
    max_iter = 10000
    step_size = 0.01
    
    methods, rates, total_mdrs, total_params = heuristic_search(P, N, max_iter, step_size)
    
    # 输出结果
    print("最终结果：")
    for i in range(P):
        print(f"网络 {i+1}: 方法 = {methods[i]}, 剪枝率 = {rates[i]:.3f}")
    print(f"总 MDRS: {total_mdrs:.3f}")
    print(f"总参数量: {total_params/1e6:.0f} M, (约束: {N/1e6:.0f} M)")
    print(f"剪枝率: {rates}")