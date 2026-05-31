# 事件检测 
# 注意：输入数据是（采样点数，通道数），内部处理采用（采样点数，通道数）处理，输出数据是（采样点数, 通道数）

import numpy as np
from scipy.signal import convolve

def counter(mydata):
    """
    EMG 活动段检测与计数的主函数
    
    参数
    mydata : 包含 'data' 和 'fs'
    
    返回
    count : float      活动次数
    segment_data : 包含标记后的数据和 Fs
    border : 分段标记,list，交替的 [起始, 结束, 起始, 结束, ...] 全局边界索引 (0‑based)
    """
    from preprocessing import preprocessing as pp
    
    # 1. 预处理
    emg = pp(mydata)   # 应返回 dict 含 'data' 和 'fs'
    
    # 2. TKEO
    emg_tkeo = tkeo(emg)
    
    # 3. 通道1处理
    ch1 = {'data': emg_tkeo['data'][:, 0], 'fs': emg['fs']}
    b1 = detection(ch1)
    b1 = split_long(ch1, b1)
    b1 = merge_fragments(ch1, b1)
    
    # 4. 通道2处理
    ch2 = {'data': emg_tkeo['data'][:, 1], 'fs': emg['fs']}
    b2 = detection(ch2)
    b2 = split_long(ch2, b2)
    b2 = merge_fragments(ch2, b2)
    
    # 5. 双通道协同验证
    border = validation(b1, b2, emg['fs'])
    
    # 6. 计数 (每对边界算一次活动)
    count = len(border) * 0.5
    
    # 7. 生成标记数据
    segment_data = mark(emg, border)
    
    return segment_data, count, border

def tkeo(emg):
    """
    计算 Teager—Kaiser 能量算子 (TKEO)
    ψ(n) = x(n)^2 - x(n-1)*x(n+1)

    emg : 包含 'data' 和 'fs' (采样率)

    return : tkeo_data : 包含 'data' (TKEO结果) 和 'fs'
    """
    signal = emg['data']
    rows, cols = signal.shape
    out = np.zeros_like(signal)
    
    for i in range(cols):
        x = signal[:, i]
        # 用 np.roll 实现前后移位，边界置零
        x_prev = np.roll(x, 1)
        x_prev[0] = 0.0
        x_next = np.roll(x, -1)
        x_next[-1] = 0.0
        out[:, i] = x**2 - x_prev * x_next
        
    return {'data': out, 'fs': emg['fs']}


def detection(mydata):
    """
    双阈值状态机检测激活边界
    
    mydata : dict, 包含 'data' (一维) 和 'fs'

    return : acts : 数组 (N,2)，每行 [起始, 结束] (0‑based 索引)
    """
    signal = mydata['data']
    L = len(signal)
    fs = mydata['fs']
    
    BaselinePct  = 20
    OnSigma      = 15
    OffSigma     = 5
    MinActDur    = 0.2
    MinRestDur   = 0.2
    MaxInterrupt = 0.05
    
    # 计算 RMS 包络
    win_len = 0.25
    win_samps = int(np.round(win_len * fs))
    if win_samps % 2 == 0:
        win_samps += 1                     # 强制奇数
    rect_win = np.ones(win_samps) / win_samps
    env = np.sqrt(convolve(signal**2, rect_win, mode='same'))
    
    # 估计基线噪声
    sorted_env = np.sort(env)
    n_low = int(np.round(L * BaselinePct / 100))
    low_data = sorted_env[:n_low]
    baseline = np.median(low_data)
    noise_sigma = 1.4826 * np.median(np.abs(low_data - baseline))
    
    th_on  = baseline + OnSigma  * noise_sigma
    th_off = baseline + OffSigma * noise_sigma
    
    # 时间参数 (采样点)
    min_act_samps  = int(np.round(MinActDur * fs))
    min_rest_samps = int(np.round(MinRestDur * fs))
    max_int_samps  = int(np.round(MaxInterrupt * fs))
    
    # 状态常量
    REST, PRE_ACT, ACTIVE, PRE_REST, INTERRUPT = 0, 1, 2, 3, 4
    state = REST
    acts = []               # 存储 [start, end] (0‑based)
    cand_start = 0
    cand_end   = 0
    int_cnt    = 0
    
    for n in range(L):
        val = env[n]
        if state == REST:
            if val > th_on:
                cand_start = n
                state = PRE_ACT
                
        elif state == PRE_ACT:
            if val > th_on:
                if (n - cand_start + 1) >= min_act_samps:
                    state = ACTIVE
                    int_cnt = 0
            else:
                if (n - cand_start) < min_act_samps:
                    state = REST
                else:
                    int_cnt = 1
                    state = INTERRUPT
                    
        elif state == ACTIVE:
            if val > th_off:
                int_cnt = 0
            else:
                int_cnt += 1
                if int_cnt > max_int_samps:
                    cand_end = n - int_cnt          # 回溯到最后一个高于 th_off 的位置
                    state = PRE_REST
                    int_cnt = 0
                else:
                    state = INTERRUPT
                    
        elif state == INTERRUPT:
            if val > th_off:
                state = ACTIVE
                int_cnt = 0
            else:
                int_cnt += 1
                if int_cnt > max_int_samps:
                    cand_end = n - int_cnt
                    state = PRE_REST
                    int_cnt = 0
                    
        elif state == PRE_REST:
            if val > th_off:
                if (n - cand_end) < min_rest_samps:
                    # 静息时长不足，回到动作中
                    state = ACTIVE
                    int_cnt = 0
                else:
                    # 确认前一个动作结束，开始新动作
                    acts.append([cand_start, cand_end])
                    cand_start = n
                    state = PRE_ACT
            else:
                if (n - cand_end) >= min_rest_samps:
                    acts.append([cand_start, cand_end])
                    state = REST
                    
    # 处理结尾未结束的动作
    if state in (ACTIVE, INTERRUPT, PRE_ACT):
        acts.append([cand_start, L - 1])     # 0‑based 结束索引 = L-1
    elif state == PRE_REST:
        acts.append([cand_start, cand_end])
        
    return np.array(acts, dtype=int)   # 形状 (N,2)


def split_long(tkeo, acts):
    """
    将过长的动作切分为多个较短的动作段
    
    参数
    ----
    tkeo : dict, 包含 'data' (一维 TKEO 序列) 和 'fs'
    acts : numpy 数组 (N,2)，每行 [起始, 结束] (0‑based)
    
    返回
    ----
    split_acts : numpy 数组 (M,2)，切割后的动作边界
    """
    fs = tkeo['fs']
    sig = tkeo['data']
    MaxDur      = 3.0
    MinDur      = 0.15
    MinGap      = 0.1
    SigmaStart  = 100
    SigmaEnd    = 5
    StepSigma   = 1
    BaselinePct = 20
    
    # 计算 RMS 包络
    win_len = 0.25
    win_samps = int(np.round(win_len * fs))
    if win_samps % 2 == 0:
        win_samps += 1
    rect_win = np.ones(win_samps) / win_samps
    env = np.sqrt(convolve(sig**2, rect_win, mode='same'))
    
    L = len(env)
    sorted_env = np.sort(env)
    n_low = int(np.round(L * BaselinePct / 100))
    baseline = np.median(sorted_env[:n_low])
    sigma = 1.4826 * np.median(np.abs(sorted_env[:n_low] - baseline))
    
    max_samps = int(np.round(MaxDur * fs))
    min_samps = int(np.round(MinDur * fs))
    min_gap   = int(np.round(MinGap * fs))
    
    split_acts = []
    for start, end in acts:
        dur = (end - start) / fs
        if dur < MaxDur:               # 未超过最大时长，直接保留
            split_acts.append([start, end])
            continue
        
        sub_env = env[start:end+1]     # 切片
        th_list = np.arange(SigmaStart, SigmaEnd - StepSigma, -StepSigma)
        best_segments = None
        
        for th_mult in th_list:
            th = baseline + sigma * th_mult
            above = sub_env > th
            # 寻找连续高于阈值的片段
            d_above = np.diff(np.concatenate(([0], above, [0])))
            starts = np.where(d_above == 1)[0]
            ends   = np.where(d_above == -1)[0] - 1
            
            segments = []
            for s, e in zip(starts, ends):
                if (e - s + 1) >= min_samps:
                    segments.append([s, e])
            if not segments:
                continue
            
            segments = np.array(segments)
            # 检查每个片段长度 ≤ max_samps，且片段间间隔 ≥ min_gap
            ok = True
            for seg in segments:
                if seg[1] - seg[0] + 1 > max_samps:
                    ok = False
                    break
            if ok:
                for k in range(1, len(segments)):
                    gap = segments[k, 0] - segments[k-1, 1] - 1
                    if gap < min_gap:
                        ok = False
                        break
            if ok:
                best_segments = segments
                break          # 阈值合适，停止尝试更低的阈值
        
        if best_segments is not None:
            # 将局部坐标转换为全局坐标
            for seg in best_segments:
                g_start = start + seg[0]
                g_end   = start + seg[1]
                split_acts.append([g_start, g_end])
        else:
            # 所有阈值都失败，保留原长动作
            split_acts.append([start, end])
    
    if split_acts:
        split_acts = np.array(split_acts)
        split_acts = split_acts[np.argsort(split_acts[:, 0])]   # 按起始点排序
    else:
        split_acts = np.empty((0,2), dtype=int)
    return split_acts


def merge_fragments(tkeo, acts):
    """
    合并因抖动引起的短小动作片段
    
    参数
    ----
    tkeo : dict, 包含 'data' (一维 TKEO 序列) 和 'fs'
    acts : numpy 数组 (N,2)，边界 (0‑based)
    
    返回
    ----
    clean_acts : numpy 数组 (M,2)
    """
    fs = tkeo['fs']
    sig = tkeo['data']
    MinDuration = 0.3
    MinGap      = 0.05
    UseEnergy   = True
    EnergySigma = 30
    BaselinePct = 20
    
    min_dur_samps = int(np.round(MinDuration * fs))
    min_gap_samps = int(np.round(MinGap * fs))
    
    # 估计 TKE 静息基线
    if UseEnergy:
        win_len = 0.25
        win_samps = int(np.round(win_len * fs))
        if win_samps % 2 == 0:
            win_samps += 1
        rect_win = np.ones(win_samps) / win_samps
        tkeo_env = np.sqrt(convolve(sig**2, rect_win, mode='same'))
        L = len(tkeo_env)
        sorted_tke = np.sort(tkeo_env)
        n_low = int(np.round(L * BaselinePct / 100))
        low_tke = sorted_tke[:n_low]
        base_tke = np.median(low_tke)
        noise_sigma = 1.4826 * np.median(np.abs(low_tke - base_tke))
        energy_th = base_tke + EnergySigma * noise_sigma
    else:
        tkeo_env = sig
        energy_th = -np.inf
    
    work_acts = acts.copy()
    changed = True
    max_iter = 10
    iter_cnt = 0
    
    while changed and iter_cnt < max_iter:
        changed = False
        iter_cnt += 1
        new_acts = []
        i = 0
        while i < len(work_acts):
            dur = work_acts[i, 1] - work_acts[i, 0] + 1
            if dur >= min_dur_samps:
                new_acts.append(work_acts[i])
                i += 1
                continue
            
            # 片段太短
            merged = False
            # 尝试与前一个动作合并
            if new_acts:
                gap = work_acts[i, 0] - new_acts[-1][1] - 1
                if gap <= min_gap_samps:
                    new_acts[-1][1] = max(new_acts[-1][1], work_acts[i, 1])
                    merged = True
                    changed = True
            # 如果未合并，尝试与后一个动作合并
            if not merged and i + 1 < len(work_acts):
                gap = work_acts[i+1, 0] - work_acts[i, 1] - 1
                if gap <= min_gap_samps:
                    work_acts[i+1, 0] = min(work_acts[i, 0], work_acts[i+1, 0])
                    work_acts[i+1, 1] = max(work_acts[i, 1], work_acts[i+1, 1])
                    merged = True
                    changed = True
                    i += 1   # 当前片段已被合并到下一个，跳过它
                    continue
            
            # 如果未能合并，考虑丢弃（基于能量）
            if not merged:
                if UseEnergy:
                    seg_tke = tkeo_env[work_acts[i, 0]:work_acts[i, 1]+1]
                    avg_tke = np.mean(seg_tke)
                    if avg_tke >= energy_th:
                        new_acts.append(work_acts[i])   # 能量足够，保留
                    else:
                        changed = True   # 丢弃
                # 若不使用能量，则丢弃
            i += 1
        work_acts = np.array(new_acts) if new_acts else np.empty((0,2), dtype=int)
    
    if len(work_acts) > 0:
        work_acts = work_acts[np.argsort(work_acts[:, 0])]
    return work_acts


def validation(b1, b2, fs, tolerance=0.05):
    """
    双通道协同验证，合并两个通道的动作边界
    
    参数
    ----
    b1, b2 : numpy 数组 (N,2)，两个通道的动作边界 (0‑based)
    fs      : 采样率
    tolerance : 容差 (秒)
    
    返回
    ----
    border : list，交替的 [起始, 结束, 起始, 结束, ...] 全局边界索引 (0‑based)
    """
    tol = int(np.round(tolerance * fs))
    
    i = 0
    j = 0
    merged = []
    b1 = b1.tolist() if len(b1) > 0 else []
    b2 = b2.tolist() if len(b2) > 0 else []
    
    while i < len(b1) and j < len(b2):
        s1, e1 = b1[i]
        s2, e2 = b2[j]
        
        # 检查是否重叠或在容差内
        if (s1 <= e2 + tol) and (s2 <= e1 + tol):
            merged.extend([min(s1, s2), max(e1, e2)])
            i += 1
            j += 1
        elif s1 < s2:
            i += 1     # 通道1的事件无对应，丢弃
        else:
            j += 1     # 通道2的事件无对应，丢弃
    
    # 排序并确保严格交替
    merged.sort()
    # 移除可能的重复或内嵌？此处仅排序，原 MATLAB 无去重操作
    return merged


def mark(data, markers):
    """
    根据边界标记数据，在第三列写入 1（活动）或 0（静息）
    
    参数
    ----
    data : dict, 包含 'data' (numpy 数组, 2列通道) 和 'Fs'
    markers : list of int，交替的起始/结束索引 (0‑based)
    
    返回
    ----
    marker_data : 包含 'data' (原数据加第三列标记) 和 'fs'
    """
    signal = data['data']          # (N, 2)
    fs = data['fs']
    N = signal.shape[0]
    temp = np.zeros(N, dtype=int)
    
    # 成对处理 markers
    for k in range(0, len(markers)-1, 2):
        start = markers[k]
        end = markers[k+1]
        # 确保索引在有效范围
        start = max(0, start)
        end = min(N-1, end)
        temp[start:end+1] = 1
        
    marker_data = np.column_stack((signal, temp))
    return marker_data, fs
