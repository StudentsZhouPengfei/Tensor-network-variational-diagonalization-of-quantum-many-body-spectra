# -*- coding: utf-8 -*-
"""
tn_utils.py - 张量网络初始化、幺正化、评估与高阶前向演化控制器功能库
"""

import torch as tc
import numpy as np
import matplotlib.pyplot as plt
from . import class_evolve_TNO_cut_dims as ev


def fprint(content, file='./record.log', print_screen=True, append=True):
    way = 'ab' if append else 'wb'
    with open(file, way, buffering=0) as log:
        log.write((content + '\n').encode(encoding='utf-8'))
    if print_screen:
        print(content)


def inner_with_differ_2mpo(tensor_up, tensor_down):
    """严格保持原脚本计算两个 MPO 求和表示的 trace 内积收缩路径"""
    tv = tc.einsum('astb,astd->bd', tensor_up[0], tc.conj(tensor_down[0]))
    for gt in range(1, len(tensor_up)):
        if gt < (len(tensor_up) - 1):
            tv = tc.einsum('ac,astb,cstd->bd', tv, tensor_up[gt], tc.conj(tensor_down[gt]))
        else:
            tv = tc.einsum('ac,astb,cstb->', tv, tensor_up[gt], tc.conj(tensor_down[gt]))
    return tv


def mps_indot(tar_tensor):
    """严格保持原脚本对矩阵乘积态进行归一化的收缩路径"""
    tv = tc.einsum('asb,asd->bd', tar_tensor[0], tc.conj(tar_tensor[0]))
    for gt in range(1, len(tar_tensor)):
        if gt < (len(tar_tensor) - 1):
            tv = tc.einsum('ac,asb,csd->bd', tv, tar_tensor[gt], tc.conj(tar_tensor[gt]))
        else:
            tv = tc.einsum('ac,asb,csb->', tv, tar_tensor[gt], tc.conj(tar_tensor[gt]))
    return tv


def construct_physical_layer_tensors(num_tensors, device, dtype=tc.complex128):
    """构建物理层张量元，严格保持原有 vol = 1e-1 的随机微扰机制"""
    physical_tensors_list = []
    vol = 1e-1
    for n in range(num_tensors):
        vol_gate_8 = tc.mul(tc.rand((4, 4), device=device, dtype=dtype), vol)
        ele_tensor = tc.add(tc.eye(4, device=device, dtype=dtype), vol_gate_8).reshape(2, 2, 2, 2)
        ele_tensor.requires_grad = True
        physical_tensors_list.append(ele_tensor)
    return physical_tensors_list


def construct_entanglement_layer_tensors(num_tensors, device, dtype=tc.complex128):
    """构建纠缠层张量元，严格保持原有 vol = 1e-3 的近单位阵微扰机制"""
    vol = 1e-3
    entanglement_tensors_list = []
    for n in range(num_tensors):
        vol_gate_8 = tc.mul(tc.rand((4, 4), device=device, dtype=dtype), vol)
        ele_tensor = tc.add(tc.eye(4, device=device, dtype=dtype), vol_gate_8).reshape(2, 2, 2, 2)
        ele_tensor.requires_grad = True
        entanglement_tensors_list.append(ele_tensor)
    return entanglement_tensors_list


def construct_eigen_spectrum_mps_strict(num_mps, mdims, device, dtype=tc.float64):
    """自适应阶梯式严格 MPS 构建器，安全处理首尾边界 virtual_bond=1 约束"""
    p_dims = 2
    eigen_mps_list = []

    # 节点 0
    a0 = tc.randn(1, p_dims, mdims, dtype=dtype, device=device)
    a0.requires_grad = True
    eigen_mps_list.append(a0)

    # 体区节点
    for n in range(1, num_mps - 1):
        an = tc.randn(mdims, p_dims, mdims, dtype=dtype, device=device)
        an.requires_grad = True
        eigen_mps_list.append(an)

    # 末尾节点
    a_end = tc.randn(mdims, p_dims, 1, dtype=dtype, device=device)
    a_end.requires_grad = True
    eigen_mps_list.append(a_end)
    return eigen_mps_list


def decomposition_4_bond_gate(add_tensor):
    """物理层极分解幺正化，完整保留 1e-12 实虚部噪声与虚部净化 Hook"""
    svd_n = add_tensor.reshape(add_tensor.shape[0] * add_tensor.shape[1], -1)

    noise_real = tc.normal(0, 1e-12, size=svd_n.size(), device=svd_n.device, dtype=tc.float64)
    noise_imag = tc.normal(0, 1e-12, size=svd_n.size(), device=svd_n.device, dtype=tc.float64)
    noise = tc.complex(noise_real, noise_imag)
    matrix_with_noise = svd_n + noise

    u, lm, vh = tc.linalg.svd(matrix_with_noise, full_matrices=False)

    if u.requires_grad:
        u_det = u.detach()
        u.register_hook(lambda g: g - 1j * (u_det @ tc.diag(tc.imag(tc.diagonal(u_det.conj().t() @ g))).to(g.dtype)))
    if vh.requires_grad:
        vh_det = vh.detach()
        vh.register_hook(lambda g: g - 1j * (tc.diag(tc.imag(tc.diagonal(g @ vh_det.conj().t()))).to(g.dtype) @ vh_det))

    or_tensor = u.mm(vh).reshape(add_tensor.shape)
    return or_tensor


def finite_layer_orthogonal_unitary(evolve_list):
    """对线路列表内的所有张量执行幺正极分解投影"""
    sys_double_list = []
    for n in range(len(evolve_list)):
        sys_double_list.append(decomposition_4_bond_gate(evolve_list[n]))
    return sys_double_list


# =====================================================================
#  【严格等价移植】：单层有梯度与无梯度的级联控制函数（原汁原味移入工具库）
# =====================================================================

def evolve_single_ladder_entanglement_layer_tensors(h_mpo, single_tensors, cut_dims):
    """有梯度的单层演化收缩控制"""
    evo = ev.Eigen_orthogonal(h_mpo)
    for n in range(0, len(single_tensors)):
        evo.evolve_bra_ket_4_bond_gate(n, single_tensors[n], cut_dims, 'right', True, False)
    return evo.tensor


def evolve_single_nograd_ladder_entanglement_layer_tensors(h_mpo, single_tensors, cut_dims, direction):
    """无梯度的单层演化收缩控制（处理特定边界与方向切换）"""
    evo = ev.Eigen_orthogonal(h_mpo, eps=1e-12)
    if direction == 0:
        for n in range(0, len(single_tensors)):
            evo.evolve_bra_ket_4_bond_gate_dont_cut(n, single_tensors[n], cut_dims, 'right', True, False)
    else:
        # 严格保持原脚本 len(single_tensors) - 3 到 2 的倒序收缩路径
        for z in range(len(single_tensors) - 3, 1, -1):
            evo.evolve_bra_ket_4_bond_gate_dont_cut(z, single_tensors[z], cut_dims, 'left', True, False)
    # 兼容原脚本中的 clone_list 操作
    new_list = []
    for n in range(len(evo.tensor)):
        new_list.append(evo.tensor[n].clone() * 1.0)
    return new_list


# =====================================================================
#  【完全体级联控制器】：完美兼容多层前向收缩链路
# =====================================================================

def forward_circuit_evolution(h_mpo, physical_tensors, entanglement_layers_list, cut_dim):
    """
    【高阶演化控制器】：一键托管全线路前向极分解幺正化与哈密顿量通道收缩演化。
    内部严格通过你原本的单层演化控制函数进行链式推进。
    """
    # 1. 对基础物理层进行幺正化投影
    physical_unitary = finite_layer_orthogonal_unitary(physical_tensors)

    # 2. 先把哈密顿量通过第一层（物理层）进行收缩演化
    mpo_tensors = evolve_single_ladder_entanglement_layer_tensors(h_mpo, physical_unitary, cut_dim)

    # 3. 如果存在更深的网络层（NL >= 1），依次通过单层演化控制器向下游推进
    if len(entanglement_layers_list) > 0:
        for e_layer in entanglement_layers_list:
            # 极分解幺正化当前纠缠层
            entangle_unitary = finite_layer_orthogonal_unitary(e_layer)
            # 链式递推哈密顿量
            mpo_tensors = evolve_single_ladder_entanglement_layer_tensors(mpo_tensors, entangle_unitary, cut_dim)

    return mpo_tensors


def compute_eig_mps_with_mpo_hamiltonian(eig_mps, eig_mpo):
    con_tensor = [None] * len(eig_mps)
    for n in range(0, len(eig_mps)):
        con_tensor[n] = tc.einsum('dbbe,abc->adce', eig_mpo[n], eig_mps[n])
    tv = con_tensor[0]
    for gt in range(1, len(con_tensor)):
        tv = tc.einsum('qwer,erty->qwty', tv, con_tensor[gt])
    tr_result = tc.einsum('abab->', tv)
    return tr_result


def paper_loss_from_residual(residual_squared, num_spins):
    """论文定义 F = log2(||H-H_tilde||_HS^2) - N。"""
    tiny = tc.finfo(residual_squared.dtype).tiny
    return tc.log2(residual_squared.clamp_min(tiny)) - num_spins


def compute_loss(eig_mpss, eig_tensors, tr_hh):
    """变分能谱 Loss 计算内核（公开版唯一科学修改：使用论文 loss）。"""
    frac_a = tr_hh
    frac_b = mps_indot(eig_mpss)
    frac_c = compute_eig_mps_with_mpo_hamiltonian(eig_mpss, eig_tensors)
    loss_trace = frac_a - frac_c - tc.conj(frac_c) + frac_b
    return paper_loss_from_residual(loss_trace.real, len(eig_mpss))


def make_spectrum_mps_from_float_to_complex128(mps):
    """
    已修正: 删除了非法篡改非叶子节点求导状态的逻辑。
    PyTorch 的 tc.complex 会原生、自动地向下游正确传导其计算图依赖。
    """
    m_list = []
    for m in mps:
        c_m = tc.complex(m, tc.zeros(m.shape, device=m.device, dtype=m.dtype))
        m_list.append(c_m)
    return m_list


def trigger_singularity_escape(Physical_tensors, Entanglement_layers_list, Eigen_MPSs, device):
    """
    已修正: 通过 .data.add_ 绕过 PyTorch 的叶子节点原地保护限制，确保任意软硬件环境百分之百稳健逃逸。
    """
    with tc.no_grad():
        for n in Physical_tensors:
            noise = tc.complex(tc.randn_like(n.real), tc.randn_like(n.imag)) * 1e-12
            n.data.add_(noise)
        for layer in Entanglement_layers_list:
            for n in layer:
                noise = tc.complex(tc.randn_like(n.real), tc.randn_like(n.imag)) * 1e-12
                n.data.add_(noise)
        for n in Eigen_MPSs:
            noise_float = tc.randn(n.shape, device=device, dtype=tc.float64) * 1e-12
            n.data.add_(noise_float)
    tc.cuda.empty_cache()


# =====================================================================
# 【多层线路级联增强版】：工业级健壮权重加载与多路径拼装算子
# =====================================================================

def load_and_align_tensor_list(file_path, device, dtype=tc.complex128, requires_grad=True):
    """从磁盘载入张量列表，斩断旧图，强行对齐硬件设备、数据类型并重新激活叶子梯度"""
    raw_list = tc.load(file_path, map_location=device)
    aligned_list = []
    for tensor in raw_list:
        aligned_tensor = tensor.detach().to(device=device, dtype=dtype)
        aligned_tensor.requires_grad = requires_grad
        aligned_list.append(aligned_tensor)
    return aligned_list


def try_warm_start_loading_v2(init_cfg, log_file, device):
    """
    支持多层纠缠线路（单文件嵌套/多文件列表拼装）的高度健壮加载引擎
    返回: (success_bool, Physical_tensors, Eigen_MPSs, Entanglement_layers_list, loaded_layers_count)
    """
    import os
    mode = init_cfg.get("mode", "random")

    if mode == "random":
        fprint("[INIT] 策略配置为 'random'，跳过磁盘检查，执行原生随机冷启动。", file=log_file)
        return False, None, None, None, 0

    p_path, m_path = "", ""
    e_paths = []  # 统一转化为列表处理
    load_nl = init_cfg.get("load_nl", 0)

    # 1. 路由解析
    if mode == "auto":
        from .config import config
        plot_save_dir = config["logging"]["plot_save_dir"]
        p_path = os.path.join(plot_save_dir, f'Physical_tensorsNL={load_nl}.pth')
        m_path = os.path.join(plot_save_dir, f'Eigen_MPSsNL={load_nl}.pth')
        if load_nl > 0:
            e_paths = [os.path.join(plot_save_dir, f'Entanglement_layers_listNL={load_nl}.pth')]
        fprint(f"[INIT] 'auto' 模式：尝试自动搜寻 NL={load_nl} 的完整存档...", file=log_file)

    elif mode == "manual":
        paths = init_cfg.get("manual_paths", {})
        p_path = paths.get("physical_tensors", "")
        m_path = paths.get("eigen_mps", "")
        raw_e = paths.get("entanglement_layers", "")

        # 核心：动态识别单路径(str)与多路径拼装(list)
        if isinstance(raw_e, str):
            if raw_e.strip():
                e_paths = [raw_e]
        elif isinstance(raw_e, list):
            e_paths = [p for p in raw_e if p.strip()]
        fprint(f"[INIT] 'manual' 模式：解析完成，检测到物理层*1，波函数*1，纠缠线路文件*{len(e_paths)}", file=log_file)

    # 2. 防御性存在检查
    files_missing = False
    if not (os.path.exists(p_path) and os.path.exists(m_path)):
        files_missing = True
    for path in e_paths:
        if not os.path.exists(path):
            files_missing = True

    if files_missing:
        msg = f"❌ [INIT ERROR] 未能找齐必要的初始化存档文件，请检查路径是否存在。"
        if not init_cfg.get("fallback_on_failure", True):
            raise FileNotFoundError(msg)
        fprint(msg + "\n⚠️ [FALLBACK] 已触发安全自愈机制：自动降级为全随机冷启动初始化！", file=log_file)
        return False, None, None, None, 0

    # 3. 执行安全的对齐载入与多级拼接
    try:
        fprint(f" -> 正在安全载入物理层并强制转换为 complex128 叶子节点...", file=log_file)
        loaded_physical = load_and_align_tensor_list(p_path, device, dtype=tc.complex128, requires_grad=True)

        fprint(f" -> 正在安全载入波函数 MPS 并强制转换为 float64 叶子节点...", file=log_file)
        loaded_mps = load_and_align_tensor_list(m_path, device, dtype=tc.float64, requires_grad=True)

        loaded_entangle_layers = []

        # 遍历所有待拼装的文件路径
        for path in e_paths:
            raw_data = tc.load(path, map_location=device)

            # 防御性解析：判断读取出来的是单层(List of Tensors) 还是 多层嵌套(List of Lists of Tensors)
            if isinstance(raw_data, list) and len(raw_data) > 0:
                if isinstance(raw_data[0], list):
                    # 情况 A：保存的是一个多层嵌套列表（如原生的完整的全局存档）
                    for layer in raw_data:
                        aligned_layer = []
                        for tensor in layer:
                            aligned_tensor = tensor.detach().to(device=device, dtype=tc.complex128)
                            aligned_tensor.requires_grad = True
                            aligned_layer.append(aligned_tensor)
                        loaded_entangle_layers.append(aligned_layer)
                else:
                    # 情况 B：该文件是一个被拆分出来的单层线路列表，直接对齐并归入大列表中
                    aligned_layer = []
                    for tensor in raw_data:
                        aligned_tensor = tensor.detach().to(device=device, dtype=tc.complex128)
                        aligned_tensor.requires_grad = True
                        aligned_layer.append(aligned_tensor)
                    loaded_entangle_layers.append(aligned_layer)

        actual_layers_count = len(loaded_entangle_layers)
        fprint(f"🔥 [INIT SUCCESS] 全套参数热启动加载成功！累计拼装并注入可微纠缠线路：{actual_layers_count} 层。", file=log_file)
        return True, loaded_physical, loaded_mps, loaded_entangle_layers, actual_layers_count

    except Exception as e:
        msg = f"❌ [INIT CRASH] 读取或拼装 .pth 线路发生未预料的底层结构损坏: {e}"
        if not init_cfg.get("fallback_on_failure", True):
            raise e
        fprint(msg + "\n⚠️ [FALLBACK] 已触发安全自愈机制：自动降级为全随机冷启动初始化！", file=log_file)
        return False, None, None, None, 0


def plot_and_save_energy(loss_history, current_nl, save_dir):
    """绘制符合高质量学术期刊规范的能量收敛曲线"""
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(10, 7), dpi=100)
    plt.tick_params(labelsize=14)
    plt.xlabel("Num of Optimize (Epoch)", fontsize=16)
    plt.ylabel("Variational Spectrum Loss", fontsize=16)
    plt.grid(axis='both', c='g', linestyle='--', alpha=0.3)
    plt.plot(loss_history, color='deeppink', linewidth=2.5, label=f'NL={current_nl} Minimize Profile')
    plt.legend(prop={'family': 'Times New Roman', 'size': 14}, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Minimize_Energy_NL_{current_nl}.pdf'))
    plt.close()
