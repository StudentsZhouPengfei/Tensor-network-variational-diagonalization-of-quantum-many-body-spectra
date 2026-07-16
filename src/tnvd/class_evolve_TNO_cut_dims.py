from torch.optim.lr_scheduler import StepLR
import torch as tc
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
device_cpu = tc.device('cpu')  # 声明cpu设备
device_cuda = tc.device('cuda')  # 设备cuda设备
dtype = tc.float64  # 全文的数据类型


def clone_list(ini_tensor):
    new_list = []
    for n in range(len(ini_tensor)):
        new_list.append(ini_tensor[n].clone() * 1.0)
    return new_list


def get_full_tensor(wait_list):
    eig_values = tc.einsum('dquf,fwig,geoh,hrpj,jtak,kysl->dqwertyuiopasl', wait_list[0], wait_list[1],
                           wait_list[2], wait_list[3], wait_list[4], wait_list[5]).reshape(2 ** 6, -1)
    return eig_values


def custom_svd(svd_n, noise):
    # 进行 SVD 分解
    u, lm, v = tc.linalg.svd(svd_n + noise, full_matrices=False)

    # 调整奇异向量的相位使得每个奇异向量的第一个非零元素为正
    for i in range(min(u.size(1), v.size(0))):
        # u[:, i] 和 v[i, :] 分别是左奇异向量和右奇异向量
        if u[0, i] < 0:
            u[:, i] *= -1
        if v[i, 0] < 0:
            v[i, :] *= -1

    return u, lm, v


def robust_complex_svd(wait_tensor, cut_d=None, eps=1e-8):
    """
    稳定的带噪声复数 SVD 替代方案。
    包含：预归一化稳定数值 -> 注入 float64 噪声打破简并 -> SVD 分解 -> 奇异值尺度还原
    """
    # 1. 计算当前张量的整体范数（返回的是一个实数标量 float64）
    tensor_norm = tc.norm(wait_tensor)

    # [安全锁]：防止张量极度接近 0 导致除以 0 产生 NaN 梯度
    if tensor_norm < 1e-15:
        # 如果张量太小，干脆不进行缩放，避免放大本应被丢弃的数值噪声
        safe_norm = tc.tensor(1.0, dtype=tensor_norm.dtype, device=tensor_norm.device)
    else:
        safe_norm = tensor_norm

    # 2. 预归一化：将张量元素尺度强行拉回 O(1)，极大提升 BP 稳定性
    normalized_tensor = wait_tensor / safe_norm

    # 3. 构造 float64 级别的纯实数微小噪声
    noise_real = tc.normal(0, eps, size=normalized_tensor.size(), device=normalized_tensor.device, dtype=tc.float64)
    noise_imag = tc.zeros_like(noise_real)
    noise = tc.complex(noise_real, noise_imag)

    # 4. 将噪声注入预归一化后的张量
    safe_tensor = normalized_tensor + noise

    # 5. 使用 PyTorch 原生的 svd (此时输入矩阵的 norm 接近 1，极其健康)
    u, lm, vh = tc.linalg.svd(safe_tensor, full_matrices=False)

    # 6. 【核心】把借走的范数乘回到奇异值上！
    # 注意：此时的 lm 是实数(float64)，safe_norm 也是实数，所以直接相乘不会引发数据类型冲突
    lm = lm * safe_norm * 1.0

    # 7. 底层维度截断
    if cut_d is not None and cut_d < len(lm):
        lm = lm[:cut_d]
        u = u[:, :cut_d]
        vh = vh[:cut_d, :]

    return u, lm, vh


def robust_complex_svd_gauge_fixed(wait_tensor, cut_d=None, eps=1e-8):
    tensor_norm = tc.norm(wait_tensor)
    if tensor_norm < 1e-15:
        safe_norm = tc.tensor(1.0, dtype=tensor_norm.dtype, device=tensor_norm.device)
    else:
        safe_norm = tensor_norm

    normalized_tensor = wait_tensor / safe_norm

    # 注入噪声
    noise_real = tc.normal(0, eps, size=normalized_tensor.size(), device=normalized_tensor.device, dtype=tc.float64)
    noise_imag = tc.zeros_like(noise_real)
    noise = tc.complex(noise_real, noise_imag)
    safe_tensor = normalized_tensor + noise

    # 原生 SVD
    u, lm, vh = tc.linalg.svd(safe_tensor, full_matrices=False)

    # ---------------------------------------------------------
    # 【核心修复】：强制 U(1) 规范固定 (Gauge Fixing)
    # ---------------------------------------------------------
    # 1. 计算第一行的复数角度角 phi，shape 为 (1, K)
    phase = tc.angle(u[0:1, :])

    # 2. 构造旋转因子 e^{-i * phi}，shape 为 (1, K)
    phase_factor = tc.exp(-1j * phase)

    # 3. 旋转 u 矩阵，(M, K) * (1, K) -> 沿着列正常广播
    u = u * phase_factor

    # 4. 旋转 vh 矩阵
    # 由于 vh 的 shape 是 (K, N)，必须把 phase_factor 转置为 (K, 1) 才能沿着行正确广播
    vh = vh * tc.conj(phase_factor).t()
    # ---------------------------------------------------------

    lm = lm * safe_norm * 1.0

    if cut_d is not None and cut_d < len(lm):
        lm = lm[:cut_d]
        u = u[:, :cut_d]
        vh = vh[:cut_d, :]

    return u, lm, vh


def robust_complex_svd_hook_imag_grad(wait_tensor, cut_d=None, eps=1e-8):
    """
    可微张量网络专家级 SVD：梯度净化版 (修复显存泄漏 Bug)
    """
    tensor_norm = tc.norm(wait_tensor)
    safe_norm = tc.tensor(1.0, dtype=tensor_norm.dtype, device=tensor_norm.device) if tensor_norm < 1e-15 else tensor_norm
    normalized_tensor = wait_tensor / safe_norm

    noise_real = tc.normal(0, eps, size=normalized_tensor.size(), device=normalized_tensor.device, dtype=tc.float64)
    noise_imag = tc.zeros_like(noise_real)
    noise = tc.complex(noise_real, noise_imag)
    safe_tensor = normalized_tensor + noise

    u, lm, vh = tc.linalg.svd(safe_tensor, full_matrices=False)

    # ---------------------------------------------------------
    # 【显存修复核心】：用 .detach() 生成纯数据替身，彻底斩断计算图的循环引用！
    # ---------------------------------------------------------
    if u.requires_grad:
        u_det = u.detach() # 脱离计算图
        u.register_hook(lambda g: g - 1j * (u_det @ tc.diag(tc.imag(tc.diagonal(u_det.conj().t() @ g))).to(g.dtype)))
    if vh.requires_grad:
        vh_det = vh.detach() # 脱离计算图
        vh.register_hook(lambda g: g - 1j * (tc.diag(tc.imag(tc.diagonal(g @ vh_det.conj().t()))).to(g.dtype) @ vh_det))
    # ---------------------------------------------------------

    lm = lm * safe_norm * 1.0

    if cut_d is not None and cut_d < len(lm):
        lm = lm[:cut_d]
        u = u[:, :cut_d]
        vh = vh[:cut_d, :]

    return u, lm, vh


class Eigen_orthogonal:
    def __init__(self, ini_tensor, eps=1e-12):
        self.tensor = clone_list(ini_tensor)
        self.length = len(ini_tensor)
        self.center = -1
        self.eps = eps
        self.id2 = tc.eye(2, dtype=ini_tensor[0].dtype, device=ini_tensor[0].device)   # 变形所用张量
        self.identity22 = tc.einsum('ab,cg->abcg', self.id2, self.id2).data

    def tns_orthogonalization(self, location, cut_d, if_turn):
        if self.center < -0.5:
            self.orthogonalize_n1_n2(0, location, cut_d, if_turn)
            self.orthogonalize_n1_n2(self.length - 1, location, cut_d, if_turn)
        elif self.center != location:
            self.orthogonalize_n1_n2(self.center, location, cut_d, if_turn)
        self.center = location

    def orthogonalize_n1_n2(self, n1, n2, cut_d, if_turn):
        if n1 < n2:
            for nt in range(n1, n2, 1):
                self.orthogonalize_left2right(nt, cut_d, if_turn)
        else:
            for nt in range(n1, n2, -1):
                self.orthogonalize_right2left(nt, cut_d, if_turn)

    def orthogonalize_left2right(self, nt, cut_d, if_turn, normalize=False):       # 默认不对奇异值归一
        # if dc=-1 则不发生裁剪。
        if if_turn:
            if 0 < cut_d < self.tensor[nt].shape[-1]:
                pass  # 保持 if_turn 为 True
            else:
                if_turn = False
        else:
            if_turn = False
        # a = tc.einsum('qwer,rtyu->qwetyu', self.tensor[nt], self.tensor[nt + 1])
        s_n = self.tensor[nt].shape
        svd_tensor = self.tensor[nt].reshape(s_n[0] * s_n[1] * s_n[2], -1)
        u, lm, v = robust_complex_svd_hook_imag_grad(svd_tensor)
        lm = lm.to(dtype=tc.complex128)
        if if_turn:
            lm = lm[:cut_d]
            if normalize is True:
                lm = lm / tc.norm(lm)
            u = u[:, :cut_d]
            r = tc.diag(lm).mm(v[:cut_d, :])
        else:
            r = tc.diag(lm).mm(v)
        self.tensor[nt] = u.reshape(s_n[0], s_n[1], s_n[2], -1)
        self.tensor[nt + 1] = tc.einsum('ab,bcde->acde', r, self.tensor[nt + 1])
        # b = tc.einsum('qwer,rtyu->qwetyu', self.tensor[nt], self.tensor[nt + 1])
        # print('正交中心向右移动，相邻两个site求和后的相减范数差', tc.norm(a - b).item())

    def orthogonalize_right2left(self, nt, cut_d, if_turn, normalize=False):      # 默认不对奇异值归一
        if if_turn:
            if 0 < cut_d < self.tensor[nt].shape[0]:
                if_turn = True  # 保持 if_turn 为 True
            else:
                if_turn = False
        else:
            if_turn = False
        # a = tc.einsum('qwer,rtyu->qwetyu', self.tensor[nt - 1], self.tensor[nt])
        s_n = self.tensor[nt].shape
        svd_tensor = self.tensor[nt].reshape(-1, s_n[1] * s_n[2] * s_n[3]).clone()
        u, lm, v_t = robust_complex_svd_hook_imag_grad(svd_tensor)
        lm = lm.to(dtype=tc.complex128)
        if if_turn:
            lm = lm[:cut_d]
            if normalize is True:
                lm = lm / tc.norm(lm)
            v_t = v_t[:cut_d, :]
            r = u[:, :cut_d].mm(tc.diag(lm))
        else:
            r = u.mm(tc.diag(lm))
        self.tensor[nt] = v_t.reshape(-1, s_n[1], s_n[2], s_n[3])
        self.tensor[nt - 1] = tc.einsum('abcd,de->abce', self.tensor[nt - 1], r)
        # print(v.mm(v.t()))
        # b = tc.einsum('qwer,rtyu->qwetyu', self.tensor[nt - 1], self.tensor[nt])
        # print('正交中心向左移动，相邻两个site求和后的相减范数差', tc.norm(a - b).item())

    def evolve_bra_ket_4_bond_gate(self, n_g, gate, cut_dim, which_dir, if_turn=True, normalize=False):
        # 默认不对奇异值归一
        if which_dir == 'right':
            self.tns_orthogonalization(n_g, cut_dim, if_turn)  # 对施密特tns进行正交化
            # tensor_bk_2 = tc.einsum('qwer,rtyu,iowt,eypa->qipoau', self.tensor[n_g], self.tensor[n_g + 1], gate, tc.conj(gate))
            # print(tc.einsum('qwer,aser->qwas', gate, tc.conj(gate)).reshape(4, 4))
            tensor_bk_0 = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            tensor_bk_1 = tc.einsum('qwerty,uiwr->queity', tensor_bk_0, gate)
            tensor_bk_2 = tc.einsum('qwerty,uiet->qwuriy', tensor_bk_1, tc.conj(gate))
            # print(tc.norm(tensor_bk_0 - tensor_bk_2).item())
            sn = tensor_bk_2.shape
            svd_n = tensor_bk_2.reshape(sn[0] * sn[1] * sn[2], -1)
            u, lm, v = robust_complex_svd_hook_imag_grad(svd_n)
            lm = lm.to(dtype=tc.complex128)
            # print(tc.norm(u.mm(tc.diag(lm).mm(v)) - svd_n))
            if if_turn is True:
                lm = lm[:cut_dim]
                if normalize is True:
                    lm = lm / tc.norm(lm)
                u = u[:, :cut_dim]
                r = tc.diag(lm).mm(v[:cut_dim, :])
            else:
                r = tc.diag(lm).mm(v)
            self.tensor[n_g] = u.reshape(sn[0], sn[1], sn[2], -1)
            self.tensor[n_g + 1] = r.reshape(-1, sn[3], sn[4], sn[5])
            self.center = n_g + 1
            # b = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            # print('正交中心向右移动，并演化相应的量子门，相邻两个site求和后的相减范数差', tc.norm(tensor_bk_2 - b).item())
            # print(u.t().mm(u))
        elif which_dir == 'left':
            self.tns_orthogonalization(n_g + 1, cut_dim, if_turn)  # 对施密特tns进行正交化
            tensor_bk_0 = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            tensor_bk_1 = tc.einsum('qwerty,uiwr->queity', tensor_bk_0, gate)
            tensor_bk_2 = tc.einsum('qwerty,uiet->qwuriy', tensor_bk_1, tc.conj(gate))
            sn = tensor_bk_2.shape
            svd_n = tensor_bk_2.reshape(sn[0] * sn[1] * sn[2], -1)
            u, lm, v_h = robust_complex_svd_hook_imag_grad(svd_n)
            lm = lm.to(dtype=tc.complex128)

            if if_turn is True:
                lm = lm[:cut_dim]
                if normalize is True:
                    lm = lm / tc.norm(lm)
                # 直接对 v_h 进行行截断，无需转置
                v_h = v_h[:cut_dim, :]
                r = u[:, :cut_dim].mm(tc.diag(lm))
            else:
                # 保持 v_h 不变
                r = u.mm(tc.diag(lm))
            self.tensor[n_g + 1] = v_h.reshape(-1, sn[3], sn[4], sn[5])
            self.tensor[n_g] = r.reshape(sn[0], sn[1], sn[2], -1)
            self.center = n_g
            # b = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            # print('正交中心向右移动，并演化相应的门，相邻两个site求和后的相减范数差', tc.norm(tensor_bk_2 - b))

    def evo_4_bond_gate(self, n_g, gate):         # 每次演化对称层中的 上U和下U_dagger
        evo_gate_u = tc.einsum('qwer,tywu->qteyur', self.tensor[n_g], gate)
        evo_gate_u_t = tc.einsum('qwerty,uieo->qwurtyoi', evo_gate_u, tc.conj(gate))
        s_l = self.tensor[n_g].shape
        self.tensor[n_g] = evo_gate_u_t.reshape(s_l[0], s_l[1], s_l[2], -1) * 1.0

        evo_gate_idl = tc.einsum('qwer,tyuw->qutyer', self.tensor[n_g + 1], self.identity22)
        evo_gate_idr = tc.einsum('qwerty,uiot->ewqouriy', evo_gate_idl, tc.conj(self.identity22))
        s_id = self.tensor[n_g + 1].shape
        self.tensor[n_g + 1] = evo_gate_idr.reshape(-1, s_id[1], s_id[2], s_id[3]) * 1.0

    def evolve_bra_ket_4_bond_gate_dont_cut(self, n_g, gate, cut_dim, which_dir, if_turn=True, normalize=False):
        # 默认不对奇异值归一
        if which_dir == 'right':
            if self.center not in [n_g, n_g + 1]:
                self.tns_orthogonalization(n_g, cut_dim, if_turn)  # 对施密特tns进行正交化
            # tensor_bk_2 = tc.einsum('qwer,rtyu,iowt,eypa->qipoau', self.tensor[n_g], self.tensor[n_g + 1], gate, tc.conj(gate))
            # print(tc.einsum('qwer,aser->qwas', gate, tc.conj(gate)).reshape(4, 4))
            tensor_bk_0 = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            tensor_bk_1 = tc.einsum('qwerty,uiwr->queity', tensor_bk_0, gate)
            tensor_bk_2 = tc.einsum('qwerty,uiet->qwuriy', tensor_bk_1, tc.conj(gate))
            # print(tc.norm(tensor_bk_0 - tensor_bk_2).item())
            sn = tensor_bk_2.shape
            svd_n = tensor_bk_2.reshape(sn[0] * sn[1] * sn[2], -1)
            u, lm, v = robust_complex_svd_hook_imag_grad(svd_n)
            lm_ = lm.to(dtype=tc.complex128)
            if if_turn is True:
                lm_ = lm_[:cut_dim]
                if normalize is True:
                    lm_ = lm_ / tc.norm(lm_)

                self.tensor[n_g] = u[:, :cut_dim].reshape(sn[0], sn[1], sn[2], -1)
                self.tensor[n_g + 1] = tc.diag(lm_).mm(v[:cut_dim, :]).reshape(-1, sn[3], sn[4], sn[5])
            else:
                # r = tc.diag(lm_).mm(v)
                self.tensor[n_g] = u.reshape(sn[0], sn[1], sn[2], -1)
                self.tensor[n_g + 1] = tc.diag(lm_).mm(v).reshape(-1, sn[3], sn[4], sn[5])
            self.center = n_g + 1
            # b = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            # print('正交中心向右移动，并演化相应的量子门，相邻两个site求和后的相减范数差', tc.norm(tensor_bk_2 - b).item())
        elif which_dir == 'left':
            if self.center not in [n_g, n_g + 1]:
                self.tns_orthogonalization(n_g + 1, cut_dim, if_turn)  # 对施密特tns进行正交化
            tensor_bk_0 = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            tensor_bk_1 = tc.einsum('qwerty,uiwr->queity', tensor_bk_0, gate)
            tensor_bk_2 = tc.einsum('qwerty,uiet->qwuriy', tensor_bk_1, tc.conj(gate))
            sn = tensor_bk_2.shape
            svd_n = tensor_bk_2.reshape(sn[0] * sn[1] * sn[2], -1)
            u, lm, v = robust_complex_svd_hook_imag_grad(svd_n)
            lm = lm.to(dtype=tc.complex128)
            if if_turn:
                lm = lm[:cut_dim]
                if normalize:
                    lm = lm / tc.norm(lm)
                v = v[:cut_dim, :]
                r = u[:, :cut_dim].mm(tc.diag(lm))
            else:
                r = u.mm(tc.diag(lm))
            self.tensor[n_g + 1] = v.reshape(-1, sn[3], sn[4], sn[5])
            self.tensor[n_g] = r.reshape(sn[0], sn[1], sn[2], -1)
            self.center = n_g
            # b = tc.einsum('qwer,rtyu->qwetyu', self.tensor[n_g], self.tensor[n_g + 1])
            # print('正交中心向右移动，并演化相应的门，相邻两个site求和后的相减范数差', tc.norm(tensor_bk_2 - b))
