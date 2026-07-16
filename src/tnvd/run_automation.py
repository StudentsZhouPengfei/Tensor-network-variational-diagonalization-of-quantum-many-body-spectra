# -*- coding: utf-8 -*-
"""
run_automation.py - 变分量子态对角化全自动运行主引擎（多层拼装完全体版）
"""

import os
import copy
import torch as tc
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import tn_utils as ut
from .config import config

# 强制环境配置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
device_cuda = tc.device('cuda' if tc.cuda.is_available() else 'cpu')


def set_gradient_state(parameters_list, requires_grad=True):
    for param in parameters_list:
        param.requires_grad = requires_grad


def is_absolutely_converged(loss_history, patience, tol):
    if len(loss_history) < patience:
        return False
    window = loss_history[-patience:]
    return (max(window) - min(window)) < tol


def main():
    sys_cfg = config["system"]
    sch_cfg = config["schedule"]
    conv_cfg = config["convergence"]
    strat_cfg = config["strategy"]
    init_cfg = config.get("init_strategy", {"mode": "random"})
    log_cfg = config["logging"]

    ut.fprint("=" * 60, file=log_cfg["log_file"])
    ut.fprint(f"▶️ 启动可微张量网络自动化对角化引擎 (多层拼装完全体版)", file=log_cfg["log_file"])
    ut.fprint("=" * 60, file=log_cfg["log_file"])

    # 1. 基础物理环境载入与基准 Trace 计算
    MPO_Hamiltonian_qising = tc.load(sys_cfg['save_qising_name'], map_location=device_cuda)
    tr_QIsing_HH = ut.inner_with_differ_2mpo(MPO_Hamiltonian_qising, MPO_Hamiltonian_qising)

    # =====================================================================
    # 2. 核心变分结构构建控制流 (支持多文件异地级联加载与拼装)
    # =====================================================================
    success, loaded_p, loaded_m, loaded_e, loaded_layers_count = ut.try_warm_start_loading_v2(init_cfg,
                                                                                              log_cfg["log_file"],
                                                                                              device_cuda)

    if success:
        # 如果热启动成功，无缝接管变量与层数基准
        Physical_tensors = loaded_p
        Eigen_MPSs = loaded_m
        Entanglement_layers_list = loaded_e
        # 如果用户指定的模式是按 load_nl 自动匹配，则加载基准设为 load_nl；否则设为实际拼装出的层数
        base_loaded_nl = load_nl = init_cfg.get("load_nl", 0) if init_cfg.get("mode") == "auto" else loaded_layers_count
    else:
        # 完美回归原生随机初始化状态
        Physical_tensors = ut.construct_physical_layer_tensors(sys_cfg['spin_num'] - 1, device_cuda)
        Eigen_MPSs = ut.construct_eigen_spectrum_mps_strict(sys_cfg['spin_num'], sys_cfg['mdims'], device_cuda)
        Entanglement_layers_list = []
        base_loaded_nl = 0

    # =====================================================================
    # 3. 开始多层自动化流水线 (平滑级联推进)
    # =====================================================================
    for current_nl in range(sch_cfg["start_layer"], sch_cfg["max_layers"] + 1):
        ut.fprint(f"\n[LAYER IN PROGRESS] 正在进入层数 NL = {current_nl} 的深度变分网络...", file=log_cfg["log_file"])

        # 核心增强：只有当流水线当前层数大于已加载的权重基准时，才追加新的纠缠层
        # 比如：拼装了3层历史线路(0,1,2)，当 current_nl=3 时，会自动为网络追加第3层近单位阵线路，完美衔接！
        if current_nl > 0 and (not success or current_nl > base_loaded_nl):
            ut.fprint(f" -> 承接上一层最佳态，安全追加第 {current_nl} 层近单位阵纠缠层...", file=log_cfg["log_file"])
            new_layer = ut.construct_entanglement_layer_tensors(sys_cfg['spin_num'] - 1, device_cuda)
            Entanglement_layers_list.append(new_layer)

        # 如果加载的纠缠层过多而控制器范围设置较小，执行防御性滑动截取，确保计算图尺寸完全契合当前层数
        active_entanglement_layers = Entanglement_layers_list[:current_nl]

        # 4. 聚合当前层数下所有变分网络拥有的有效叶子参数
        current_circuit_params = copy.copy(Physical_tensors)
        for e_layer in active_entanglement_layers:
            current_circuit_params.extend(e_layer)

        all_trainable_parameters = copy.copy(Eigen_MPSs) + current_circuit_params

        # 动态配给学习率
        init_lr = sch_cfg["nl0_lr"] if current_nl == 0 else sch_cfg["base_lr"]

        optimizer = tc.optim.AdamW(all_trainable_parameters, lr=init_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=sch_cfg["max_epoch_per_layer"], eta_min=sch_cfg["eta_min"])

        loss_history = []

        # 5. 当前层变分 Epoch 主循环
        for vt in range(sch_cfg["max_epoch_per_layer"]):

            # --- 交替优化梯度开关切换 ---
            if strat_cfg["alternative_optimize"]:
                cycle_period = strat_cfg["mps_epochs"] + strat_cfg["circuit_epochs"]
                if (vt % cycle_period) < strat_cfg["mps_epochs"]:
                    set_gradient_state(Eigen_MPSs, requires_grad=True)
                    set_gradient_state(current_circuit_params, requires_grad=False)
                else:
                    set_gradient_state(Eigen_MPSs, requires_grad=False)
                    set_gradient_state(current_circuit_params, requires_grad=True)
            else:
                set_gradient_state(all_trainable_parameters, requires_grad=True)

            # --- 前向算符收缩链与反向传播 ---
            try:
                Eigen_MPS128 = ut.make_spectrum_mps_from_float_to_complex128(Eigen_MPSs)

                mpo_tensors = ut.forward_circuit_evolution(
                    MPO_Hamiltonian_qising,
                    Physical_tensors,
                    active_entanglement_layers,  # 传入契合当前层的线路切片
                    sys_cfg['cut_entanglement_dims']
                )

                loss = ut.compute_loss(Eigen_MPS128, mpo_tensors, tr_QIsing_HH)
                loss_value = loss.item().real
                loss_history.append(loss_value)

                loss.backward()
                tc.nn.utils.clip_grad_norm_(all_trainable_parameters, max_norm=1.0)
                optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            except Exception as e:
                optimizer.zero_grad(set_to_none=True)
                ut.trigger_singularity_escape(Physical_tensors, active_entanglement_layers, Eigen_MPSs, device_cuda)
                continue

            # 6. 信息打印与收敛检查
            if (vt % log_cfg["dt_print"]) == 0:
                ut.fprint(
                    f"NL={current_nl} | Epoch {vt:4d} | Loss = {loss_value:.12f} | LR = {optimizer.param_groups[0]['lr']:.6f}",
                    file=log_cfg["log_file"])

            if is_absolutely_converged(loss_history, conv_cfg["patience"], conv_cfg["absolute_tol"]):
                ut.fprint(
                    f"🎯 [CONVERGED] NL={current_nl} 触发绝对收敛判据(波动范围 < {conv_cfg['absolute_tol']})，于第 {vt} Epoch 提前终止终止训练。",
                    file=log_cfg["log_file"])
                break

        # 7. 当前层退出，进行持久化归档与画图
        os.makedirs(log_cfg["plot_save_dir"], exist_ok=True)

        tc.save(Physical_tensors, os.path.join(log_cfg["plot_save_dir"], f'Physical_tensorsNL={current_nl}.pth'))
        tc.save(Eigen_MPSs, os.path.join(log_cfg["plot_save_dir"], f'Eigen_MPSsNL={current_nl}.pth'))
        if current_nl > 0:
            tc.save(active_entanglement_layers,
                    os.path.join(log_cfg["plot_save_dir"], f'Entanglement_layers_listNL={current_nl}.pth'))

        ut.plot_and_save_energy(loss_history, current_nl, log_cfg["plot_save_dir"])

        ut.fprint(f"💾 层数 NL={current_nl} 优化完成。最佳 Loss: {loss_history[-1]:.12f}，数据已归档。",
                  file=log_cfg["log_file"])

    ut.fprint("\n🎉 全套多层变分对角化自动化工程全部圆满成功！", file=log_cfg["log_file"])


if __name__ == '__main__':
    main()
