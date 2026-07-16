# -*- coding: utf-8 -*-
"""
config.py - 变分量子态对角化全局参数配置文件（多层线路拓展版）
"""

config = {
    "system": {
        "spin_num": 14,
        "cut_entanglement_dims": 48,
        "mdims": 16,
        "save_qising_name": 'MBL_random_TFIM_automata_Hamiltonian_MPO_Jz=1_hz=0.0_N=14.pth',
    },
    "schedule": {
        "start_layer": 0,
        "max_layers": 10,
        "max_epoch_per_layer": 4000,
        "base_lr": 2e-3,
        "nl0_lr": 8e-2,
        "eta_min": 1e-6,
    },
    "convergence": {
        "patience": 50,
        "absolute_tol": 1e-3,
    },
    "strategy": {
        "alternative_optimize": True,
        "mps_epochs": 2,
        "circuit_epochs": 4,
    },
    "init_strategy": {
        "mode": "auto",  # 'random', 'auto', 'manual'
        "load_nl": 9,  # 'auto' 模式下读取的基准层数
        "fallback_on_failure": True,  # 失败时是否自动降级为随机冷启动

        # 核心增强：当 mode == 'manual' 时，纠缠层路径完美兼容单个字符串或多个文件路径组成的列表
        "manual_paths": {
            "physical_tensors": "./results/Physical_tensorsNL=9.pth",
            "eigen_mps": "./results/Eigen_MPSsNL=9.pth",

            # 兼容写法1（单文件完整载入）:  "./results/Entanglement_layers_listNL=2.pth"
            # 兼容写法2（多层异地拼装级联）: ["./exp1/layer1.pth", "./exp2/layer2.pth", "./exp3/layer3.pth"]
            "entanglement_layers": ["./results/Entanglement_layers_listNL=9.pth"]
        }
    },
    "logging": {
        "dt_print": 10,
        "log_file": './automation_record.log',
        "plot_save_dir": './results'
    }
}
