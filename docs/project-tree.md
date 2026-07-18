# Project Tree

This tree highlights the cleaned local layout for Corrective Foresight.

```text
my_uva-main/
├── README.md
├── train.py
├── train_corrective_foresight.sh
├── eval_corrective_foresight.sh
├── docs/
│   ├── README.md
│   ├── architecture/
│   │   └── corrective-foresight.md
│   ├── experiments/
│   │   └── robotics-a-conference-plan.md
│   ├── implementation/
│   │   └── optimization-summary.md
│   └── project-tree.md
├── tests/
│   ├── test_causal_configs.py
│   ├── test_causal_mask.py
│   ├── test_corrective_foresight_policy.py
│   ├── test_causal_video_action.py
│   ├── test_checkpoint_discovery.py
│   ├── test_config_propagation.py
│   ├── test_mar_self_correction_shapes.py
│   └── test_self_correction.py
└── unified_video_action/
    ├── config/
    │   ├── corrective_foresight_pusht.yaml
    │   └── model/
    │       ├── corrective_foresight_policy.yaml
    │       └── corrective_foresight_backbone.yaml
    ├── eval/
    │   ├── checkpoint_discovery.py
    │   └── eval.py
    ├── model/
    │   └── autoregressive/
    │       ├── causal_mask.py
    │       ├── causal_video_action.py
    │       ├── flow_matching.py
    │       ├── flow_matching_action_loss.py
    │       ├── mar_con_unified.py
    │       └── self_correction.py
    └── policy/
        ├── corrective_foresight_policy.py
        └── unified_video_action_policy.py
```

## Folder Roles

- `docs/`: maintained project documentation for architecture, implementation, and experiments.
- `tests/`: lightweight local verification tests.
- `unified_video_action/model/autoregressive/`: model and loss code.
- `unified_video_action/eval/`: evaluation and checkpoint utilities.
