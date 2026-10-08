"""Core learn must leave the PHP checkpoint export hook as the sole exporter."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from wbt_training.utils.student_teacher import PhpDistillation, PhpDistillationPPO


@pytest.mark.parametrize("algo_cls", [PhpDistillation, PhpDistillationPPO])
def test_php_save_hook_is_the_only_automatic_export(algo_cls, monkeypatch, tmp_path):
    events = []
    algo = object.__new__(algo_cls)
    algo.current_learning_iteration = 0
    algo.config = SimpleNamespace(num_learning_iterations=2, distillation_warmup_steps=0, save_interval=1)
    algo._train_mode = lambda: None
    algo.policy = SimpleNamespace(
        require_loaded_teachers=lambda: None,
        state_dict=dict,
        num_teachers=5,
        loaded_teachers=[True] * 5,
    )
    algo.optimizer = SimpleNamespace(state_dict=dict)
    algo.learning_rate = 0.001
    algo.ppo_coef = 0.0
    algo.env = SimpleNamespace(reset_all=dict)
    algo.is_multi_gpu = False
    algo.is_main_process = True
    algo.log_dir = str(tmp_path)
    algo._collect_env_state = dict
    algo._checkpoint_metadata = lambda **kwargs: {}
    algo.logging_helper = SimpleNamespace(
        record_collection_time=nullcontext,
        record_learn_time=nullcontext,
        save_checkpoint_artifact=lambda payload, path: events.append(("save", payload["iter"])),
    )
    algo._rollout_step = lambda obs: obs
    algo._training_step = dict
    algo._post_epoch_logging = lambda iteration, losses: None
    algo.adjust_ppo_dagger_coeff = lambda iteration: None
    algo._set_std_lr_from_ppo_coef = lambda: None
    algo.export = lambda **kwargs: pytest.fail("Core automatic export would overwrite the PHP bundle")
    monkeypatch.setattr(
        "wbt_training.utils.exporter.export_depth_student_bundle",
        lambda model, iteration: events.append(("php_export", iteration)),
    )

    # Calls the actual core learn loop, PHP save mixin, and core checkpoint save.
    algo.learn()

    assert events == [
        ("save", 0), ("php_export", 0),
        ("save", 1), ("php_export", 1),
        ("save", 1), ("php_export", 1),
    ]
