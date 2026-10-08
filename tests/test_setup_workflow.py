"""Installer contracts using subprocess doubles, without installing packages."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/bash\n' + text)
    path.chmod(0o755)


@pytest.mark.parametrize('env_name', [None, 'php-custom', 'hssim'])
def test_php_setup_passes_environment_name_to_holosoma(tmp_path, env_name):
    script = tmp_path / 'wbt_training/training_runs/setup_env.sh'
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / 'wbt_training/training_runs/setup_env.sh', script)
    holo = tmp_path / 'holo'
    executable(holo / 'scripts/setup_isaacsim.sh', 'printf "%s" "$CONDA_ENV_NAME" > "$SETUP_TEST_OUTPUT"\n')
    executable(tmp_path / 'scripts/source_isaacsim_setup.sh', ':\n')
    executable(tmp_path / 'python', 'exit 0\n')
    env = os.environ.copy()
    env.pop('CONDA_ENV_NAME', None)
    env.pop('ENV_NAME', None)
    if env_name is not None:
        env['ENV_NAME'] = env_name
    env.update(HOLOSOMA_LOCAL_DIR=str(holo), HSSIM_PYTHON=str(tmp_path / 'python'),
               SETUP_TEST_OUTPUT=str(tmp_path / 'name'))
    subprocess.run(['bash', str(script)], env=env, check=True, capture_output=True)
    assert (tmp_path / 'name').read_text() == (env_name or 'php')


@pytest.mark.parametrize('name,version', [('setup_isaacsim.sh', '3.11'), ('setup_mujoco.sh', '3.10'), ('setup_inference.sh', '3.10')])
def test_holosoma_bootstrap_creates_env_without_mutating_base(tmp_path, name, version):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy2(ROOT / 'thirdparty/holosoma/scripts' / name, scripts / name)
    (scripts / 'source_common.sh').write_text('WORKSPACE_DIR="$BOOTSTRAP_TEST_ROOT"\nCONDA_ROOT="$BOOTSTRAP_TEST_ROOT/miniconda3"\n')
    prefix = tmp_path / 'workspace'
    calls = tmp_path / 'calls'
    executable(prefix / 'miniconda3/bin/conda', 'printf "%s\\n" "$*" >> "$BOOTSTRAP_TEST_CALLS"\nexit 37\n')
    executable(tmp_path / 'bin/sudo', 'exit 0\n')
    env = os.environ.copy()
    env.update(BOOTSTRAP_TEST_ROOT=str(prefix), BOOTSTRAP_TEST_CALLS=str(calls),
               CONDA_ENV_NAME='bootstrap-test', PATH=str(tmp_path / 'bin') + os.pathsep + env['PATH'])
    result = subprocess.run(['bash', str(scripts / name)], env=env, capture_output=True, text=True)
    assert result.returncode == 37, result.stdout + result.stderr
    commands = calls.read_text().splitlines()
    assert len(commands) == 1
    assert commands[0].split() == ['create', '-y', '--prefix', str(prefix / 'miniconda3/envs/bootstrap-test'),
                                  'python=' + version, '--override-channels', '-c', 'conda-forge']
