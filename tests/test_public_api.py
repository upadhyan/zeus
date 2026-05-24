"""The public surface advertised by `zeus.__init__`."""
import subprocess
import sys

import zeus


def test_top_level_exports():
    assert hasattr(zeus, "Zeus")
    assert hasattr(zeus, "ZeusClusterer")
    assert hasattr(zeus, "__version__")
    assert isinstance(zeus.__version__, str)


def test_no_torch_train_imports_on_module_load():
    """`import zeus` should not pull in wandb, openml, or omegaconf.

    Runs in a subprocess to isolate from sys.modules pollution: when pytest
    collects tests/test_openml_regression.py, that module imports openml at
    module-load time, so by the time this test runs in the parent process
    `openml in sys.modules` is True regardless of what `import zeus` did.
    The subprocess gives us a clean interpreter where we can actually
    observe zeus's import side-effects.
    """
    code = (
        "import sys; import zeus; "
        "forbidden = ('wandb', 'openml', 'omegaconf'); "
        "leaked = [m for m in forbidden if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"`import zeus` leaked forbidden modules: {result.stderr}"
    )
