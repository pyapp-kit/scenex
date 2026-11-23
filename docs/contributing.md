---
icon: material/handshake
---

# Contributing

Contributions are welcome in many forms — bug fixes, new features, documentation improvements, and issue reports are all appreciated.

## Contributing Code

Development happens on [GitHub](https://github.com/pyapp-kit/scenex). To contribute a code improvement, [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository on GitHub, clone your fork, and install with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/<your-username>/scenex
cd scenex
uv sync
uv run --group lint pre-commit install
```

A plain `uv sync` installs `scenex` with its test and debugging tools, but without any rendering backend or GUI framework. Add the backend and GUI you want to work with as extras (see the table below), for example:

```bash
uv sync --extra pygfx --extra pyqt6 --group testqt
```

### Running tests

Once you've made a change, please write a test showing that your change works! This repository uses [pytest](https://docs.pytest.org/en/stable/) to run the `scenex` unit tests.

*The test suite needs at least one graphics backend (pygfx or VisPy) and one GUI frontend (Qt, Jupyter, or wx).* For example:

```bash
uv run --exact --extra pygfx --extra pyqt6 --group testqt pytest
```

Choose the extras and groups for the components you want to exercise:


| Component | Extra parameters |
|---|---|
| Qt | `--extra pyqt6 --group testqt` |
| Jupyter | `--extra jupyter --group testjupyter` |
| WxPython | `--extra wx` |
| VisPy | `--extra vispy` |
| pygfx | `--extra pygfx` |

For example, the following will run scenex tests pertaining to the `vispy` backend, the `qt` app, and the scenex model code:

```bash
uv run --exact --extra vispy --extra pyqt6 --group testqt pytest
```

### Code style

Pre-commit hooks handle formatting, linting, and type checking automatically on each commit. To run them manually:

```bash
uv run --group lint pre-commit run --all-files
```

The hooks run inside the `lint` dependency group. Because mypy and pyright check every adaptor, that group installs **all** backends (pygfx, VisPy, Qt, wx, Jupyter, imgui), and they're added to your `.venv` the first time the hooks run. If you then want to go back to a minimal environment, re-run your `uv sync --exact ...` command.

## Contributing Documentation

Documentation is built with [zensical](https://zensical.org/) (an mkdocs-based tool), sourced from the `docs/` folder.

### Regenerating example pages

Before building or serving the docs, regenerate the example pages and screenshots:

```bash
uv run --exact --extra pygfx --extra wx --group docs python docs/gen_examples.py
```

Re-run this script whenever examples are added, removed, or their docstrings change.

### Serving locally

```bash
uv run --group docs python -m zensical serve
```

The docs will be available at `http://127.0.0.1:8000` and will live-reload as you edit.

??? tip "Serving from VS Code"

    Add this launch configuration to `.vscode/launch.json`:

    ```json
    {
        "name": "zensical serve",
        "type": "debugpy",
        "request": "launch",
        "module": "zensical",
        "args": ["serve"],
        "python": "${workspaceFolder}/.venv/Scripts/python.exe",
        "cwd": "${workspaceFolder}",
        "console": "integratedTerminal",
        "justMyCode": false,
        "subProcess": true
    }
    ```
