This file is designed to inform contributions to the scenex docs.

# Building the docs

## Generating example pages

Before building or serving the docs, regenerate the example pages and screenshots:

```bash
uv run --exact --group docs python docs/gen_examples.py
```

This script:

- Runs each Python file in `examples/`, capturing an offscreen screenshot via `canvas.render()`
- Parses each file's module-level docstring
- Writes a `docs/examples/{name}.md` page with the description, screenshot, and source code
- Regenerates `docs/examples/index.md` as a gallery of all examples
- Updates the `{Examples = [...]}` entry in `zensical.toml` to match the current example set

Re-run this script whenever examples are added, removed, or their docstrings change. **Note the CI will run `gen_examples.py` as a part of the documentation build process**

## Serving locally

```bash
uv run --exact --group docs python -m zensical serve
```

### Serving locally via VSCode

The following launch configuration can be added to a `.vscode/launch.json` file:
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
