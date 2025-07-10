# control-guided-nas

## Setup

### Install uv

This project uses the [uv](https://github.com/astral-sh/uv) package manager. If it is not already installed, it can be easily installed by

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

or using [alternative methods](https://docs.astral.sh/uv/getting-started/installation/) (such as through [homebrew](https://brew.sh/)).

### Install dependencies

Once uv is installed, all dependencies, including appropriate python versions, can be installed to a dedicated virtual environment via a single command:

```bash
uv sync --all-extras
```

The `--all-extras` flag installs extra dependencies such as `ipykernel` useful for running the code interactively.
