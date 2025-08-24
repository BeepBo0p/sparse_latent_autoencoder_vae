# Henrik's Simple Python Project Template

A barebones modern Python project template. This repository serves as a starting point for new Python projects, providing a well-structured foundation with modern tooling and best practices, which can easily be tailored to project-specific needs.

## 🚀 Getting Started

This template is designed to be used as a GitHub template repository. Click "Use this template" to create a new repository based on this structure.

### After Creating Your Project

1. Clone your new repository
2. Update the project name in `pyproject.toml`
3. Update this README with your project-specific information
4. Start building your application!

## 📁 Project Structure

```
├── project/                   # Main package directory
│   ├── __init__.py            # Package initialization
│   └── __main__.py            # Entry point for CLI execution
├── pyproject.toml             # Project configuration and dependencies
├── uv.lock                    # Dependency lock file (managed by uv)
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── .python-version            # Python version specification
├── .gitignore                 # Git ignore patterns
└── README.md                  # This file
```

## 🛠️ Bundled Tools

This template comes pre-configured with modern Python development tools:

### **uv** - Fast Python Package Manager
- Ultra-fast package resolution and installation
- Replaces pip and pip-tools with better performance
- Manages virtual environments automatically

### **Ruff** - Lightning Fast Linter & Formatter
- Combines the functionality of multiple tools (flake8, black, isort, etc.)
- Extremely fast performance written in Rust
- Configured via `pyproject.toml`

### **Pyrefly** - Type Checker
- Fast type checking for Python
- Alternative to mypy with better performance
- Configured in `pyproject.toml`

### **Pre-commit Hooks**
- Automatically runs checks before each commit
- Includes trailing whitespace, YAML validation, and more
- Runs Ruff for linting and formatting
- Runs Pyrefly for type checking

## 🚀 Usage

### Running the Application

```bash
# Using uv
uv run start

# Or if installed
python -m start
```

### Development Setup

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up pre-commit hooks (optional but recommended):
   ```bash
   uvx pre-commit install
   ```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

### Code Quality

```bash
# Run linting and formatting
uvx ruff check .
uvx ruff format .

# Run type checking
uvx pyrefly

# Run pre-commit hooks manually
uvx pre-commit run --all-files
```

## 📝 Customization

1. **Update project metadata** in `pyproject.toml`:
   - Change `name`, `description`, `version`
   - Add your dependencies
   - Configure tool settings as needed

2. **Rename the main package**:
   - Rename the `project/` directory to your package name
   - Update the entry point in `pyproject.toml`

3. **Configure tools** via `pyproject.toml`:
   - Adjust Ruff rules and settings
   - Modify Pyrefly configuration
   - Add tool-specific configurations

## 🤝 Contributing

This template is designed to be a starting point. Feel free to:
- Fork and customize for your application
- Submit issues and improvements
- Share your experience using this template

## 📄 License

[MIT](LICENSE)
