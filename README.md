# 🧩 My Template Project

A clean and modern Python project template with **Poetry**, **Ruff**, **Mypy**, **Pytest**, and **Pre-commit** configured out of the box.

---

## 🚀 Features

- 🧱 Standardized project layout (`src/` + `tests/`)
- 🧹 Auto linting & formatting with [Ruff](https://github.com/astral-sh/ruff)
- 🔍 Static type checking via [Mypy](https://mypy.readthedocs.io/)
- 🧪 Testing setup with [Pytest](https://pytest.org/)
- 🪝 Git hooks via [Pre-commit](https://pre-commit.com/)
- ⚙️ Fully managed dependencies using [Poetry](https://python-poetry.org/)

---

## 📦 Project Structure

```text
.
├── src/
│ └── project_name/
│   └── __init__.py
├── tests/
│ └── test_sample.py
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── init_template.py
└── pyproject.toml
```

## 🧰 Setup Instructions

### 1️⃣ Install Dependencies

```bash
poetry install --with dev
python init_template.py
```
