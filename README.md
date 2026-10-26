🚀 MLOps Tutorial — End-to-End ML Platform

<p align="center">
<strong>Building reliable, reproducible, and production-ready machine learning workflows.</strong>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+" />
<img src="https://img.shields.io/badge/MLOps-Model%20Lifecycle-6C47FF" alt="MLOps" />
<img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow" />
<img src="https://img.shields.io/badge/Feast-Feature%20Store-00A98F" alt="Feast" />
<img src="https://img.shields.io/badge/FastAPI-API%20Development-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License" />
</p>

---

## 📌 Overview

**MLOps Tutorial** This project is based on, but not an exact copy of a tutorial from [FreeCodeCamp](https://www.freecodecamp.org/news/build-end-to-end-ml-platform-locally-from-experiment-tracking-to-cicd/).
This is a hands-on machine learning engineering project focused on the tools and practices used to build, manage, validate, and monitor machine learning systems.

The project brings together data processing, feature management, model development, experiment tracking, data validation, and API-related tooling to demonstrate the building blocks of an end-to-end MLOps platform.

Rather than treating a machine learning model as an isolated Python script, this project explores the broader lifecycle required to make ML workflows more reproducible, maintainable, and production-oriented.

---

## ✨ Key Features

* 🧹 **Data processing:** Work with structured datasets using Pandas, NumPy, and PyArrow.
* 🗃️ **Feature management:** Use Feast to organize and manage machine learning features.
* 🧪 **Experiment tracking:** Use MLflow to support experiment and model lifecycle management.
* 🛡️ **Data validation:** Apply Pandera and Pydantic to define and enforce data contracts.
* 📊 **Model monitoring:** Explore data and model quality monitoring with Evidently.
* 🤖 **Machine learning:** Build and evaluate models with scikit-learn.
* 🌐 **API development:** Use FastAPI, Uvicorn, and Pydantic for model-serving workflows.
* 🔄 **Testing and automation:** Use Pytest and coverage tooling to support quality assurance.
* ⚙️ **Dependency management:** Use `uv` and the committed lockfile for reproducible environments.

---

## 🧰 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| 🐍 **Python** | Core programming language |
| 📦 **uv** | Dependency and environment management |
| 🐼 **Pandas & NumPy** | Data manipulation and numerical computing |
| 🏹 **PyArrow** | Columnar data processing |
| 🧠 **scikit-learn** | Machine learning |
| 🗃️ **Feast** | Feature store and feature management |
| 🧪 **MLflow** | Experiment tracking and ML lifecycle management |
| 📈 **Evidently** | Data quality and model monitoring |
| 🛡️ **Pandera** | DataFrame validation |
| ✅ **Pydantic** | Data schemas and validation |
| 🚀 **FastAPI** | API development |
| ⚡ **Uvicorn** | ASGI application server |
| 🧫 **Pytest** | Automated testing |
| 📊 **pytest-cov** | Test coverage reporting |
| 🔐 **skops** | Security-conscious handling of scikit-learn models |
| 🧑‍💻 **GitHub Actions** | CI/CD automation when configured |

---

## 🏗️ Project Architecture

The repository contains the main components needed to organize an ML engineering workflow.

```mermaid
flowchart TD
    A["📂 Data Sources"] --> B["🧹 Data Processing"]
    B --> C["🛡️ Data Validation"]
    C --> D["🗃️ Feature Management"]
    D --> E["🧠 Model Training"]
    E --> F["🧪 Experiment Tracking"]
    E --> G["📦 Model Artifacts"]
    G --> H["🚀 API / Model Serving"]
    H --> I["📈 Monitoring"]
    I --> J["🔄 Feedback and Improvement"]
```

*The diagram illustrates the intended MLOps workflow; the exact execution sequence depends on the implementation of the individual modules.*

---

## 🔍 Core Components

* **Data layer:** Stores datasets and supports data preparation.
* **Feature layer:** Organizes reusable features through Feast.
* **Model layer:** Contains model artifacts and supports machine learning workflows.
* **Tracking layer:** Uses MLflow tooling for experiment tracking.
* **Validation layer:** Uses Pandera and Pydantic to help catch invalid data.
* **Serving layer:** Provides the dependencies needed to develop model APIs.
* **Monitoring layer:** Uses Evidently to assess data or model quality.

---

## 📁 Repository Structure

The following tree summarizes the main directories and files currently present in the repository.

```text
mlops_tutorial_fcc/
│
├── 📂 data/                 # Data assets
├── 📂 feature_repo/         # Feature store configuration and assets
├── 📂 mlruns/1/             # MLflow-related tracking artifacts
├── 📂 models/               # Model artifacts
├── 📂 reports/              # Reports and analysis outputs
├── 📂 src/                  # Source code
│
├── 📄 .gitignore            # Git ignore rules
├── 📄 .python-version       # Python version configuration
├── 📄 LICENSE               # Project license
├── 📄 README.md             # Project documentation
├── 📄 mlruns.db             # Local tracking database
├── 📄 pyproject.toml        # Project metadata and dependencies
└── 📄 uv.lock               # Locked dependency versions
```

---

## ⚙️ Getting Started

### Prerequisites

Install the following before getting started:
* Python 3.12 or newer
* Git
* `uv` for Python project and dependency management

### 1. Clone the repository

```bash
git clone https://github.com/sathyamangalamshreyas/mlops_tutorial_fcc.git
cd mlops_tutorial_fcc
```

### 2. Install `uv`

If `uv` is not already installed, follow the official installation instructions. Verify your installation:

```bash
uv --version
python --version
```

### 3. Create the environment and install dependencies

The project includes both `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

This command creates or updates the project's virtual environment and installs the declared dependencies.

Activate the environment if you want to work directly inside it:

* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
* **Windows PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

*(You can also run project commands with `uv run` without manually activating the environment.)*

### 4. Verify the installation

Check that the Python environment can import the core dependencies:

```bash
uv run python -c "import pandas, sklearn, mlflow, feast, evidently; print('Environment ready!')"
```

---

## 🧪 Testing

The project declares Pytest and pytest-cov as dependencies.

Run the test suite from the repository root:
```bash
uv run pytest
```

To generate a coverage report:
```bash
uv run pytest --cov=src --cov-report=term-missing
```

To generate an HTML coverage report:
```bash
uv run pytest --cov=src --cov-report=html
```
*(The HTML report is written to `htmlcov/` by default.)*

---

## 📊 MLflow Experiment Tracking

MLflow provides tooling for recording model experiments, metrics, parameters, and artifacts.

To launch the MLflow tracking UI for a conventional local file-based tracking directory, run:
```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Then open `http://127.0.0.1:5000` in your browser.

> **Important:** The repository also contains `mlruns.db`. Confirm the tracking URI and backend configuration used by your implementation before selecting a tracking-store command. The example above uses `./mlruns` and may not display runs stored in a separate database.

---

## 🗃️ Feature Management with Feast

Feast helps manage and serve features used by machine learning models. The repository includes a `feature_repo/` directory for feature-store-related assets.

A typical Feast workflow includes:
1. Define entities, feature views, and data sources.
2. Configure the feature store.
3. Apply feature definitions.
4. Retrieve features for training or inference.
5. Keep training and serving features consistent.

To use Feast commands, first navigate to the directory containing the feature-store configuration:
```bash
cd feature_repo
uv run feast --help
```

---

## 🛡️ Data Validation and Monitoring

### Pandera and Pydantic
Data validation helps catch invalid records, missing fields, unexpected data types, and schema violations before they affect downstream model workflows.
* **Pandera:** DataFrame and tabular data validation.
* **Pydantic:** Python data models and structured input validation.
* **FastAPI integration:** Request and response validation for APIs.

### Evidently
Evidently provides tools for evaluating data quality and monitoring data or model behavior. Monitoring workflows can help identify:
* Missing or invalid values.
* Changes in feature distributions.
* Data drift between reference and current datasets.
* Changes in model quality when suitable ground-truth data is available.

---

## 🚀 API Development

FastAPI and Uvicorn are included as dependencies for building API-based applications. A typical model-serving workflow involves:
1. Load a validated model artifact.
2. Define request and response schemas.
3. Validate incoming data.
4. Generate predictions.
5. Return structured responses.
6. Record relevant metrics and monitor input data.

To launch the application, identify the FastAPI application object in `src/` and use its actual Python module path. For example, if the application object is named `app` in `src/main.py`:
```bash
uv run uvicorn src.main:app --reload
```

---

## 🔄 Continuous Integration and Automation

GitHub Actions can automate tasks such as:
* Installing project dependencies.
* Running unit and integration tests.
* Checking data and model validation logic.
* Measuring test coverage.
* Running quality checks before changes are merged.

---

## 🔐 Best Practices

* 📌 Keep dependencies reproducible with `uv.lock`.
* 🧪 Test data-processing and prediction logic.
* 🛡️ Validate data before training and inference.
* 🗃️ Version feature definitions and model artifacts.
* 📊 Track experiment parameters, metrics, and model versions.
* 🔍 Monitor data quality and detect unexpected distribution changes.
* 🔑 Keep credentials and secrets out of source control.
* 📦 Avoid committing large generated artifacts unless required by the project.
* 🔄 Automate repeatable checks with CI/CD.

---

## 🗺️ Suggested Learning Path

If you're using this repository to learn MLOps, follow this progression:
1. **Data engineering:** Understand how datasets are organized and processed.
2. **Data contracts:** Add schema and data-quality validation.
3. **Feature engineering:** Explore feature definitions and Feast.
4. **Model development:** Train and evaluate a scikit-learn model.
5. **Experiment tracking:** Record runs and compare model metrics with MLflow.
6. **Model serving:** Expose predictions through a FastAPI application.
7. **Monitoring:** Use Evidently to investigate data quality and drift.
8. **Automation:** Run tests and quality checks in a repeatable pipeline.

---

## 🤝 Contributing

Contributions, improvements, and suggestions are welcome!

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Make your changes and add appropriate tests.
4. Run the test suite:
   ```bash
   uv run pytest
   ```
5. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Describe your changes"
   git push origin feature/your-feature
   ```
6. Open a pull request describing your changes.

---

## 📄 License

This project includes an MIT license. See [LICENSE](LICENSE) for the full license text.

---

## 🔗 Resources

* 🐍 [Python](https://www.python.org/)
* 📦 [uv documentation](https://github.com/astral-sh/uv)
* 🗃️ [Feast documentation](https://docs.feast.dev/)
* 🧪 [MLflow documentation](https://mlflow.org/docs/latest/index.html)
* 📈 [Evidently documentation](https://docs.evidentlyai.com/)
* 🛡️ [Pandera documentation](https://pandera.readthedocs.io/)
* 🚀 [FastAPI documentation](https://fastapi.tiangolo.com/)
* 🧫 [Pytest documentation](https://docs.pytest.org/)
* ⚙️ [GitHub Actions documentation](https://docs.github.com/en/actions)

---

<p align="center">
⭐ If you find this project useful, consider starring the repository!
</p>

<p align="center">
Built for learning, experimentation, and better machine learning engineering.
</p>
