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

**MLOps Tutorial** by FreeCodeCamp (https://www.freecodecamp.org/news/build-end-to-end-ml-platform-locally-from-experiment-tracking-to-cicd/) is a hands-on machine learning engineering project focused on the tools and practices used to build, manage, validate, and monitor machine learning systems.

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
