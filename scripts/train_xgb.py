"""
scripts/train_xgb.py

Trains the XGBoost hire-probability model and TF-IDF vectorizer
used by app/ml/ml_scorer.py.

Generates synthetic resume+JD pairs as training data — sufficient
for the capstone. Replace with real labeled data for production.

Usage:
    python scripts/train_xgb.py

Outputs (to data/ml/):
    tfidf_vectorizer.joblib
    xgb_model.joblib
    xgb_features.joblib
"""
from __future__ import annotations
import logging
import os
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Output paths (must match config.py) ──────────────────────────────────────
OUTPUT_DIR = Path("data/ml")
TFIDF_PATH = OUTPUT_DIR / "tfidf_vectorizer.joblib"
XGB_PATH   = OUTPUT_DIR / "xgb_model.ubj"
FEAT_PATH  = OUTPUT_DIR / "xgb_features.joblib"

# ── Synthetic training data ───────────────────────────────────────────────────
# Strong match keywords — high hire probability
STRONG_SKILLS = [
    "Python LangChain LangGraph FastAPI AWS SageMaker Docker Kubernetes RAG FAISS",
    "Java Spring Boot Microservices REST API Kafka Oracle JPA Hibernate AWS",
    "Python machine learning scikit-learn XGBoost NLP transformers PyTorch TensorFlow",
    "React TypeScript Node.js GraphQL PostgreSQL Redis Docker CI/CD Jenkins",
    "Python data engineering Spark Airflow dbt Snowflake AWS Glue ETL pipeline",
    "Java Spring Cloud Kubernetes Istio gRPC Prometheus Grafana OpenTelemetry",
    "MLOps SageMaker MLflow DVC model deployment monitoring Python AWS",
    "LLM fine-tuning RAG vector database Pinecone Weaviate LangChain agents",
    "DevOps Terraform AWS CDK GitHub Actions ArgoCD Helm Kubernetes",
    "Python FastAPI PostgreSQL SQLAlchemy Alembic pytest Docker AWS Lambda",
]

# Weak match keywords — low hire probability
WEAK_SKILLS = [
    "Microsoft Excel PowerPoint Word data entry customer service",
    "PHP WordPress HTML CSS jQuery basic web development",
    "Java Swing desktop application JDBC SQL Server Windows",
    "COBOL mainframe batch processing JCL VSAM IBM",
    "Manual testing QA test cases bug reporting Jira Excel",
    "Basic Python scripting automation Excel macros VBA",
    "Network administration Cisco CCNA firewall VPN DNS",
    "SAP ERP configuration ABAP business analyst requirements",
    "Android Java mobile development Firebase Play Store",
    "Ruby Rails PostgreSQL Heroku basic CRUD web app",
]

# Job description templates
JD_AI_ENGINEER = (
    "AI Engineer LLM LangChain LangGraph Python FastAPI AWS SageMaker "
    "RAG vector store FAISS embeddings agentic systems Docker Kubernetes "
    "MLOps model deployment observability LangFuse evaluation"
)
JD_JAVA_ENGINEER = (
    "Senior Java Engineer Spring Boot Microservices REST API Kafka AWS "
    "Oracle JPA Hibernate Docker Kubernetes CI/CD Jenkins Agile"
)
JD_DATA_ENGINEER = (
    "Data Engineer Python Spark Airflow dbt Snowflake AWS ETL pipeline "
    "SQL PostgreSQL S3 Glue Lambda data modeling"
)
JD_FULLSTACK = (
    "Full Stack Engineer React TypeScript Node.js FastAPI Python "
    "PostgreSQL Docker AWS REST GraphQL CI/CD"
)

JD_POOL = [JD_AI_ENGINEER, JD_JAVA_ENGINEER, JD_DATA_ENGINEER, JD_FULLSTACK]


def generate_training_data(n_samples: int = 600) -> tuple[list[str], list[int]]:
    """
    Generate synthetic (resume+JD, label) pairs.
    label=1 → strong match (hire), label=0 → weak match (no hire).
    """
    random.seed(42)
    texts, labels = [], []

    half = n_samples // 2

    # Positive samples — strong resume + matching JD
    for _ in range(half):
        resume = random.choice(STRONG_SKILLS)
        # Add some noise
        noise = random.choice(WEAK_SKILLS).split()[:3]
        resume = resume + " " + " ".join(noise)
        jd = random.choice(JD_POOL)
        texts.append(resume + " [SEP] " + jd)
        labels.append(1)

    # Negative samples — weak resume + any JD
    for _ in range(half):
        resume = random.choice(WEAK_SKILLS)
        jd = random.choice(JD_POOL)
        texts.append(resume + " [SEP] " + jd)
        labels.append(0)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def train() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic training data...")
    texts, labels = generate_training_data(n_samples=600)
    logger.info("Total samples: %d (pos=%d, neg=%d)",
                len(labels), sum(labels), len(labels) - sum(labels))

    # ── TF-IDF vectorizer ─────────────────────────────────────────────────────
    logger.info("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    feature_names = vectorizer.get_feature_names_out().tolist()
    logger.info("TF-IDF features: %d", len(feature_names))

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── XGBoost model ─────────────────────────────────────────────────────────
    logger.info("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    logger.info("Classification report:\n%s",
                classification_report(y_test, y_pred,
                                      target_names=["No hire", "Hire"]))

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(vectorizer, TFIDF_PATH)
    logger.info("Saved TF-IDF vectorizer → %s", TFIDF_PATH)

    model.save_model(str(XGB_PATH))
    logger.info("Saved XGBoost model     → %s", XGB_PATH)

    joblib.dump(feature_names, FEAT_PATH)
    logger.info("Saved feature names     → %s", FEAT_PATH)

    logger.info("Training complete. Restart the backend to load the new model.")


if __name__ == "__main__":
    train()
