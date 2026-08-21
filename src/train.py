import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATIO = 0.248
DRIFT_TOLERANCE = 0.05
EXPERIMENT_NAME = "adult-income"


def _configure_mlflow() -> None:
    """Cau hinh tracking local mac dinh, nhung van ho tro DagsHub qua bien moi truong."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", EXPERIMENT_NAME)
    mlflow.set_tracking_uri(tracking_uri)

    # Voi tracking local, tao experiment voi artifact root ro rang de cac file
    # khong bi rai rac trong repo. Khi dung DagsHub, server tu quan ly artifact.
    if tracking_uri.startswith("sqlite"):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            artifact_root = Path(
                os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts")
            ).resolve()
            mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_root.as_uri(),
            )

    mlflow.set_experiment(experiment_name)


def _write_detail_report(
    path: str,
    y_true,
    default_preds,
    best_preds,
    best_threshold: float,
) -> None:
    """Ghi confusion matrix va precision/recall rieng cho tung lop."""
    matrix = confusion_matrix(y_true, default_preds, labels=[0, 1])
    precision, recall, f1_by_class, support = precision_recall_fscore_support(
        y_true,
        default_preds,
        labels=[0, 1],
        zero_division=0,
    )
    best_f1 = f1_score(y_true, best_preds, zero_division=0)

    lines = [
        "ADULT INCOME - DETAILED EVALUATION",
        "",
        "Confusion matrix at default threshold 0.50",
        "Rows = actual class, columns = predicted class",
        "                 predicted_0  predicted_1",
        f"actual_0         {matrix[0, 0]:11d}  {matrix[0, 1]:11d}",
        f"actual_1         {matrix[1, 0]:11d}  {matrix[1, 1]:11d}",
        "",
        "Per-class metrics at default threshold 0.50",
        "class  precision  recall  f1_score  support",
    ]
    for label, label_name in enumerate(("thu_nhap_thap", "thu_nhap_cao")):
        lines.append(
            f"{label} ({label_name})  "
            f"{precision[label]:.4f}  {recall[label]:.4f}  "
            f"{f1_by_class[label]:.4f}  {int(support[label])}"
        )

    lines.extend(
        [
            "",
            "Optimized decision threshold",
            f"best_threshold={best_threshold:.2f}",
            f"best_f1_score={best_f1:.4f}",
            "",
            "Interpretation note",
            "False negative (bo sot nguoi thu nhap cao) lam giam recall cua lop 1.",
            "False positive (gan nham thu nhap cao) lam giam precision cua lop 1.",
        ]
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    for name, dataframe in (("train", df_train), ("eval", df_eval)):
        if "target" not in dataframe.columns:
            raise ValueError(f"Dataset {name} thieu cot target")

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    if list(X_train.columns) != list(X_eval.columns):
        raise ValueError("Train va eval khong co cung thu tu dac trung")

    positive_ratio = float(y_train.mean())
    distribution_shift = abs(positive_ratio - REFERENCE_POSITIVE_RATIO)
    drift_warning = distribution_shift > DRIFT_TOLERANCE

    if drift_warning:
        print(
            "WARNING: Ty le lop duong "
            f"{positive_ratio:.1%} lech {distribution_shift:.1%} "
            f"so voi muc tham chieu {REFERENCE_POSITIVE_RATIO:.1%}."
        )
    else:
        print(
            "Data distribution OK: ty le lop duong "
            f"{positive_ratio:.1%} (tham chieu {REFERENCE_POSITIVE_RATIO:.1%})."
        )

    _configure_mlflow()
    git_sha = os.getenv("GITHUB_SHA", "local")
    execution_context = "github-actions" if os.getenv("GITHUB_ACTIONS") else "local"
    run_name = (
        f"gb-n{params['n_estimators']}-lr{params['learning_rate']}"
        f"-d{params['max_depth']}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "git_sha": git_sha,
                "execution_context": execution_context,
            }
        )

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        if 1 not in model.classes_:
            raise ValueError("Du lieu huan luyen khong chua lop duong target=1")
        positive_class_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_eval)[:, positive_class_index]

        default_preds = (probabilities >= 0.5).astype(int)
        f1 = float(f1_score(y_eval, default_preds, zero_division=0))
        acc = float(accuracy_score(y_eval, default_preds))

        threshold_results = []
        for index in range(17):
            threshold = round(0.10 + index * 0.05, 2)
            threshold_preds = (probabilities >= threshold).astype(int)
            threshold_f1 = float(
                f1_score(y_eval, threshold_preds, zero_division=0)
            )
            threshold_results.append((threshold, threshold_f1))

        # Neu nhieu threshold dong F1, uu tien threshold gan 0.5 nhat.
        best_threshold, best_f1 = max(
            threshold_results,
            key=lambda result: (result[1], -abs(result[0] - 0.5)),
        )
        best_preds = (probabilities >= best_threshold).astype(int)

        os.makedirs("outputs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        report = {
            "git_sha": git_sha,
            "execution_context": execution_context,
            "f1_score": f1,
            "accuracy": acc,
            "default_threshold": 0.5,
            "f1_score_default": f1,
            "best_threshold": best_threshold,
            "best_f1_score": best_f1,
            "positive_class_ratio": positive_ratio,
            "reference_positive_class_ratio": REFERENCE_POSITIVE_RATIO,
            "distribution_shift_percentage_points": distribution_shift * 100,
            "data_drift_warning": drift_warning,
            "train_rows": len(df_train),
            "eval_rows": len(df_eval),
        }
        with open("outputs/report.json", "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        _write_detail_report(
            "outputs/detail.txt",
            y_eval,
            default_preds,
            best_preds,
            best_threshold,
        )
        joblib.dump(model, "models/model.joblib")

        mlflow.log_metrics(
            {
                "f1_score": f1,
                "accuracy": acc,
                "f1_score_default": f1,
                "best_f1_score": best_f1,
                "best_threshold": best_threshold,
                "positive_class_ratio": positive_ratio,
                "distribution_shift_percentage_points": distribution_shift * 100,
            }
        )
        mlflow.log_artifact("outputs/report.json", artifact_path="reports")
        mlflow.log_artifact("outputs/detail.txt", artifact_path="reports")
        mlflow.sklearn.log_model(model, "model")

        print(
            f"F1@0.50: {f1:.4f} | Accuracy: {acc:.4f} | "
            f"Best F1: {best_f1:.4f} @ threshold {best_threshold:.2f}"
        )

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
