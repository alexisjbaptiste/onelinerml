import numpy as np
import pandas as pd


def compute_analytics(data, target, estimator, X_test, y_test, y_pred,
                      feature_columns, preprocessor):
    """Compute comprehensive analytics for the dashboard."""
    analytics = {}

    # --- Correlation matrix (numeric columns only) ---
    numeric_df = data.select_dtypes(include=["number"])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        analytics["correlation"] = {
            "columns": corr.columns.tolist(),
            "values": np.round(corr.values, 4).tolist(),
        }

    # --- Target distribution ---
    target_vals = data[target]
    if pd.api.types.is_numeric_dtype(target_vals):
        analytics["target_distribution"] = {
            "type": "numeric",
            "mean": round(float(target_vals.mean()), 4),
            "std": round(float(target_vals.std()), 4),
            "min": round(float(target_vals.min()), 4),
            "max": round(float(target_vals.max()), 4),
            "median": round(float(target_vals.median()), 4),
            "histogram": _histogram(target_vals),
        }
    else:
        counts = target_vals.value_counts()
        analytics["target_distribution"] = {
            "type": "categorical",
            "labels": counts.index.tolist(),
            "counts": counts.values.tolist(),
        }

    # --- Feature importance ---
    analytics["feature_importance"] = _feature_importance(
        estimator, feature_columns, preprocessor
    )

    # --- Residuals / confusion data ---
    if pd.api.types.is_float_dtype(np.array(y_test)):
        residuals = (np.array(y_test) - np.array(y_pred)).tolist()
        analytics["residuals"] = {
            "y_test": np.round(y_test, 4).tolist(),
            "y_pred": np.round(y_pred, 4).tolist(),
            "residuals": [round(r, 4) for r in residuals],
        }
    else:
        from sklearn.metrics import confusion_matrix
        labels = sorted(list(set(y_test) | set(y_pred)), key=str)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        analytics["confusion_matrix"] = {
            "labels": [str(l) for l in labels],
            "values": cm.tolist(),
        }

    # --- Feature stats ---
    numeric_cols = set(data.select_dtypes(include=["number"]).columns)
    feature_stats = []
    for col in feature_columns:
        stat = {"name": col}
        if col in numeric_cols:
            stat["type"] = "numeric"
            stat["mean"] = round(float(data[col].mean()), 4)
            stat["std"] = round(float(data[col].std()), 4)
            stat["missing"] = int(data[col].isna().sum())
        else:
            stat["type"] = "categorical"
            stat["unique"] = int(data[col].nunique())
            stat["top"] = str(data[col].mode().iloc[0]) if not data[col].mode().empty else "N/A"
            stat["missing"] = int(data[col].isna().sum())
        feature_stats.append(stat)
    analytics["feature_stats"] = feature_stats

    # --- Dataset info ---
    analytics["dataset"] = {
        "rows": len(data),
        "features": len(feature_columns),
        "target": target,
        "model": type(estimator).__name__,
    }

    return analytics


def _histogram(series, bins=20):
    counts, edges = np.histogram(series.dropna(), bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": np.round(edges, 4).tolist(),
    }


def _feature_importance(estimator, feature_columns, preprocessor):
    """Extract feature importance from the estimator if available."""
    importance = None

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.array(estimator.coef_)
        if coef.ndim > 1:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)

    if importance is None:
        return None

    # Map back to feature names (preprocessor may have expanded categoricals)
    try:
        names = preprocessor.get_feature_names_out()
        names = [str(n) for n in names]
    except Exception:
        names = [f"feature_{i}" for i in range(len(importance))]

    if len(names) != len(importance):
        names = [f"feature_{i}" for i in range(len(importance))]

    # Sort by importance
    idx = np.argsort(importance)[::-1]
    return {
        "names": [names[i] for i in idx],
        "values": [round(float(importance[i]), 6) for i in idx],
    }
