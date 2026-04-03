import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Max unique values allowed for categorical encoding ──
MAX_CARDINALITY = 50


def _prepare_features(df: pd.DataFrame, target_col: str):
    """
    Smart feature preparation:
    - Drops ID-like columns (all unique)
    - Drops high-cardinality categorical columns (> MAX_CARDINALITY unique vals)
    - Returns X, num_cols, cat_cols
    """
    X = df.drop(columns=[target_col]).copy()
    n = len(X)

    drop_cols = []
    for col in X.columns:
        u = X[col].nunique()
        # Drop if all unique (ID column) or too high cardinality cat
        if u == n:
            drop_cols.append(col)
            print(f"[ML] Dropping ID-like column: {col} ({u} unique)")
        elif X[col].dtype == object and u > MAX_CARDINALITY:
            drop_cols.append(col)
            print(f"[ML] Dropping high-cardinality column: {col} ({u} unique vals)")

    if drop_cols:
        X = X.drop(columns=drop_cols)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[ML] Features used — numeric: {num_cols}, categorical: {cat_cols}")
    return X, num_cols, cat_cols


def _build_preprocessor(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols))
    return ColumnTransformer(transformers)


def _get_feature_names(prep, num_cols, cat_cols):
    names = list(num_cols)
    try:
        ct = prep.named_transformers_.get('cat')
        if ct:
            enc = ct.named_steps.get('enc')
            if enc and hasattr(enc, 'get_feature_names_out'):
                names += enc.get_feature_names_out(cat_cols).tolist()
    except:
        names += cat_cols
    return names


def _get_importance(model, num_cols, cat_cols):
    fi = {}
    try:
        prep  = model.named_steps['preprocessing']
        mdl   = model.named_steps['model']
        names = _get_feature_names(prep, num_cols, cat_cols)
        if hasattr(mdl, 'feature_importances_'):
            imp = mdl.feature_importances_
        elif hasattr(mdl, 'coef_'):
            imp = np.abs(mdl.coef_.flatten() if mdl.coef_.ndim > 1 else mdl.coef_)
        else:
            return fi
        raw = dict(zip(names, imp))
        top = dict(sorted(raw.items(), key=lambda x: x[1], reverse=True)[:10])
        fi  = {k: round(float(v), 5) for k, v in top.items()}
    except Exception as e:
        print(f"[ML] Feature importance error: {e}")
    return fi


# ════════════════════════════════════════════════════
#  CLASSIFICATION
# ════════════════════════════════════════════════════
def train_classification(df: pd.DataFrame, target_col: str) -> dict:
    X, num_cols, cat_cols = _prepare_features(df, target_col)
    y = df[target_col]

    le    = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    feature_cols   = X.columns.tolist()
    feature_dtypes = {c: str(X[c].dtype) for c in feature_cols}

    strat = y_enc if len(np.unique(y_enc)) > 1 and len(np.unique(y_enc)) < 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=strat
    )

    prep = _build_preprocessor(num_cols, cat_cols)

    # ── Fast models — no heavy GridSearch ──
    MODELS = {
        "Random Forest":      RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "Decision Tree":      DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    best_score, best_model, best_name = -1, None, ""
    all_results = {}

    for name, clf in MODELS.items():
        try:
            pipe = Pipeline([('preprocessing', prep), ('model', clf)])
            # 3-fold CV for speed
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            score = cv_scores.mean()
            all_results[name] = round(score * 100, 2)
            print(f"[ML] {name}: CV Accuracy = {score:.4f}")
            if score > best_score:
                best_score = score
                best_name  = name
                pipe.fit(X_train, y_train)
                best_model = pipe
        except Exception as e:
            print(f"[ML] {name} failed: {e}")
            continue

    if best_model is None:
        raise ValueError("All models failed to train. Check your dataset.")

    # Re-fit best model fully on training data
    y_pred = best_model.predict(X_test)
    avg    = 'binary' if len(np.unique(y_enc)) == 2 else 'weighted'
    cm     = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":         round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision":        round(precision_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        "recall":           round(recall_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        "f1_score":         round(f1_score(y_test, y_pred, average=avg, zero_division=0) * 100, 2),
        "cv_accuracy":      round(best_score * 100, 2),
        "confusion_matrix": cm,
        "all_models":       all_results,
        "classes":          le.classes_.tolist(),
    }

    fi = _get_importance(best_model, num_cols, cat_cols)

    return {
        "model":            best_model,
        "best_model_name":  best_name,
        "metrics":          metrics,
        "feature_importance": fi,
        "feature_cols":     feature_cols,
        "feature_dtypes":   feature_dtypes,
        "label_encoder":    le,
    }


# ════════════════════════════════════════════════════
#  REGRESSION
# ════════════════════════════════════════════════════
def train_regression(df: pd.DataFrame, target_col: str) -> dict:
    X, num_cols, cat_cols = _prepare_features(df, target_col)
    y = pd.to_numeric(df[target_col], errors='coerce')

    # Drop rows where target is NaN after conversion
    valid  = y.notna()
    X, y   = X[valid], y[valid]

    feature_cols   = X.columns.tolist()
    feature_dtypes = {c: str(X[c].dtype) for c in feature_cols}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    prep = _build_preprocessor(num_cols, cat_cols)

    MODELS = {
        "Random Forest":    RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Linear Regression":LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "Decision Tree":    DecisionTreeRegressor(max_depth=10, random_state=42),
    }

    best_score, best_model, best_name = -np.inf, None, ""
    all_results = {}

    for name, reg in MODELS.items():
        try:
            pipe      = Pipeline([('preprocessing', prep), ('model', reg)])
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
            score     = cv_scores.mean()
            all_results[name] = round(score, 4)
            print(f"[ML] {name}: CV R² = {score:.4f}")
            if score > best_score:
                best_score = score
                best_name  = name
                pipe.fit(X_train, y_train)
                best_model = pipe
        except Exception as e:
            print(f"[ML] {name} failed: {e}")
            continue

    if best_model is None:
        raise ValueError("All models failed. Check dataset — target must be numeric.")

    y_pred = best_model.predict(X_test)
    mse    = mean_squared_error(y_test, y_pred)

    metrics = {
        "r2_score":  round(r2_score(y_test, y_pred), 4),
        "mse":       round(mse, 4),
        "rmse":      round(np.sqrt(mse), 4),
        "mae":       round(mean_absolute_error(y_test, y_pred), 4),
        "cv_r2":     round(best_score, 4),
        "all_models":all_results,
    }

    fi = _get_importance(best_model, num_cols, cat_cols)

    return {
        "model":            best_model,
        "best_model_name":  best_name,
        "metrics":          metrics,
        "feature_importance": fi,
        "feature_cols":     feature_cols,
        "feature_dtypes":   feature_dtypes,
        "label_encoder":    None,
    }