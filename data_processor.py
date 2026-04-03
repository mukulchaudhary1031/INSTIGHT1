import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── File Loader ───────────────────────────────────────────
def load_file(file_path: str, ext: str) -> pd.DataFrame:
    try:
        if ext == 'csv':
            for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return pd.read_csv(file_path, encoding=enc)
                except:
                    continue
            return None
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif ext == 'pdf':
            return extract_pdf_table(file_path)
        return None
    except Exception as e:
        print(f"Load error: {e}")
        return None


def extract_pdf_table(file_path: str) -> pd.DataFrame:
    """Extract tables from PDF using pdfplumber"""
    try:
        import pdfplumber
        all_tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df.columns = [str(c).strip() for c in df.columns]
                        all_tables.append(df)
        if all_tables:
            result = pd.concat(all_tables, ignore_index=True)
            for col in result.columns:
                try:
                    result[col] = pd.to_numeric(result[col])
                except:
                    pass
            return result
        return None
    except Exception as e:
        print(f"PDF error: {e}")
        return None


# ─── Data Info ─────────────────────────────────────────────
def get_dataset_info(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return {
        "shape":            {"rows": df.shape[0], "cols": df.shape[1]},
        "numeric_cols":     numeric_cols,
        "categorical_cols": cat_cols,
        "missing_pct":      {col: round(df[col].isnull().mean() * 100, 2)
                             for col in df.columns if df[col].isnull().any()},
        "unique_counts":    {col: int(df[col].nunique()) for col in df.columns},
    }


# ─── Data Cleaner ──────────────────────────────────────────
def clean_data(df: pd.DataFrame, target_col: str):
    report = {}
    df = df.copy()

    # 1. Drop columns with >80% missing
    thresh = len(df) * 0.2
    before_cols = df.shape[1]
    df.dropna(axis=1, thresh=int(thresh), inplace=True)
    report["cols_dropped_high_missing"] = before_cols - df.shape[1]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was dropped (>80% missing)")

    # 2. Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = before - len(df)

    # ── NEW FIX: Remove rows where categorical target is a bare number ──
    # e.g. target values like "0", "1", "2.0" mixed in with "Low","High","Medium"
    # These are corrupted/missing data disguised as a class label.
    if df[target_col].dtype == object or str(df[target_col].dtype) == 'category':
        numeric_mask = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.match(r'^-?\d+(\.\d+)?$')  # matches "0", "1", "2.5", "-1" etc.
        )
        n_bad = int(numeric_mask.sum())
        if n_bad > 0:
            bad_vals = df.loc[numeric_mask, target_col].unique().tolist()
            print(f"[Clean] ⚠️  Dropping {n_bad} rows where '{target_col}' has numeric-only values: {bad_vals}")
            df = df[~numeric_mask].copy()
            report["numeric_class_rows_dropped"] = n_bad
            report["numeric_class_values_removed"] = [str(v) for v in bad_vals]
        else:
            report["numeric_class_rows_dropped"] = 0

    # 3. Missing value stats before filling
    missing_before = df.isnull().sum()
    report["missing_before"] = {k: int(v) for k, v in missing_before.items() if v > 0}

    # 4. Fill numeric with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # 5. Fill categorical with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown", inplace=True)

    # 6. Outlier capping (IQR) — only on feature numeric cols
    outlier_report = {}
    for col in numeric_cols:
        if col == target_col:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out > 0:
            df[col] = df[col].clip(lower, upper)
            outlier_report[col] = n_out
    report["outliers_capped"] = outlier_report

    report["final_shape"] = {"rows": df.shape[0], "cols": df.shape[1]}
    return df, report