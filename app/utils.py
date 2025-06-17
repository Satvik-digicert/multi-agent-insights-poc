import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
from databricks import sql


def choose_chart_type(df: pd.DataFrame) -> str:
    n_rows, n_cols = df.shape
    cols = df.columns
    if n_cols == 2 and (pd.api.types.is_datetime64_any_dtype(df[cols[0]]) or pd.api.types.is_numeric_dtype(df[cols[0]])) and pd.api.types.is_numeric_dtype(df[cols[1]]):
        return "line"
    if n_cols == 2 and (pd.api.types.is_object_dtype(df[cols[0]]) or pd.api.types.is_categorical_dtype(df[cols[0]])) and pd.api.types.is_numeric_dtype(df[cols[1]]):
        return "bar"
    if n_cols == 2 and (pd.api.types.is_object_dtype(df[cols[1]]) or pd.api.types.is_categorical_dtype(df[cols[1]])) and pd.api.types.is_numeric_dtype(df[cols[0]]):
        return "column"
    if n_cols >= 2 and all(pd.api.types.is_numeric_dtype(df[c]) for c in cols[:2]):
        return "scatter"
    if n_cols >= 3 and all(pd.api.types.is_numeric_dtype(df[c]) for c in cols[:3]):
        return "bubble"
    if n_cols == 2 and pd.api.types.is_object_dtype(df[cols[0]]) and pd.api.types.is_numeric_dtype(df[cols[1]]) and n_rows <= 10:
        return "pie"
    if n_cols == 2 and pd.api.types.is_object_dtype(df[cols[0]]) and pd.api.types.is_numeric_dtype(df[cols[1]]) and n_rows <= 10:
        return "donut"
    if n_cols >= 2 and all(pd.api.types.is_object_dtype(df[c]) for c in cols[:-1]) and pd.api.types.is_numeric_dtype(df[cols[-1]]):
        return "treemap"
    if n_cols == 3 and all(pd.api.types.is_object_dtype(df[c]) for c in cols[:2]) and pd.api.types.is_numeric_dtype(df[cols[2]]):
        return "heatmap"
    if n_cols == 2 and pd.api.types.is_object_dtype(df[cols[0]]) and pd.api.types.is_numeric_dtype(df[cols[1]]) and n_rows > 10:
        return "pareto"
    if set(['lat', 'latitude', 'lon', 'longitude']).issubset(set(map(str.lower, cols))):
        return "geo"
    if n_cols == 2 and pd.api.types.is_object_dtype(df[cols[0]]) and pd.api.types.is_numeric_dtype(df[cols[1]]) and "total" in df[cols[0]].str.lower().values:
        return "waterfall"
    if n_cols == 2 and "step" in cols[0].lower() and pd.api.types.is_numeric_dtype(df[cols[1]]):
        return "funnel"
    if n_cols == 1 and pd.api.types.is_numeric_dtype(df[cols[0]]):
        return "histogram"
    ohlc = set(['open', 'high', 'low', 'close'])
    if ohlc.issubset(set(map(str.lower, cols))):
        return "candlestick"
    if n_cols == 2 and (pd.api.types.is_datetime64_any_dtype(df[cols[0]]) or pd.api.types.is_object_dtype(df[cols[0]])) and pd.api.types.is_numeric_dtype(df[cols[1]]):
        return "area"
    if n_cols == 1 and n_rows == 1 and pd.api.types.is_numeric_dtype(df[cols[0]]):
        return "kpi"
    if set(['source', 'target', 'value']).issubset(set(map(str.lower, cols))):
        return "sankey"
    if n_cols > 2 and pd.api.types.is_object_dtype(df[cols[0]]) and all(pd.api.types.is_numeric_dtype(df[c]) for c in cols[1:]):
        return "radar"
    return "bar"

def generate_visualization(df: pd.DataFrame, query_hash: str) -> str:
    chart_type = choose_chart_type(df)
    plt.figure(figsize=(6, 4))
    img_filename = f"visualization_{query_hash}.png"
    try:
        if chart_type == "bar":
            df.iloc[:10].plot(kind="bar", ax=plt.gca())
        elif chart_type == "line":
            df.iloc[:10].plot(kind="line", ax=plt.gca())
        elif chart_type == "scatter" and df.shape[1] >= 2:
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
        else:
            df.iloc[:10].plot(kind="bar", ax=plt.gca())
        plt.tight_layout()
        plt.savefig(img_filename)
        plt.close()
    except Exception as e:
        plt.close()
        print(f"Visualization generation failed: {e}")
        return None
    return img_filename

def image_to_base64(img_filename: str) -> str:
    if not img_filename or not os.path.exists(img_filename):
        return ""
    with open(img_filename, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
    return b64
