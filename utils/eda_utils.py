import pandas as pd
import platform
import sys
import os
import pandas as pd
import platform
import sys
import os


def get_column_types(df: pd.DataFrame):
    return df.dtypes


def get_dataframe_info(df: pd.DataFrame):
    # Capture output of df.info()
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def get_summary_stats(df: pd.DataFrame):
    return df.describe(include='all')


def get_column_names(df: pd.DataFrame):
    return df.columns.tolist()


def get_null_counts(df: pd.DataFrame):
    return df.isnull().sum()


def get_correlation_matrix(df: pd.DataFrame):
    try:
        return df.corr(numeric_only=True)
    except Exception as e:
        return f"Could not compute correlation matrix: {e}"


def dataset_info_summary(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    return {
        "name": name,
        "column_types": get_column_types(df),
        "dataframe_info": get_dataframe_info(df),
        "summary_statistics": get_summary_stats(df),
        "column_names": get_column_names(df),
        "null_counts": get_null_counts(df),
        "correlation_matrix": get_correlation_matrix(df)
    }




def system_info() -> None:
    print("\n🖥️ System Info")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Interpreter Path: {sys.executable}")
    print(f"Current Working Dir: {os.getcwd()}")
    print(f"CPU Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
