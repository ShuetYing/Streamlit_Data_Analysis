import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def check_missing(df):
    df_copy = df.copy()
    return df_copy.isnull().sum()

def drop_missing(df, how="any"):
    df_copy = df.copy()
    return df_copy.dropna(how=how)

def fill_missing(df, columns, method="mean", value=None):
    df_copy = df.copy()
    for col in columns:
        if method == "mean":
            avg = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(avg)
        elif method == "median":
            median = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median)
        elif method == "frequency":
            idx = df_copy[col].value_counts().idxmax()
            df_copy[col] = df_copy[col].fillna(idx)
        elif method == "value" and value is not None:
            df_copy[col] = df_copy[col].fillna(value)
    return df_copy

def rename_column(df, old_name, new_name):
    df_copy = df.copy()
    return df_copy.rename(columns={old_name: new_name})

def drop_columns(df, columns):
    df_copy = df.copy()
    return df_copy.drop(columns=columns)

def drop_duplicates(df):
    df_copy = df.copy()
    return df_copy.drop_duplicates()

def scale_data(df, method="minmax"):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include="number").columns
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    return df_copy

def one_hot_encode(df, columns):
    df_copy = df.copy()
    return pd.get_dummies(df_copy, columns=columns)

def aggregate_data(df, group_col, agg_func):
    df_copy = df.copy()
    return df_copy.groupby(group_col).agg(agg_func).reset_index()

def log_transform(df, columns):
    df_copy = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            if (df_copy[col] >= 0).all():
                df_copy[col] = np.log1p(df_copy[col])
            else:
                st.warning(f"Log transform on '{col}' skipped because it contains negative values.")
    return df_copy

def convert_column(df, column, target):
    df_copy = df.copy()
    if target == "datetime":
        df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
    elif target == "numeric":
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    else:
        df_copy[column] = df_copy[column].astype('category')
    return df_copy

def extract_datetime_features(df, column):
    df_copy = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_copy[column]):
        df_copy[f'{column}_year'] = df_copy[column].dt.year
        df_copy[f'{column}_month'] = df_copy[column].dt.month
        df_copy[f'{column}_day'] = df_copy[column].dt.day
        df_copy[f'{column}_dayofweek'] = df_copy[column].dt.dayofweek
        df_copy[f'{column}_hour'] = df_copy[column].dt.hour
        df_copy = df_copy.drop(columns=[column])
    return df_copy

def reset_session_state():
    st.session_state.working_df = None if st.session_state.initial_df is None else st.session_state.initial_df.copy()

def reset_options():
    keys = ["show_missing", "drop_missing", "fill_missing", "rename_cols", "drop_col", "drop_dup",
            "scale_num", "one_hot", "aggregate", "log_transform", "convert_column", "extract_dt" ]
    for key in keys:
        st.session_state[key] = False

def main():
    st.set_page_config(page_title="Explore Data", layout="wide")

    if "initial_df" not in st.session_state:
        st.session_state.initial_df = None
    if "working_df" not in st.session_state:
        st.session_state.working_df = None
    
    left_col, right_col = st.columns([8,1.5])

    with left_col:
        st.header("Exploratory Data Analysis")
        st.caption("""The essential first step in any data project. **Dive deep into your dataset's structure and composition.** This suite of tools provides everything you need to understand your data and prepare it for analysis.""")
        st.caption("""
        - **Data Overview:** Preview, check dimensions (shape), and examine variable types.
        - **Summary Statistics:** Generate descriptive stats (mean, median, standard deviation, etc.) for each variable.
        - **Data Preprocessing:** Clean your dataset by handling missing values, encoding categories, and scaling features.
        """)
    
    if "initial_df" not in st.session_state or st.session_state.initial_df is None:
        st.warning("No data found. Please upload a dataset or use sample dataset on the 'Data Upload' page.")
        if st.button("Go to Upload Data", type='primary'):
            st.switch_page("pages/2_upload_data.py")
        return

    if 'working_df' not in st.session_state or st.session_state.working_df is None:
        st.session_state.working_df = st.session_state.initial_df.copy()

    with right_col:
        if st.button("Reset to Original Data", use_container_width=True):
            reset_session_state()
            reset_options()
            st.rerun()

    st.markdown("---")

    st.subheader("👀 Data Overview")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total Rows", st.session_state.working_df.shape[0])
        st.metric("Total Columns", st.session_state.working_df.shape[1])
        st.write("**Data Types**")
        st.dataframe(st.session_state.working_df.dtypes.rename("Data Type"), use_container_width=True)

    with col2:
        st.write("**Data Preview**")
        num_rows = st.slider("Select number of rows", 0, 50, 5)
        df_rows = len(st.session_state.working_df)
        if num_rows > df_rows:
            st.warning(f"The number of rows you selected ({num_rows}) exceeds the available data ({df_rows}) rows.")
        else:
            st.dataframe(st.session_state.working_df.head(num_rows), hide_index=True, use_container_width=True)

    with st.expander("Show Descriptive Statistics"):
        st.dataframe(st.session_state.working_df.describe(), use_container_width=True)

    st.markdown("---")

    st.subheader("🧹 Data Preprocessing")
    st.write("Apply transformations to your dataset.")

    temp_df = st.session_state.working_df.copy()

    pre_col1, pre_col2 = st.columns(2)
    with pre_col1:
        st.write("#### Missing Values")
        show_missing_summary = st.checkbox("Show missing values summary", key="show_missing")
        if show_missing_summary:
            st.dataframe(check_missing(temp_df), use_container_width=True)
        if st.checkbox("Drop missing values", key="drop_missing"):
            how = st.radio("Drop rows with:", ["any", "all"], help="Determine if row or column is removed. any : If any NA values are present, drop that row or column. all : If all values are NA, drop that row or column.", key="drop_how")
            temp_df = drop_missing(temp_df, how)
        if st.checkbox("Fill missing values", key="fill_missing"):
            method = st.selectbox("Fill method", ["mean", "median", "frequency", "value"], key="fill_method")
            columns = st.multiselect("Columns to fill NA", temp_df.columns, key="fill_col")
            value = None
            if method == "value":
                value = st.text_input("Enter value to fill", key="fill_value")
            if columns:
                temp_df = fill_missing(temp_df, columns, method=method, value=value)
    with pre_col2:
        st.write("#### Column Management")
        if st.checkbox("Rename columns", key="rename_cols"):
            old = st.selectbox("Column to rename", temp_df.columns, key="rename_col")
            new = st.text_input("New name", key="new_col_name")
            if new:
                temp_df = rename_column(temp_df, old, new)
        if st.checkbox("Drop columns", key="drop_col"):
            cols = st.multiselect("Columns to drop", temp_df.columns, key="cols_to_drop")
            temp_df = drop_columns(temp_df, cols)
        if st.checkbox("Drop duplicate columns", key="drop_dup"):
            temp_df = drop_duplicates(temp_df)

    st.write("\n")
    st.write("#### Transformation & Encoding")
    pre_col3, pre_col4 = st.columns(2)
    with pre_col3:
        if st.checkbox("Scale numeric columns", key="scale_num"):
            method = st.radio("Scaling method", ["minmax", "standard"], key="scale_method")
            temp_df = scale_data(temp_df, method)
        if st.checkbox("One-hot encode categorical columns", key="one_hot"):
            cat_cols = temp_df.select_dtypes(include=["object", "category"]).columns
            cols = st.multiselect("Columns to encode", cat_cols, key="cols_to_encode")
            temp_df = one_hot_encode(temp_df, cols)
        if st.checkbox("Aggregate data", key="aggregate"):
            group_col = st.selectbox("Group by", temp_df.columns, key="grp_by")
            agg_func = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"], key="agg_func")
            temp_df = aggregate_data(temp_df, group_col, agg_func)
    with pre_col4:
        if st.checkbox("Apply Log Transformation", key="log_transform"):
            numeric_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
            cols_to_log = st.multiselect("Select columns for log transform", [c for c in numeric_cols if (temp_df[c] >= 0).all()], key="log_cols")
            if cols_to_log:
                temp_df = log_transform(temp_df, cols_to_log)
        if st.checkbox("Convert column data type", key="convert_col"):
            col_to_convert = st.selectbox("Select column to convert", temp_df.columns, key="convert_col")
            target_type = st.selectbox("Convert to", ["datetime", "category", "numeric"], key="target_type")
            if col_to_convert and target_type:
                temp_df = convert_column(temp_df, col_to_convert, target_type)
        if st.checkbox("Extract datetime features", key="extract_dt"):
            datetime_cols = [col for col in temp_df.columns if pd.api.types.is_datetime64_any_dtype(temp_df[col])]
            if not datetime_cols:
                st.info("No datetime columns found. Convert a column's data type first.")
            else:
                col_to_extract = st.selectbox("Select datetime column", datetime_cols, key="extract_col")
                if col_to_extract:
                    temp_df = extract_datetime_features(temp_df, col_to_extract)

    st.write("\n")
    if st.button("Apply Preprocessing Steps"):
        st.session_state.working_df = temp_df
        st.success("Preprocessing steps applied successfully!")
        st.rerun()

    st.markdown("---")

    st.markdown(f"**Continue Your Analysis**")
    st.write("With your data prepared, advance to the next stage of your workflow.")
    col_option1, col_option2, col_option3= st.columns(3)
    with col_option1:
        if st.button("📊 Visualise Data"):
            st.switch_page("pages/4_data_visualisation.py")
    with col_option2:
        if st.button("📉 Statistical Analysis"):
            st.switch_page("pages/5_statistical_analysis.py")
    with col_option3:
        if st.button("🤖 Machine Learning"):
            st.switch_page("pages/6_ML.py")

if __name__ == "__main__":
	main()
