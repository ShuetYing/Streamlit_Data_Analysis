import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

@st.cache_data
def load_sample_dataset(dataset_name):
    '''load 4 sample datasets'''
    datasets = {
        "Iris": {
            "loader": lambda: load_iris(),
            "description": """
            **Iris Dataset**  
            - **Samples**: 150  
            - **Features**: 4 (sepal length, sepal width, petal length, petal width)  
            - **Target**: 3 iris species (setosa, versicolor, virginica)  
            - **Task**: Classification  
            - **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
            """,
            "preprocess": lambda data: pd.DataFrame(data.data, columns=data.feature_names)
                                .assign(species=[data.target_names[i] for i in data.target])
        },
        "Tips": {
            "loader": lambda: sns.load_dataset('tips'),
            "description": """
            **Tips Dataset**  
            - **Samples**: 244  
            - **Features**: 7 (total_bill, tip, sex, smoker, day, time, size)  
            - **Task**: Regression/EDA  
            - **Source**: Seaborn built-in dataset
            """,
            "preprocess": lambda data: data
        },
        "Diamonds": {
            "loader": lambda: sns.load_dataset('diamonds'),
            "description": """
            **Diamonds Dataset**  
            - **Samples**: 53,940  
            - **Features**: 10 (carat, cut, color, clarity, etc.)  
            - **Task**: Regression/EDA  
            - **Source**: Seaborn built-in dataset
            """,
            "preprocess": lambda data: data.sample(5000) if st.session_state.get('full_dataset') else data.sample(1000)
        },
        "Wine": {
            "loader": lambda: load_wine(),
            "description": """
            **Wine Dataset**  
            - **Samples**: 178  
            - **Features**: 13 (alcohol, malic_acid, ash, etc.)  
            - **Target**: 3 wine cultivars  
            - **Task**: Classification  
            - **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine)
            """,
            "preprocess": lambda data: pd.DataFrame(data.data, columns=data.feature_names)
                                .assign(wine_class=[data.target_names[i] for i in data.target])
        }
    }
    
    dataset = datasets[dataset_name]
    data = dataset["loader"]()
    df = dataset["preprocess"](data)
    return df, dataset["description"]

def show_data_preview(df):
    '''data preview'''
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total Rows", df.shape[0])
        st.metric("Total Columns", df.shape[1])
        
        if df.shape[0] > 1000:
            st.checkbox("Load full dataset", 
                      key="full_dataset",
                      help="Warning: Large datasets may impact performance")
    
    with col2:
        preview_tab, stats_tab = st.tabs(["Preview", "Statistics"])
        
        with preview_tab:
            num_rows = st.slider("Rows to show", 5, 50, 10)
            st.dataframe(df.head(num_rows), use_container_width=True, hide_index=True)
            
        with stats_tab:
            st.write("**Descriptive Statistics**")
            st.dataframe(df.describe(), use_container_width=True)

def create_matplotlib_plot(df, x_col, y_col, hue_col=None):
    '''create scatter plot for quick visulisation'''
    plt.figure(figsize=(10, 6))
    if hue_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.tight_layout()
    return plt

def main():
    st.set_page_config(page_title="Sample Datasets", layout="wide")
    st.header("Sample Datasets")
    st.caption("Explore and analyse these built-in datasets")
    st.markdown("---")

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = "Iris"

    sample_dataset_names = ["Iris", "Tips", "Diamonds", "Wine"]
    current_index = sample_dataset_names.index(st.session_state.selected_sample)

    # choose dataset
    dataset_option = st.selectbox(
        "Choose a dataset:",
        sample_dataset_names,
        index=current_index,
        help="Select from popular sample datasets",
        key="dataset_selectbox" 
    )

    df, description = load_sample_dataset(dataset_option)

    st.session_state.initial_df = df
    st.session_state.working_df = df

    if dataset_option != st.session_state.selected_sample:
        st.session_state.selected_sample = dataset_option
        st.rerun()

    with st.expander("📋 Dataset Information", expanded=True):
        st.markdown(description)
        st.download_button(
            label="Download Dataset",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"{dataset_option.lower()}_dataset.csv",
            mime="text/csv"
        )

    st.subheader("🔍 Data Exploration")
    show_data_preview(df)

    if st.checkbox("Show quick visualisation", False):
        plt.style.use('dark_background')
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            cols = st.columns(2)
            with cols[0]:
                x_axis = st.selectbox("X-axis", numeric_cols, index=0)
                y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
            with cols[1]:
                hue_options = [None] + [col for col in df.columns if df[col].nunique() < 10]
                hue_col = st.selectbox("Color by (categorical)", hue_options)
            
            fig = create_matplotlib_plot(df, x_axis, y_axis, hue_col)
            st.pyplot(fig)

    st.divider()
    st.markdown(f"**Continue Your Analysis**")
    col_option1, col_option2, col_option3, col_option4 = st.columns(4)
    with col_option1:			
        if st.button("🔍 Explore Data"):
            st.switch_page("pages/3_data_analysis.py")
    with col_option2:
        if st.button("📊 Visualise Data"):
            st.switch_page("pages/4_data_visualisation.py")
    with col_option3:
        if st.button("📉 Statistical Analysis"):
            st.switch_page("pages/5_statistical_analysis.py")
    with col_option4:
        if st.button("🤖 Machine Learning"):
            st.switch_page("pages/6_ML.py")

if __name__ == "__main__":
    main()