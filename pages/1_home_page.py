import streamlit as st

st.set_page_config(
    page_title="Homepage - Data Analysis Tools",
    layout="wide"
)

st.title("Your All-in-One Data Analysis Toolkit")
st.subheader("From Raw Data to Insightful Visualisations and Machine Learning Models")
st.write("""
This application is a comprehensive suite of tools designed to streamline your data analysis workflow. 
Whether you're exploring a new dataset, creating compelling visualisations, or building predictive models, 
you'll find everything you need right here.
""")

st.markdown("---")

st.header("How to Get Started")
st.write("Follow these simple steps to begin your analysis journey:")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("##### Step 1: Load Your Data", icon=":material/upload:")
    st.write("Navigate to the **Data Upload** section. You can paste raw text, upload a file (CSV, Excel, JSON), connect to a database, or simply use one of our sample datasets to get a feel for the app.")
    upload_bt, sample_bt = st.columns(2)
    with upload_bt:
        if st.button("Go to Data Upload", use_container_width=True):
            st.switch_page("pages/2_upload_data.py")
    with sample_bt:
        if st.button("Use Sample Datasets", use_container_width=True):
            st.switch_page("pages/example_dataset.py")

with col2:
    st.info("##### Step 2: Analyse & Visualise", icon=":material/search_insights:")
    st.write("Once your data is loaded, head over to the **Exploratory Analysis** section. Clean your data, view summary statistics, and build a wide range of customisable charts to uncover hidden patterns.")
    if st.button("Explore Your Data", use_container_width=True):
        st.switch_page("pages/3_data_analysis.py")

with col3:
    st.info("##### Step 3: Apply Machine Learning", icon=":material/network_intelligence:")
    st.write("Ready to make predictions? Go to the **Machine Learning** page to train various models on your dataset and evaluate their performance.")
    if st.button("Build ML Models", use_container_width=True):
        st.switch_page("pages/6_ML.py")

st.markdown("---")

st.header("Key Features")
feature_cols = st.columns(5)

with feature_cols[0]:
    st.info("**Data Connectivity**", icon=":material/data_table:")
    st.write("- Upload CSV, TSV, TXT, Excel, and JSON files")
    st.write("- Paste data directly from your clipboard")
    st.write("- Connect to SQL databases (MySQL, PostgreSQL, etc.)")
    st.write("- Explore pre-loaded sample datasets")

with feature_cols[1]:
    st.info("**Data Preprocessing**", icon=":material/cleaning_services:")
    st.write("- View and filter your data interactively")
    st.write("- Handle missing values with various strategies")
    st.write("- Scale numerical features and one-hot encode categories")
    st.write("- Aggregate data for summary views")

with feature_cols[2]:
    st.info("**Dynamic Visualisations**", icon=":material/add_chart:")
    st.write("- Generate a wide array of plots: Line, Bar, Scatter, Histogram, Pie, and more")
    st.write("- Customise charts with titles, labels, colors, and styles")
    st.write("- Download your plots as high-quality images")

with feature_cols[3]:
    st.info("**Statistical Analysis**", icon=":material/analytics:")
    st.write("- Check the basics: Test for normality and homogeneity of variance")
    st.write("- Various statistical parametric and non-parametric tests")
    st.write("- Measure relationships with correlation analysis")

with feature_cols[4]:
    st.info("**Machine Learning**", icon=":material/model_training:")
    st.write("- Train classification and regression models")
    st.write("- Tune model hyperparameters")
    st.write("- Evaluate model performance with key metrics")
    st.write("- Reduce the dimensions of the data")

st.markdown("---")
st.write("Ready to dive in? Select a page from the navigation bar at the top to begin!")
