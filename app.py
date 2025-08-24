import streamlit as st

st.set_page_config(
	page_title="Homepage - Data Analysis Tools", 
	layout="wide",
	initial_sidebar_state="collapsed"
)

# navigation bar
pages = {
	"Home": [
		st.Page("pages/1_home_page.py", title="Overview of features")
 	],
	"Data Upload": [
		st.Page("pages/2_upload_data.py", title="Upload file"),
		st.Page("pages/example_dataset.py", title="Sample datasets")
	],
	"Exploratory Analysis": [
		st.Page("pages/3_data_analysis.py", title="Explore data"),
		st.Page("pages/4_data_visualisation.py", title="Visualise data"),
		st.Page("pages/5_statistical_analysis.py", title="Statistical analysis")
	],
	"Machine Learning": [
		st.Page("pages/6_ML.py", title="Machine Learning models"),
        st.Page("pages/7_reduction.py", title="Dimensionality Reduction")
	]
}

pg = st.navigation(pages, position="top")
pg.run()

