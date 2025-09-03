import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import mean_squared_error
import umap.umap_ as umap 
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def get_image_download_link(fig, filename, text):
    '''allow user to download and download plot'''
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(text, buf, file_name=filename, mime="image/png")

def main():
    st.set_page_config(page_title="Dimensionality Reduction", layout="wide")
    plt.style.use('dark_background')

    st.header("Dimensionality Reduction")
    st.caption("Visualise your high-dimensional data in 2D or 3D using techniques like PCA, t-SNE, and UMAP. This is for visual exploration and does not affect the data used for model training.")
    st.markdown("---")

    # description
    st.subheader("Dimensionality Reduction Techniques")
    st.markdown("""
                ***Principal Component Analysis (PCA)***\n
                **What it is:** A linear technique that finds the directions of maximum variance in data and projects the data onto a new, smaller set of dimensions called 'principal components'\n
                **Good at:**
                - Computationally efficient and scales well to very large datasets
                - Ideal for a quick, first look at the data to understand its major axes of variation
                - General-purpose dimensionality reduction and great for noise reduction
                """)
    st.markdown("""
                ***t-Distributed Stochastic Neighbor Embedding (t-SNE)***\n
                **What it is:** A non-linear technique that excels at visualising high-dimensional data which focuses on preserving the local structure\n
                **Good at:**
                - Revealing intricate, non-linear cluster structures
                - Ensure that similar data points are grouped closely in the final plot
                - Visualise complex, high-dimensional data like gene expression profiles, image features, or word embedding
                """)
    st.markdown("""
                ***Uniform Manifold Approximation and Projection (UMAP)***\n
                **What it is:** A non-linear technique that is often seen as an evolution of t-SNE, offering a better balance of speed, accuracy and better preserve glocal structure of data\n
                **Good at:**
                - Significantly faster than t-SNE and scales better to large datasets
                - Preserve both the local and global structure of the data, which can lead to more interpretable visualisations
                - General-purpose embedding which can be used for out-of-sample data
                """)

    st.markdown("---")

    if "working_df" not in st.session_state or st.session_state.working_df is None:
        st.warning("No data found. Please upload a dataset or use sample dataset on the 'Data Upload' page.")
        if st.button("Go to Upload Data", type='primary'):
            st.switch_page("pages/2_upload_data.py")
        return
    
    df = st.session_state.working_df

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols:
        st.error("This technique requires at least one numeric column in the dataset.")
        return
    
    config_col, plot_col = st.columns([1, 2])
    with config_col:
        st.markdown("#### Configuration")
        features = st.multiselect("Select features to reduce", numeric_cols, default=numeric_cols, key="redux_features")
        
        if features:
            max_components = len(features)
            # select method
            method = st.selectbox("Method", ["PCA", "t-SNE", "UMAP"], key="redux_method")
            if method == "PCA":
                n_components = st.slider("Number of Components", 2, max_components, min(2, max_components), key="redux_components")
            elif method == "t-SNE":
                n_components = st.slider("Number of Components", 2, 3, 2, key="redux_components")
                perplexity = st.slider("Perplexity", 5.0, 50.0, 30.0, help="The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.", key="redux_perplexity")
            else:
                n_components = st.slider("Number of Components", 2, max_components, min(2, max_components), key="redux_components")
                n_neighbors = st.slider("Number of Neighbors", 2, 100, 15, help="Ccntrols how UMAP balances local versus global structure in the data", key="redux_neighbors")
                min_dist = st.slider("Minimum Dostance", 0.0, 0.99, 0.1, help="Controls how tightly UMAP is allowed to pack points together", key="redux_distance")
            
            if st.button("Run Reduction & Visualise", use_container_width=True):
                X_redux = df[features]
                X_redux_scaled = StandardScaler().fit_transform(X_redux)
                
                if method == "PCA": 
                    reducer = PCA(n_components=n_components)
                    st.session_state.reducer_type = "PCA"
                elif method == "t-SNE": 
                    reducer = TSNE(n_components=n_components, random_state=42, init='pca', perplexity=perplexity, learning_rate='auto')
                    st.session_state.reducer_type = "t-SNE"
                else: 
                    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                    st.session_state.reducer_type = "UMAP"
                
                with st.spinner(f"Running {method}..."):
                    X_reduced = reducer.fit_transform(X_redux_scaled)
                
                st.session_state.reduced_data = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(n_components)])
                
                if method == "PCA":
                    st.session_state.explained_variance = reducer.explained_variance_ratio_
                    X_reconstructed = reducer.inverse_transform(X_reduced)
                    st.session_state.reconstruction_error = mean_squared_error(X_redux_scaled, X_reconstructed)
                    if 'preservation_score' in st.session_state:
                        del st.session_state['preservation_score']
                else:
                    if 'explained_variance' in st.session_state:
                        del st.session_state['explained_variance']
                    if 'reconstruction_error' in st.session_state:
                        del st.session_state['reconstruction_error']
                    st.session_state.preservation_score = trustworthiness(X_redux_scaled, X_reduced)

    with plot_col:
        if 'reduced_data' in st.session_state:
            st.markdown("#### Reduction Results")
            
            with st.container(border=True):
                if st.session_state.reducer_type == "PCA" and 'explained_variance' in st.session_state:
                    st.write("**Explained Variance Ratio (PCA):**")
                    variance_df = pd.Series(st.session_state.explained_variance, index=[f"Component {i+1}" for i in range(len(st.session_state.explained_variance))])
                    st.dataframe(variance_df)
                    st.metric("Total Variance Explained", f"{variance_df.sum():.2%}")
                    if 'reconstruction_error' in st.session_state:
                        st.metric("Reconstruction Error (MSE)", f"{st.session_state.reconstruction_error:.4f}")

                if 'preservation_score' in st.session_state:
                    st.metric("Trustworthiness Score", f"{st.session_state.preservation_score:.3f}")
                    st.caption("Measures how well the local structure is maintained (0 to 1). Higher is better.")

            reduced_cols = list(st.session_state.reduced_data.columns)
            color_options = [None] + categorical_cols + reduced_cols
            color_by = st.selectbox("Color points by (optional)", color_options, key="redux_color")
            
            hue_data = None
            if color_by:
                if color_by in categorical_cols:
                    hue_data = df[color_by].reset_index(drop=True)
                elif color_by in reduced_cols:
                    hue_data = st.session_state.reduced_data[color_by]

            fig = plt.figure(figsize=(10, 8))
            
            if st.session_state.reduced_data.shape[1] >= 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=20., azim=45)
                
                c_values = None
                if hue_data is not None:
                    if pd.api.types.is_categorical_dtype(hue_data) or pd.api.types.is_object_dtype(hue_data):
                        c_values = hue_data.astype('category').cat.codes
                    else:
                        c_values = hue_data
                
                scatter = ax.scatter(
                    st.session_state.reduced_data['Component 1'], 
                    st.session_state.reduced_data['Component 2'], 
                    st.session_state.reduced_data['Component 3'], 
                    c=c_values,
                    cmap='viridis' if c_values is not None else None,
                    s=30
                )
                ax.set_zlabel('Component 3')
            else:
                ax = fig.add_subplot(111)
                sns.scatterplot(data=st.session_state.reduced_data, x='Component 1', y='Component 2', hue=hue_data, palette='viridis', ax=ax)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            st.pyplot(fig)
            get_image_download_link(fig, "dimensionality_reduction.png", "Download Plot")



if __name__ == "__main__":
    main()