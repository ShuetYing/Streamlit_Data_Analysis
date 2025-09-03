import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error, silhouette_score, mean_absolute_error, davies_bouldin_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def get_image_download_link(fig, filename, text):
    '''allow user to download and save plot'''
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(text, buf, file_name=filename, mime="image/png")

def plot_confusion_matrix(y_true, y_pred, classes):
    '''plot confusion matrix'''
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    get_image_download_link(fig, "confusion_matrix.png", "Download Confusion Matrix")

def plot_feature_importance(model, feature_names):
    '''plot feature importance'''
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Feature Importances")
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, "feature_importance.png", "Download Feature Importance Bar Plot")
    else:
        st.info("This model type does not provide feature importances.")

def plot_regression_results(y_test, y_pred):
    '''regression plot'''
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.8)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5, color="red")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs. Actual Values")
    st.pyplot(fig)
    get_image_download_link(fig, "regression_plot.png", "Download Regression Plot")

def plot_clusters(df, features, cluster_labels):
    '''plot data points in cluster'''
    st.write("### Cluster Visualisation")
    col1, col2 = st.columns(2)
    x_axis = col1.selectbox("Select X-axis for plot", options=features, index=0, key="cluster_x_axis")
    y_axis = col2.selectbox("Select Y-axis for plot", options=features, index=1 if len(features) > 1 else 0, key="cluster_y_axis")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=cluster_labels, palette='viridis', ax=ax, legend='full')
    ax.set_title('Cluster Distribution')
    st.pyplot(fig)
    get_image_download_link(fig, "cluster_distribution.png", "Download Cluster Scatter Plot")

def plot_elbow_method(data):
    '''elbow plot to determine k'''
    st.write("#### Elbow Method for Optimal K")
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_range, inertias, marker='o')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    st.pyplot(fig)
    get_image_download_link(fig, "elbow_plot.png", "Download Elbow Plot")
    st.info("The 'elbow' in the plot (the point where the rate of decrease sharply changes) suggests the optimal number of clusters.")

def get_classification_params(model_name, location):
    '''allow user to do hyperparameter tuning for classification models'''
    params = {}
    if model_name == "Logistic Regression":
        params['penalty'] = location.selectbox("Penalty", ['l2', 'l1', 'elasticnet', 'None'], 
                            help="Regularization technique to prevent model from overfitting. L2 penalty: Ridge regression & L1 penalty: Lasso regression & Elastic Net penalty: hybrid that combines L1 and L2", 
                            key="lr_penalty")
        params['C'] = location.slider("C (Regularization)", 0.01, 10.0, 1.0, 
                        help="Inverse of regularization strength",
                        key="lr_c")
        params['solver'] = location.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 
                            help="Algorithm to use in the optimisation problem",
                            key="lr_solver")
    elif model_name == "Random Forest":
        params['n_estimators'] = location.slider("Number of Estimators", 10, 500, 100, 
                                    help="The number of trees in the forest", 
                                    key="rf_clf_estimators")
        params['criterion'] = location.selectbox("Criterion", ['gini', 'entropy', 'log_loss'],
                                help="The function to measure the quality of a split", 
                                key="rf_clf_criterion")
        params['max_depth'] = location.slider("Max Depth", 2, 50, 10, 
                                help="The maximum depth of the tree", 
                                key="rf_clf_depth")
        params['min_impurity_decrease'] = location.slider("Minimum impurity decrease", 0.0, 50.0, 0.0, 
                                            help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                                            key="rf_clf_min_impurity")
    elif model_name == "k-Nearest Neighbors":
        params['n_neighbors'] = location.slider("Number of Neighbors (k)", 1, 50, 5, 
                                help="Number of neighbors to use", 
                                key="knn_clf_k")
        params['weights'] = location.selectbox("Weight", ['uniform', 'distance'], 
                            help="Weight function used in prediction. Uniform: All points in each neighborhood are weighted equally & Distance: closer neighbors of a query point will have a greater influence than neighbors which are further away",
                            key="knn_clf_weight")
        params['algorithm'] = location.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                help="Algorithm used to compute the nearest neighbors",
                                key="knn_clf_algorithm")
    elif model_name == "Support Vector Machine":
        params['C'] = location.slider("C (Regularization)", 0.01, 10.0, 1.0, 
                        help="Regularization parameter. The strength of the regularization is inversely proportional to C. ",
                        key="svc_c")
        params['kernel'] = location.selectbox("Kernel", ['rbf', 'linear', 'sigmoid', 'precomputed', 'poly'], 
                            help="Specifies the kernel type to be used in the algorithm", 
                            key="svc_kernel")
    return params

def get_regression_params(model_name, location):
    '''allow user to do hyperparameter tuning for regression models'''
    params = {}
    if model_name == "Random Forest Regressot":
        params['n_estimators'] = location.slider("Number of Estimators", 10, 500, 100, 
                                    help="The number of trees in the forest", 
                                    key="rf_reg_estimators")
        params['max_depth'] = location.slider("Max Depth", 2, 50, 10, 
                                help="The maximum depth of the tree", 
                                key="rf_reg_depth")
        params['criterion'] = location.selectbox("Criterion", ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                                help="The function to measure the quality of a split", 
                                key="rf_reg_criterion")
        params['min_impurity_decrease'] = location.slide("Minimum impurity decrease", 0.0, 50.0, 0.0,
                                            help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value",
                                            key="rf_reg_min_impurity")
    elif model_name == "KNeighborsRegressor":
        params['n_neighbors'] = location.slider("Number of Neighbors (k)", 1, 50, 5, 
                                help="Number of neighbors to use", 
                                key="knn_reg_k")
        params['weights'] = location.selectbox("Weight", ['uniform', 'distance'], 
                            help="Weight function used in prediction. Uniform: All points in each neighborhood are weighted equally & Distance: closer neighbors of a query point will have a greater influence than neighbors which are further away",
                            key="knn_reg_weight")
        params['algorithm'] = location.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                help="Algorithm used to compute the nearest neighbors",
                                key="knn_reg_algorithm")
    elif model_name == "Support Vector Regression":
        params['C'] = location.slider("C (Regularization)", 0.01, 10.0, 1.0, 
                        help="Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.", 
                        key="svr_c")
        params['kernel'] = location.selectbox("Kernel", ['rbf', 'linear', 'sigmoid', 'precomputed', 'poly'], 
                            help="Specifies the kernel type to be used in the algorithm", 
                            key="svr_kernel")
    return params

def get_clustering_params(model_name, location):
    '''allow user to do hyperparameter tuning for clustering models'''
    params = {}
    if model_name == "K-Means":
        params['n_clusters'] = location.slider("Number of Clusters (k)", 2, 20, 3, 
                                help="The number of clusters to form as well as the number of centroids to generate", 
                                key="kmeans_k")
        params['algorithm'] = location.selectbox("Algorithm", ['lloyd', 'elkan'], 
                                help="K-means algorithm to use.",
                                key="kmeans_algorithm")
    elif model_name == "DBSCAN":
        params['eps'] = location.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 
                        help="The maximum distance between two samples for one to be considered as in the neighborhood of the other", 
                        key="dbscan_eps")
        params['min_samples'] = location.slider("Minimum Samples", 1, 20, 5, 
                                help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point", 
                                key="dbscan_min_samples")
    return params

def main():
    st.set_page_config(page_title="Machine Learning", layout="wide")
    plt.style.use('dark_background')

    st.header("Machine Learning")
    st.caption("""
        **Build, train, and evaluate predictive models.** Leverage the insights from your exploratory and statistical analysis to select features and algorithms that will uncover patterns and make accurate predictions on new data.
    """)
    st.caption("""
        **The model building workflow:**
        - **Objective Definition:** Choose between regression, classification, or clustering.
        - **Algorithm Selection:** Pick from a suite of supervised and unsupervised learning models.
        - **Training & Tuning:** Fit models and optimize hyperparameters for peak performance.
        - **Evaluation:** Compare models using robust metrics and validation techniques.
        - **Prediction:** Generate forecasts on new data or export the model for deployment.         
    """)

    st.markdown("---")

    st.subheader("Model Description")
    st.markdown("**Types of Machine Learning**")
    st.markdown("- **Classification**: A supervised learning task to predict a categorical label")
    st.markdown("- **Regression**: A supervised learning task to predict a continuous numerical value")
    st.markdown("- **Clustering**: An unsupervised learning task to discover natural groupings in your data without pre-existing labels")

    model_description = st.selectbox("Model Description", ['Logistic Regression', 'Linear Regression', 'Random Forest', 'k-Nearest Neighbors', 'Support Vector Machine', 'K-Means', 'DBSCAN'], placeholder="Select a model to see its description.", index=None)

    # short description for different model
    if model_description:
        with st.container(border=True):
            if model_description == "Logistic Regression":
                st.markdown("""
                            **Type:** Classification\n
                            **What it is:** A fundamental and interpretable algorithm that predicts the probability of a binary outcome\n
                            **Good at:**
                            - Binary classification problems
                            - When you need a simple, fast, and explainable model
                            - Establishing a baseline for classification tasks
                """)
            if model_description == "Linear Regression":
                st.markdown("""
                            **Type:** Regression\n
                            **What it is:** The foundational regression algorithm that models the linear relationship between a dependent variable and one or more independent variables and predict targets by linear approximation\n
                            **Good at:**
                            - Predicting continuous numerical values
                            - When the relationship between variables is believed to be linear
                            - Understanding the impact of each feature on the outcome
                """)
            if model_description == "Random Forest":
                st.markdown("""
                            **Type:** Classification & Regression\n
                            **What it is:** An ensemble model that builds multiple decision trees from samples drawn with replacement from dataset and merges their predictions to get a more accurate and stable result\n
                            **Good at:**
                            - High accuracy on a wide range of tasks
                            - Handling complex, non-linear relationships
                            - Reducing the risk of overfitting
                """)
            if model_description == "k-Nearest Neighbors":
                st.markdown("""
                            **Type:** Classification & Regression\n
                            **What it is:** A simple, instance-based algorithm that classifies a new data point based on the majority class or average value of its 'k' closest neighbors\n
                            **Good at:**
                            - Simple, intuitive classification tasks
                            - Problems where the decision boundary is irregular
                            - When you have a smaller dataset and need a quick, easy-to-implement model
                """)
            if model_description == "Support Vector Machine":
                st.markdown("""
                            **Type:** Classification & Regression\n
                            **What it is:** A powerful model that finds the optimal boundary (hyperplane) to separate classes or fit a regression line\n
                            **Good at:**
                            - High-dimensional spaces (many features)
                            - Problems where number of dimensions is greater than the number of samples
                            - Both linear and complex, non-linear problems (using different kernels)
                """)
            if model_description == "K-Means":
                st.markdown("""
                            **Type:** Clustering\n
                            **What it is:** An unsupervised algorithm that groups data of equal variance into a pre-specified number of clusters (k) based on feature similarity\n
                            **Good at:**
                            - Large numbers of samples
                            - Finding distinct, spherical groups in data
                            - When you have an idea of how many clusters to expect
                """)
            if model_description == "DBSCAN":
                st.markdown("""
                            **Type:** Clustering\n
                            **What it is:** A density-based clustering algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions\n
                            **Good at:**
                            - Identifying arbitrarily shaped clusters
                            - Outlier detection (points labeled as noise)
                            - When the number of clusters is not known beforehand
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

    config_col, results_col = st.columns([1, 2])

    with config_col:
        st.subheader("Model Configuration")
        
        tab1, tab2, tab3 = st.tabs(["⚙️ Task & Data", "🧠 Algorithm", "🔧 Hyperparameters"])

        with tab1:

            def on_config_change():
                keys_to_delete = ['trained_model', 'cluster_results', 'classification_results', 'regression_results']
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]

            ml_task = st.selectbox("Select ML Task", ["Classification", "Regression", "Clustering"], on_change=on_config_change, key="ml_task")

            if ml_task == "Regression":
                target_column = st.selectbox("Select Target Variable", df.columns, on_change=on_config_change, key="target_col")
                feature_columns = st.multiselect("Select Feature Columns", 
                                                 [col for col in df.columns if col != target_column],
                                                 default=[col for col in df.columns if col != target_column],
                                                 on_change=on_config_change, key="feature_cols")
                use_kfold = st.checkbox("Use K-Fold Cross-Validation", key="kfold_check")
                k_folds = st.number_input("Number of Folds (k)", min_value=2, max_value=20, value=5, step=1) if use_kfold else 1
                
            elif ml_task == "Classification":
                target_column = st.selectbox("Select Target Variable", categorical_cols, key="target_col")
                feature_columns = st.multiselect("Select Feature Columns", 
                                                 [col for col in df.columns if col != target_column],
                                                 default=[col for col in df.columns if col != target_column],
                                                 on_change=on_config_change, key="feature_cols")
                use_kfold = st.checkbox("Use K-Fold Cross-Validation", key="kfold_check")
                k_folds = st.number_input("Number of Folds (k)", min_value=2, max_value=20, value=5, step=1) if use_kfold else 1

            else:
                target_column = None
                feature_columns = st.multiselect("Select Features for Clustering",
                                                 numeric_cols,
                                                 default=numeric_cols,
                                                 key="cluster_feature_cols")
                use_kfold = False

        with tab2:
            if ml_task == "Classification":
                model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "k-Nearest Neighbors", "Support Vector Machine"], on_change=on_config_change, key="clf_model_name")
            elif ml_task == "Regression":
                model_name = st.selectbox("Select Model", ["Linear Regression", "Random Forest Regressor", "Nearest Neighbors Regression", "Support Vector Regression"], on_change=on_config_change, key="reg_model_name")
            else:
                model_name = st.selectbox("Select Model", ["K-Means", "DBSCAN"], on_change=on_config_change, key="clu_model_name")

        with tab3:
            st.write("Adjust the model's parameters below.")
            if ml_task == "Classification":
                params = get_classification_params(model_name, st)
            elif ml_task == "Regression":
                params = get_regression_params(model_name, st)
            else:
                params = get_clustering_params(model_name, st)

        if st.button("Train Model", use_container_width=True):

            on_config_change()

            if not feature_columns:
                st.error("Please select at least one feature column.")
                return

            X = df[feature_columns]
            y = df[target_column] if target_column else None

            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

            preprocessor = ColumnTransformer(transformers=[
                ('num', StandardScaler(), [c for c in feature_columns if c in numeric_features]),
                ('cat', OneHotEncoder(handle_unknown='ignore'), [c for c in feature_columns if c in categorical_features])])
            
            model_classes = {
                "Classification": {
                    "Logistic Regression": LogisticRegression,
                    "Random Forest": RandomForestClassifier,
                    "k-Nearest Neighbors": KNeighborsClassifier,
                    "Support Vector Machine": SVC
                },
                "Regression": {
                    "Linear Regression": LinearRegression,
                    "Random Forest Regressor": RandomForestRegressor,
                    "Nearest Neighbors Regression": KNeighborsRegressor,
                    "Support Vector Regression": SVR
                },
                "Clustering": {
                    "K-Means": KMeans,
                    "DBSCAN": DBSCAN
                }
            }

            ModelClass = model_classes[ml_task][model_name]

            try:
                if model_name in ["Linear Regression", "k-Nearest Neighbors", "Nearest Neighbors Regression", "Support Vector Regression", "DBSCAN"]:
                     model = ModelClass(**params)
                else:
                     model = ModelClass(**params, random_state=42)
            except TypeError as e:
                st.error(f"Error initialising model {model_name}: {e}")
                return

            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            if ml_task in ["Classification", "Regression"]:
                if use_kfold:
                    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    scoring = 'accuracy' if ml_task == "Classification" else 'r2'
                    scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=scoring)
                    st.session_state.classification_results = {'kfold_scores': scores, 'model_name': model_name, 'ml_task': ml_task} if ml_task == "Classification" else None
                    st.session_state.regression_results = {'kfold_scores': scores, 'model_name': model_name, 'ml_task': ml_task} if ml_task == "Regression" else None
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    pipeline.fit(X_train, y_train)
                    st.session_state.trained_model = pipeline
                    if ml_task == "Classification":
                        st.session_state.classification_results = {'y_test': y_test, 
                                                                   'y_pred': pipeline.predict(X_test), 
                                                                   'classes': y.unique(), 
                                                                   'model_name': model_name, 
                                                                   'ml_task': ml_task, 
                                                                   'pipeline': pipeline}
                    else:
                        st.session_state.regression_results = {'y_test': y_test, 
                                                               'y_pred': pipeline.predict(X_test), 
                                                               'model_name': model_name, 
                                                               'ml_task': ml_task, 
                                                               'pipeline': pipeline}
                
            else:
                pipeline.fit(X)
                st.session_state.trained_model = pipeline
                st.session_state.feature_columns = X.columns.tolist()
                
                if hasattr(model, 'labels_'):
                    st.session_state.cluster_labels = model.labels_
                    st.session_state.cluster_features = X
                    st.session_state.cluster_results = {
                        'labels': model.labels_,
                        'features': X,
                        'model_name': model_name,
                        'ml_task': ml_task,
                        'pipeline': pipeline
                    }
            
    with results_col:
        if 'classification_results' in st.session_state and st.session_state.classification_results:
            res = st.session_state.classification_results
            st.subheader(f"Results for {res['model_name']} ({res['ml_task']})")
            if 'kfold_scores' in res:
                st.metric("Mean Accuracy (K-Fold)", f"{res['kfold_scores'].mean():.3f} (±{res['kfold_scores'].std():.3f})")
            else:
                res_col1, res_col2 = st.columns([1.2, 2])
                with res_col1:
                    st.write("#### Performance Metrics")
                    st.metric("Accuracy", f"{accuracy_score(res['y_test'], res['y_pred']):.3f}")
                    st.text("Classification Report")
                    st.code(classification_report(res['y_test'], res['y_pred'], zero_division=0))
                with res_col2:
                    plot_confusion_matrix(res['y_test'], res['y_pred'], res['classes'])
            
                st.write("---")
                st.write("#### Feature Analysis")
                feature_names = res['pipeline'].named_steps['preprocessor'].get_feature_names_out()
                plot_feature_importance(res['pipeline'].named_steps['model'], feature_names)

        if 'regression_results' in st.session_state and st.session_state.regression_results:
            res = st.session_state.regression_results
            st.subheader(f"Results for {res['model_name']} ({res['ml_task']})")
            if 'kfold_scores' in res:
                st.metric("Mean R-squared (K-Fold)", f"{res['kfold_scores'].mean():.3f} (±{res['kfold_scores'].std():.3f})")
            else:
                res_col1, res_col2 = st.columns([1.2, 2])
                with res_col1:
                    st.write("#### Performance Metrics")
                    st.metric("R-squared", f"{r2_score(res['y_test'], res['y_pred']):.3f}")
                    st.metric("Mean Absolute Error", f"{mean_absolute_error(res['y_test'], res['y_pred']):.3f}")
                    st.metric("Mean Squared Error", f"{mean_squared_error(res['y_test'], res['y_pred']):.3f}")
                with res_col2:
                    plot_regression_results(res['y_test'], res['y_pred'])

                st.write("---")
                st.write("#### Feature Analysis")
                feature_names = res['pipeline'].named_steps['preprocessor'].get_feature_names_out()
                plot_feature_importance(res['pipeline'].named_steps['model'], feature_names)

        if 'cluster_results' in st.session_state:
            res = st.session_state.cluster_results
            st.subheader(f"Results for {res['model_name']} ({res['ml_task']})")
            n_clusters = len(set(res['labels'])) - (1 if -1 in res['labels'] else 0)
            st.metric("Number of Clusters Found", n_clusters)

            X_transformed = res['pipeline'].named_steps['preprocessor'].transform(res['features']) if res['pipeline'].named_steps['preprocessor'] else res['features']
            if n_clusters > 1:
                st.metric("Silhouette Score", f"{silhouette_score(X_transformed, res['labels']):.3f}")
                st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(X_transformed, res['labels']):.3f}")

            true_labels_col = st.selectbox("Optional: Select true labels for Adjusted Rand Index", [None] + df.select_dtypes(include='category').columns.tolist())
            if true_labels_col:
                st.metric("Adjusted Rand Index", f"{adjusted_rand_score(df[true_labels_col], res['labels']):.3f}")

            if res['model_name'] == 'K-Means':
                plot_elbow_method(X_transformed)
            
            plot_clusters(res['features'], st.session_state.feature_columns, res['labels'])

        if 'trained_model' in st.session_state and not use_kfold:
            st.markdown("---")
            st.header("Make Predictions")
            if ml_task == "Clustering":
                st.info("Prediction is not applicable for clustering models. The model assigns labels to the training data.")

            else:
                pred_tab1, pred_tab2 = st.tabs(["Manual Input", "Upload File"])
                with pred_tab1:
                    input_data = {}
                    for col in st.session_state.trained_model.feature_names_in_:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()), key=f"pred_{col}")
                        else:
                            input_data[col] = st.selectbox(f"Select {col}", options=df[col].unique(), key=f"pred_{col}")
                    if st.button("Predict", use_container_width=True):
                        input_df = pd.DataFrame([input_data])
                        prediction = st.session_state.trained_model.predict(input_df)
                        st.success(f"**Prediction:** {prediction[0]}")
                with pred_tab2:
                    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])
                    if uploaded_file:
                        try:
                            pred_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                            required_cols = set(st.session_state.trained_model.feature_names_in_)
                            if not required_cols.issubset(pred_df.columns):
                                st.error(f"Missing columns: {', '.join(required_cols - set(pred_df.columns))}")
                            else:
                                pred_df_features = pred_df[st.session_state.trained_model.feature_names_in_]
                                predictions = st.session_state.trained_model.predict(pred_df_features)
                                pred_df['Prediction'] = predictions
                                st.dataframe(pred_df)
                                csv = pred_df.to_csv(index=False).encode('utf-8')
                                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv", use_container_width=True)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()