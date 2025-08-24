import streamlit as st
import graphviz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from io import BytesIO

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import lilliefors

st.set_page_config(page_title="Statistical Analysis", layout="wide")

def create_aesthetic_graph():
    graph_attrs = {
        'rankdir': 'TB',
        'bgcolor': 'transparent',
        'fontname': 'Helvetica',
        'fontcolor': '#FFFFFF',
        'fontsize': '18',
    }

    palette = {
        'blue': '#89CFF0',
        'teal': '#66C2A5',
        'green': '#8DA0CB',
        'purple': '#E78AC3',
        'yellow': '#A6D854',
        'red': '#FC8D62',
        'orange': '#FFD92F',
        'brown': '#E5C494',
    }

    base_node_style = {
        'style': 'filled',
        'shape': 'box',
        'fontname': 'Helvetica',
        'penwidth': '2.5',
    }

    decision_node_style = {
        **base_node_style,
        'shape': 'diamond',
        'style': 'filled, rounded',
        'fontcolor': '#000000',
        'fontsize': '18',
    }

    branch_node_style = {
        **base_node_style,
        'style': 'filled, rounded',
        'fillcolor': '#2D2D2D',
        'fontsize': '16',
    }

    outcome_node_style = {
        **base_node_style,
        'style': 'filled, rounded',
        'fillcolor': '#222222',
        'fontcolor': '#FFFFFF',
        'fontsize': '14',
    }
    
    graph = graphviz.Digraph(graph_attr=graph_attrs)

    graph.node('A', 'What is your main goal?', 
               **decision_node_style,
               fillcolor=palette['blue'])
    
    with graph.subgraph(name='cluster_B') as b:
        b.attr(color='transparent')
        b.edge_attr.update(color=palette['teal'], penwidth='2', fontcolor=palette['teal'])
        
        b.node('B', 'Compare group to\na known standard', **branch_node_style, fontcolor=palette['teal'], color=palette['teal'])
        b.node('B1', 'One-Sample T-Test (P)', **outcome_node_style, color=palette['teal'])
        
        graph.edge('A', 'B', color=palette['teal'])
        b.edge('B', 'B1', style='dashed', arrowhead='vee')

    with graph.subgraph(name='cluster_C') as c:
        c.attr(color='transparent')
        c.edge_attr.update(color=palette['green'], penwidth='2', fontcolor=palette['green'])
        
        c.node('C', 'Compare two groups', **branch_node_style, fontcolor=palette['green'], color=palette['green'])
        c.node('C1', 'Independent or Paired?', **decision_node_style, fillcolor=palette['green'])
        c.node('C2', 'Independent T-Test (P)\nMann-Whitney U (NP)', **outcome_node_style, color=palette['green'])
        c.node('C3', 'Paired T-Test (P)\nWilcoxon Signed-Rank (NP)', **outcome_node_style, color=palette['green'])
        
        graph.edge('A', 'C', color=palette['green'])
        c.edge('C', 'C1')
        c.edge('C1', 'C2', label='Independent', style='dashed', arrowhead='vee')
        c.edge('C1', 'C3', label='Paired', style='dashed', arrowhead='vee')

    with graph.subgraph(name='cluster_D') as d:
        d.attr(color='transparent')
        d.edge_attr.update(color=palette['purple'], penwidth='2', fontcolor=palette['purple'])
        
        d.node('D', 'Compare 3+ groups', **branch_node_style, fontcolor=palette['purple'], color=palette['purple'])
        d.node('D1', 'How many factors?', **decision_node_style, fillcolor=palette['purple'])
        d.node('D2', 'One-Way ANOVA (P)\nKruskal-Wallis (NP)', **outcome_node_style, color=palette['purple'])
        d.node('D3', 'Two-Way ANOVA (P)', **outcome_node_style, color=palette['purple'])

        graph.edge('A', 'D', color=palette['purple'])
        d.edge('D', 'D1')
        d.edge('D1', 'D2', label='One Factor', style='dashed', arrowhead='vee')
        d.edge('D1', 'D3', label='Two Factors', style='dashed', arrowhead='vee')

    with graph.subgraph(name='cluster_E') as e:
        e.attr(color='transparent')
        e.edge_attr.update(color=palette['yellow'], penwidth='2', fontcolor=palette['yellow'])

        e.node('E', 'Test an association', **branch_node_style, fontcolor=palette['yellow'], color=palette['yellow'])
        e.node('E1', 'What data types?', **decision_node_style, fillcolor=palette['yellow'])
        
        graph.edge('A', 'E', color=palette['yellow'])
        e.edge('E', 'E1')

        e.node('E2', 'Correlation / Regression', **outcome_node_style, color=palette['red'])
        e.edge('E1', 'E2', label='Two Numeric', style='dashed', arrowhead='vee', color=palette['red'], fontcolor=palette['red'])
        
        e.node('E3', 'Chi-Square Test', **outcome_node_style, color=palette['orange'])
        e.edge('E1', 'E3', label='Two Categorical', style='dashed', arrowhead='vee', color=palette['orange'], fontcolor=palette['orange'])
        
        e.node('E4', 'Redirect -> Compare Groups', **outcome_node_style, color=palette['brown'])
        e.edge('E1', 'E4', label='One of Each', style='dashed', arrowhead='vee', color=palette['brown'], fontcolor=palette['brown'])

    with graph.subgraph(name='cluster_Legend') as legend:
        legend.attr(color='transparent', fontcolor='white', fontsize='20')
        legend.node_attr.update(style='filled', color='none', fillcolor='#2D2D2D', fontcolor='white', shape='box')
        legend.node('Legend', 
                    ' (P) = Parametric Test\n(Assumes Normal Distribution)\n\n(NP) = Non-Parametric Test\n(Distribution-Free)', 
                    fontsize='12')

    return graph

def display_test_results(title, hypotheses, results, conclusion):
    st.subheader(title)
    with st.container(border=True):
        st.markdown(hypotheses)
        st.write("---")
        st.write("#### Results")
        for key, value in results.items():
            st.write(f"**{key}:** `{value}`")
        st.write("---")
        st.markdown(f"**Conclusion:** {conclusion}")

def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(text, buf, file_name=f"{filename}.png", mime="image/png")

def perform_levene_test(df, group_col, value_col):
    groups = df[group_col].unique()
    samples = [df[value_col][df[group_col] == g].dropna() for g in groups]
    valid_samples = [s for s in samples if len(s) > 1]

    if len(valid_samples) < 2:
        st.warning("⚠️ Warning: Levene's test could not be performed.")
        st.warning("The test requires at least two groups with more than one data point each.")
        return

    stat, p_value = stats.levene(*valid_samples)
    
    hypotheses = """
    - **Null Hypothesis (H₀)**: The variances of the groups are equal.
    - **Alternative Hypothesis (Hₐ)**: At least one group has a different variance.
    """
    results = {"Levene's Statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"Since the p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; the variances are not equal (assumption violated)."
    else:
        conclusion = f"Since the p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; the variances are equal (assumption met)."
    
    display_test_results("Levene's Test for Homogeneity of Variances", hypotheses, results, conclusion)

def perform_shapiro_test(data):
    stat, p_value = stats.shapiro(data)

    hypotheses = """
    - **Null Hypothesis (H₀)**: The data is normally distributed.
    - **Alternative Hypothesis (Hₐ)**: The data is not normally distributed.
    """

    results = {"Test Statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"Since the p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; the data is not normally distributed."
    else:
        conclusion = f"Since the p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; the data is consistent with normal distribution."

    display_test_results("Sahpiro-Wilk Test for Normality of Data", hypotheses, results, conclusion)

def perform_one_sample_ttest(data, popmean):
    stat, p_value = stats.ttest_1samp(data, popmean=popmean)
    hypotheses = f"""
    - **H₀**: The sample mean is equal to the population mean ({popmean}).
    - **Hₐ**: The sample mean is not equal to the population mean.
    """
    results = {"T-statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant difference** to population mean."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is **no significant difference** to population mean."
    display_test_results("One-Sample T-Test", hypotheses, results, conclusion)

def perform_ind_ttest(group1, group2):
    stat, p_value = stats.ttest_ind(group1, group2)
    hypotheses = """
    - **H₀**: The means of the two independent groups are equal.
    - **Hₐ**: The means of the two independent groups are not equal.
    """
    results = {"T-statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant difference** between two groups."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is **no significant difference** between groups."
    display_test_results("Independent Samples T-Test", hypotheses, results, conclusion)

def perform_paired_ttest(col1_data, col2_data):
    stat, p_value = stats.ttest_rel(col1_data, col2_data)
    hypotheses = """
    - **H₀**: The means of the two paired samples are equal.
    - **Hₐ**: The means of the two paired samples are not equal.
    """
    results = {"T-statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant difference** between the means of the two paired samples."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is **no significant difference** between the means of the two paired samples."
    display_test_results("Paired Samples T-Test", hypotheses, results, conclusion)

def perform_one_way_anova(df, cat_col, num_col):
    groups = df[cat_col].unique()
    samples = [df[num_col][df[cat_col] == g].dropna() for g in groups]
    valid_samples = [s for s in samples if len(s) > 1]

    if len(valid_samples) < 2:
        st.warning("ANOVA requires at least two groups with more than one data point each.")
        return

    stat, p_value = stats.f_oneway(*valid_samples)
    hypotheses = """
    - **H₀**: The means of all groups are equal.
    - **Hₐ**: At least one group mean is different from the others.
    """
    results = {"F-statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant difference** among the group means."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is **no significant difference** among the group means."
    display_test_results("One-Way ANOVA", hypotheses, results, conclusion)
    
def perform_two_way_anova(df, dep_var, iv1, iv2):
    df[dep_var] = pd.to_numeric(df[dep_var], errors='coerce')
    df = df.dropna(subset=[dep_var, iv1, iv2])

    if df.shape[0] < 3 or df[iv1].nunique() < 2 or df[iv2].nunique() < 2:
        st.warning("Not enough data or groups to perform a Two-Way ANOVA. Each independent variable must have at least two levels and sufficient data points.")
        return
    
    formula = f'Q("{dep_var}") ~ C(Q("{iv1}")) + C(Q("{iv2}")) + C(Q("{iv1}")):C(Q("{iv2}"))'
    
    try:
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        st.error(f"An error occurred during the ANOVA calculation: {e}")
        return

    st.subheader("Two-Way ANOVA")
    with st.container(border=True):
        st.markdown(f"This test examines how two independent categorical variables (**{iv1}** and **{iv2}**) affect a dependent numeric variable (**{dep_var}**).")
        
        st.write("#### ANOVA Table")
        st.dataframe(anova_table.style.format('{:.4f}'))

        st.write("#### Interpretation")
        
        alpha = 0.05

        p_iv1 = anova_table.loc[f'C(Q("{iv1}"))', 'PR(>F)']
        p_iv2 = anova_table.loc[f'C(Q("{iv2}"))', 'PR(>F)']
        p_interaction = anova_table.loc[f'C(Q("{iv1}")):C(Q("{iv2}"))', 'PR(>F)']

        st.markdown(f"**Main Effect of `{iv1}`**: The p-value is **{p_iv1:.4f}**.")
        if p_iv1 < alpha:
            st.success(f"Since the p-value is less than {alpha}, we conclude that there is a **statistically significant main effect** for `{iv1}`. This means that `{iv1}` has a significant effect on `{dep_var}`.")
        else:
            st.info(f"Since the p-value is greater than {alpha}, we conclude that there is **no statistically significant main effect** for `{iv1}`. This suggests that `{iv1}` does not have a significant effect on `{dep_var}`.")

        st.markdown(f"**Main Effect of `{iv2}`**: The p-value is **{p_iv2:.4f}**.")
        if p_iv2 < alpha:
            st.success(f"Since the p-value is less than {alpha}, we conclude that there is a **statistically significant main effect** for `{iv2}`. This means that `{iv2}` has a significant effect on `{dep_var}`.")
        else:
            st.info(f"Since the p-value is greater than {alpha}, we conclude that there is **no statistically significant main effect** for `{iv2}`. This suggests that `{iv2}` does not have a significant effect on `{dep_var}`.")

        st.markdown(f"**Interaction Effect (`{iv1}` * `{iv2}`)**: The p-value is **{p_interaction:.4f}**.")
        if p_interaction < alpha:
            st.success(f"Since the p-value is less than {alpha}, there is a **statistically significant interaction effect**. This is the most important finding: it means the effect of `{iv1}` on `{dep_var}` **depends on the level** of `{iv2}` (and vice-versa). The relationship is complex and the main effects should be interpreted with caution.")
        else:
            st.info(f"Since the p-value is greater than {alpha}, there is **no statistically significant interaction effect**. This suggests that the effect of `{iv1}` on `{dep_var}` is independent of `{iv2}`.")

def perform_mann_whitney_u(group1, group2):
    stat, p_value = stats.mannwhitneyu(group1, group2)
    hypotheses = """
    - **H₀**: The distributions of the two groups are equal.
    - **Hₐ**: The distributions are not equal.
    """
    results = {"U-statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant difference** between the two groups."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is no **significant difference** between the two groups."
    display_test_results("Mann-Whitney U Test", hypotheses, results, conclusion)

def perform_kruskal_wallis(df, group_col, value_col):
    samples = [df[value_col][df[group_col] == g].dropna() for g in df[group_col].unique()]
    stat, p_value = stats.kruskal(*samples)
    hypotheses = """
    - **H₀**: The distributions of all groups are equal.
    - **Hₐ**: At least one distribution is different.
    """
    results = {"H-statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant difference** among the groups."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is no **significant difference** among groups."
    display_test_results("Kruskal-Wallis H Test", hypotheses, results, conclusion)
    
def perform_wilcoxon_signed_rank(col1_data, col2_data):
    stat, p_value = stats.wilcoxon(col1_data, col2_data)
    hypotheses = """
    - **H₀**: The distributions of the paired samples are equal.
    - **Hₐ**: The distributions are not equal.
    """
    results = {"W-statistic": f"{stat:.4f}", "P-value": f"{p_value:.4f}"}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant difference** between the paired samples."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is no **significant difference** between paired samples."
    display_test_results("Wilcoxon Signed-Rank Test", hypotheses, results, conclusion)

def perform_chi_square(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    hypotheses = f"""
    - **H₀**: The variables **{col1}** and **{col2}** are independent.
    - **Hₐ**: They are dependent.
    """
    results = {"Chi-Square": f"{chi2:.4f}", "P-value": f"{p_value:.4f}", "Degrees of Freedom": dof}
    alpha = 0.05
    if p_value < alpha:
        conclusion = f"The p-value ({p_value:.4f}) is less than {alpha}. We reject the null hypothesis; there is a **significant association** between variables."
    else:
        conclusion = f"The p-value ({p_value:.4f}) is greater than {alpha}. We fail to reject the null hypothesis; there is no **significant association** between variables."
    display_test_results("Chi-Square Test of Independence", hypotheses, results, conclusion)
    st.write("#### Contingency Table"); st.dataframe(contingency_table)

def perform_linear_regression_1(df, iv, dv):
    X = sm.add_constant(df[iv].dropna())
    y = df[dv].dropna()
    model = sm.OLS(y, X).fit()
    st.subheader("Simple Linear Regression Analysis")
    st.code(model.summary())
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    sns.regplot(x=iv, y=dv, data=df, ax=ax, scatter_kws={'alpha': 0.5, 'color': 'grey'},line_kws={"color": "#FF5733", "lw": 1.5})
    st.pyplot(fig)

def perform_linear_regression(df, iv, dv):
    temp_df = df[[iv, dv]].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(temp_df) < 2:
        st.warning("Not enough valid data to perform linear regression.")
        return

    temp_df = temp_df.sort_values(by=iv)

    X = sm.add_constant(temp_df[iv])
    y = temp_df[dv]
    model = sm.OLS(y, X).fit()

    tab1, tab2, tab3, tab4 = st.tabs(["Model Summary", "Regression Plot", "Residual Analysis", "Influence Plot"])

    with tab1:
        st.subheader("Regression Model Summary")
        st.code(f"{model.summary()}")
        
        st.markdown("##### Key Metrics Explained:")
        st.markdown(f"- **R-squared ({model.rsquared:.3f})**: Indicates that **{model.rsquared:.1%}** of the variance in the dependent variable (`{dv}`) can be explained by the independent variable (`{iv}`).")
        st.markdown(f"- **P-value for `{iv}` ({model.pvalues[iv]:.3f})**: Tests the null hypothesis that the variable has no effect. A low p-value (< 0.05) suggests a significant relationship.")

    with tab2:
        st.subheader("Regression Line and Confidence Interval")
        with st.expander("Customisation Options"):
            title = st.text_input("Chart Title:", value=f"Regression of {dv} on {iv}", key="title_reg")
            x_label = st.text_input("X-axis Label:", value=iv, key="x_reg")
            y_label = st.text_input("Y-axis Label:", value=dv, key="y_reg")

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(temp_df[iv], temp_df[dv], alpha=0.5, color='grey')
        ax.plot(temp_df[iv], model.predict(X), color="#FF5733", lw=1.5)
        predictions = model.get_prediction(X)
        ci = predictions.conf_int()
        ax.fill_between(temp_df[iv], ci[:, 0], ci[:, 1], color='red', alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, "regression plot", "Download Regression Plot")

    with tab3:
        st.subheader("Analysis of Residuals")
        st.markdown("Residuals are the errors in the model's predictions. Their analysis is crucial for validating the model's assumptions.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("##### Normality of Residuals (Q-Q Plot)")
            fig_qq = sm.qqplot(model.resid, line='s')
            plt.title("Q-Q Plot")
            st.pyplot(fig_qq)
            st.markdown("Points should fall closely along the red line if residuals are normally distributed.")
        
        with c2:
            st.write("##### Homoscedasticity (Residuals vs. Fitted)")
            fig_resid = plt.figure()
            sns.residplot(x=model.fittedvalues, y=model.resid, scatter_kws={'alpha': 0.5})
            plt.title("Residuals vs. Fitted Values")
            plt.xlabel("Fitted Values")
            plt.ylabel("Residuals")
            st.pyplot(fig_resid)
            st.markdown("There should be no clear pattern or funnel shape in this plot.")
            st.markdown("A random scatter suggests the variance of residuals is constant (homoscedastic).")

    with tab4:
        st.subheader("Outlier and Influence Analysis")
        st.markdown("This plot helps identify influential data points that may have a disproportionate effect on the model.")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sm.graphics.influence_plot(model, criterion="cooks", ax=ax)
        ax.set_title("Influence Plot (Cook's Distance)")
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("Points with high **Cook's distance** (larger bubbles, especially > 0.5) are considered influential and may warrant further investigation.")

def perform_correlation_analysis(df, selected_cols, method):
    st.subheader(f"{method.capitalize()} Correlation Matrix")
    corr_df = df[selected_cols].corr(method=method)
    with st.expander("Customisation Options"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("##### Plot Appearance")
            cmap_options = ['coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu_r', 'PuOr', 'PRGn']
            cmap = st.selectbox("Color Map:", cmap_options, index=0, key="cmap_corr")
            show_annot = st.checkbox("Show Annotations (Values)", value=True, key="annot_corr")
            fig_width = st.slider("Figure Width:", min_value=4, max_value=20, value=10, key="width_corr")
            fig_height = st.slider("Figure Height:", min_value=3, max_value=20, value=6, key="height_corr")
        with c2:
            st.write("##### Labels & Title")
            title = st.text_input("Chart Title:", value="Correlation Heatmap", key="title_corr")
            title_fontsize = st.slider("Title Font Size:", 10, 30, 16, key="title_size_corr")
            x_label = st.text_input("X-axis Label:", value="", key="x_corr")
            y_label = st.text_input("Y-axis Label:", value="", key="y_corr")
    try:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        center_val = 0 if cmap in ['coolwarm', 'RdBu_r', 'PuOr', 'PRGn'] else None
        sns.heatmap(
            corr_df, 
            annot=show_annot, 
            cmap=cmap, 
            fmt=".2f", 
            ax=ax,
            center=center_val
        )
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, "correlation_heatmap", "Download Heatmap")
    except Exception as e:
        st.error(f"An error occurred while generating the plot: {e}")
    with st.expander("View Correlation Values"):
        st.dataframe(corr_df.style.format('{:.2f}'))

def main():
    st.header("Statistical Analysis")
    st.caption("""
        **Move from observation to inference.** This section provides tools to rigorously test the hypotheses generated during your visual exploration. Validate the significance of your insights and understand the underlying structure of your data.
    """)
    st.caption("""
        **Key analyses include:**
        - **Distribution Testing:** Assess normality (e.g., Shapiro-Wilk) to guide test selection.
        - **Parametric Tests:** For normally distributed data (e.g., T-tests, ANOVA).
        - **Non-Parametric Tests:** For non-normal data (e.g., Mann-Whitney U, Kruskal-Wallis).
        - **Correlation Analysis:** Measure relationships between variables (e.g., Pearson, Spearman).
    """)

    if "initial_df" not in st.session_state or st.session_state.initial_df is None:
        st.warning("No data found. Please upload a dataset or use sample dataset on the 'Data Upload' page.")
        if st.button("Go to Upload Data", type='primary'):
            st.switch_page("pages/2_upload_data.py")
        return
    
    if "working_df" not in st.session_state or st.session_state.working_df is None:
        st.session_state.working_df = st.session_state.initial_df.copy()

    df = st.session_state.working_df

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    tab_guide, tab_assump, tab_para, tab_nonpara, tab_assoc = st.tabs([
        "📖 Which Test?", "✅ Assumptions", "🔔 Parametric", "〰️ Non-Parametric", "🔗 Associations"
    ])

    with tab_guide:
        st.header("Find the Right Test for Your Data")
        st.markdown("""**Answer these questions to find the right statistical test for your data:**""")

        q1 = st.selectbox("1. What is your main goal?", 
            [
                "Select...", 
                "Compare a group to a known standard", 
                "Compare two groups", 
                "Compare three or more groups", 
                "Test an association or relationship"
            ]
        )

        if q1 == "Compare a group to a known standard":
            st.success("""
                **Recommendation:** Use the **One-Sample T-Test** (found in the 'Parametric' tab).
                - **Use if:** Your data is continuous and approximately normally distributed.
                "This test determines if the mean of a single group is significantly different from a known or hypothesized value.
            """)
        elif q1 == "Compare two groups":
            q2 = st.radio("2. Are the groups independent or paired?", 
                ["Independent (e.g., control vs. treatment groups)", "Paired (e.g., before-and-after measurements on the same subjects)"],
                horizontal=True
            )
            if q2.startswith("Independent"): 
                st.success("""
                    **Recommendation:** Use the **Independent Samples T-Test** (if assumptions are met) or the **Mann-Whitney U Test** (if not). 
                    You can find these in the 'Parametric' and 'Non-Parametric' tabs, respectively.
                """)
            else: 
                st.success("""
                    **Recommendation:** Use the **Paired Samples T-Test** (if assumptions are met) or the **Wilcoxon Signed-Rank Test** (if not). 
                    You can find these in the 'Parametric' and 'Non-Parametric' tabs.
                """)

        elif q1 == "Compare three or more groups":
            q3 = st.radio(
                "2. How many independent factors (variables) are you analyzing?",
                ["One (e.g., comparing test scores across three different schools)", "Two (e.g., comparing test scores across schools AND genders)"],
                horizontal=True
            )
            if q3.startswith("One"):
                st.success("""
                    **Recommendation:** Use the **One-Way ANOVA** (if assumptions are met) or the **Kruskal-Wallis H Test** (if not). 
                    You can find these in the 'Parametric' and 'Non-Parametric' tabs.
                """)
            else:
                st.success("""
                    **Recommendation:** Use the **Two-Way ANOVA** (found in the 'Parametric' tab) to examine the effect of two independent variables on a single dependent variable, including their interaction.
                """)
        elif q1 == "Test an association or relationship":
            q4 = st.radio(
                "2. What are the data types of your two variables?", 
                ["Two Numeric Variables", "Two Categorical Variables", "One Numeric and One Categorical"],
                horizontal=True
            )
            if q4 == "Two Numeric Variables":
                st.success("""
                    **Recommendation:** Use **Correlation** to measure strength and direction.
                    Pearson for linear relationships or Spearman and Kendall for monotonic relationships (use Kendall for small samples or many ties) 
                    Use **Simple Linear Regression** to model the relationship.
                    Find these in the 'Associations' tab.
                """)
            elif q4 == "Two Categorical Variables":
                st.success("""
                    **Recommendation:** Use the **Chi-Square Test of Independence**. 
                    You can find this in the 'Non-Parametric' tab.
                """)
            elif q4 == "One Numeric and One Categorical":
                st.info("""
                    **Clarification:** To analyze a numeric and a categorical variable, your goal is typically to *compare the means* of the numeric variable across the different categories. 
                    Please select **'Compare two groups'** or **'Compare three or more groups'** as your main goal instead.
                """)

        st.markdown("---")
        st.header("Visual Guide")

        final_graph = create_aesthetic_graph()
        st.graphviz_chart(final_graph)

    with tab_assump:
        st.header("Check Your Assumptions")
        st.subheader("1. Normality of Data")
        if numeric_cols:
            norm_col = st.selectbox("Select a column to test for normality:", numeric_cols)
            if st.button("Run Shapiro-Wilk Test"): perform_shapiro_test(df[norm_col].dropna())
        
        st.subheader("2. Homogeneity of Variances (for independent groups)")
        valid_levene_cols = [c for c in categorical_cols if df[c].nunique() > 1]
        if numeric_cols and valid_levene_cols:
            c1, c2 = st.columns(2)
            val_col = c1.selectbox("Select Numeric Variable:", numeric_cols, key="levene_val")
            group_col = c2.selectbox("Select Grouping Variable:", valid_levene_cols, key="levene_group")
            if st.button("Run Levene's Test"): perform_levene_test(df, group_col, val_col)

    with tab_para:
        st.header("Parametric Tests (Assume Normal Distribution)")
        para_test = st.selectbox("Select Test", ["One-Sample T-Test", "Independent Samples T-Test", "Paired Samples T-Test", "One-Way ANOVA", "Two-Way ANOVA"])
        
        if para_test == "Independent Samples T-Test":
            st.subheader("Independent Samples T-Test")
            st.markdown("Compares the means of two independent groups to determine if there is a statistically significant difference between them.")
            valid_cols = [c for c in categorical_cols if df[c].nunique() == 2]
            if valid_cols:
                c1, c2 = st.columns(2)
                group_col = c1.selectbox("Select Grouping Variable (must have 2 groups):", valid_cols)
                val_col = c2.selectbox("Select Numeric Variable:", numeric_cols)
                if st.button("Run Independent T-Test"):
                    groups = df[group_col].unique()
                    g1 = df[val_col][df[group_col] == groups[0]].dropna()
                    g2 = df[val_col][df[group_col] == groups[1]].dropna()
                    perform_ind_ttest(g1, g2)
            else: st.warning("No categorical columns with exactly two groups found. This test requires a grouping variable with two categories.")
        
        elif para_test == "One-Sample T-Test":
            st.subheader("One-Sample T-Test")
            st.markdown("Determines whether the mean of a single sample is statistically different from a known or hypothesized population mean.")
            c1, c2 = st.columns(2)
            selected_col = c1.selectbox("Select Numeric Variable:", numeric_cols)
            pop_mean = c2.number_input("Hypothesized Population Mean:", value=0.0)
            if st.button("Run One-Sample T-Test"):
                data = df[selected_col].dropna()
                if len(data) > 1:
                    perform_one_sample_ttest(data, pop_mean)
                else:
                    st.warning(f"The selected column '{selected_col}' does not have enough data to perform the test.")

        elif para_test == "Paired Samples T-Test":
            st.subheader("Paired Samples T-Test")
            st.markdown("Compares the means of two related groups (e.g., 'before' and 'after' measurements) to determine if there is a significant difference.")
            c1, c2 = st.columns(2)
            col1 = c1.selectbox("Select First Numeric Variable (e.g., 'Before'):", numeric_cols, key="paired_col1")
            col2 = c2.selectbox("Select Second Numeric Variable (e.g., 'After'):", numeric_cols, key="paired_col2")
            if st.button("Run Paired T-Test"):
                if col1 == col2:
                    st.error("Please select two different columns for the paired t-test.")
                else:
                    paired_data = df[[col1, col2]].dropna()
                    if len(paired_data) > 1:
                        perform_paired_ttest(paired_data[col1], paired_data[col2])
                    else:
                        st.warning("There is not enough overlapping data between the selected columns to perform the test.")
        
        elif para_test == "One-Way ANOVA":
            st.subheader("One-Way ANOVA")
            st.markdown("Compares the means of three or more independent groups to determine if at least one group mean is statistically different from the others.")
            valid_cols = [c for c in categorical_cols if df[c].nunique() > 2]
            if not valid_cols:
                st.warning("No categorical columns with three or more distinct groups were found. ANOVA is typically used for comparing 3+ groups.")
            else:
                c1, c2 = st.columns(2)
                cat_col = c1.selectbox("Select Categorical Grouping Variable (3+ groups):", valid_cols, key="anova_cat")
                num_col = c2.selectbox("Select Numeric Variable:", numeric_cols, key="anova_num")
                if st.button("Run One-Way ANOVA"):
                    perform_one_way_anova(df, cat_col, num_col)
        
        elif para_test == "Two-Way ANOVA":
            st.subheader("Two-Way ANOVA")
            st.markdown("Examines the influence of two different categorical independent variables on one continuous dependent variable.")
            c1, c2, c3 = st.columns(3)
            dep_var = c1.selectbox("Dependent Variable (Numeric):", numeric_cols, key="2anova_dep")
            iv1 = c2.selectbox("Independent Variable 1 (Categorical):", categorical_cols, key="2anova_iv1")
            iv2 = c3.selectbox("Independent Variable 2 (Categorical):", categorical_cols, key="2anova_iv2")
            if st.button("Run Two-Way ANOVA"):
                if iv1 == iv2:
                    st.error("Please select two different independent variables.")
                else:
                    perform_two_way_anova(df, dep_var, iv1, iv2)

    with tab_nonpara:
        st.header("Non-Parametric Tests (Do Not Assume Normal Distribution)")
        nonpara_test = st.selectbox("Select Test", ["Mann-Whitney U Test", "Kruskal-Wallis H Test", "Wilcoxon Signed-Rank Test", "Chi-Square Test"])

        if nonpara_test == "Mann-Whitney U Test":
            st.subheader("Mann-Whitney U Test")
            st.markdown("Compares the distributions of two independent groups. It's the non-parametric equivalent of the independent t-test.")
            valid_cols = [c for c in categorical_cols if df[c].nunique() == 2]
            if not valid_cols:
                st.warning("No categorical columns with exactly two distinct groups were found for this test.")
            else:
                c1, c2 = st.columns(2)
                group_col = c1.selectbox("Select Grouping Variable (2 groups):", valid_cols, key="mw_group")
                val_col = c2.selectbox("Select Numeric Variable:", numeric_cols, key="mw_val")
                if st.button("Run Mann-Whitney U Test"):
                    groups = df[group_col].unique()
                    g1 = df[val_col][df[group_col] == groups[0]].dropna()
                    g2 = df[val_col][df[group_col] == groups[1]].dropna()
                    perform_mann_whitney_u(g1, g2)
        
        elif nonpara_test == "Kruskal-Wallis H Test":
            st.subheader("Kruskal-Wallis H Test")
            st.markdown("Compares the distributions of three or more independent groups. It's the non-parametric equivalent of One-Way ANOVA.")
            valid_cols = [c for c in categorical_cols if df[c].nunique() > 2]
            if not valid_cols:
                st.warning("No categorical columns with three or more distinct groups were found for this test.")
            else:
                c1, c2 = st.columns(2)
                group_col = c1.selectbox("Select Grouping Variable (3+ groups):", valid_cols, key="kw_group")
                val_col = c2.selectbox("Select Numeric Variable:", numeric_cols, key="kw_val")
                if st.button("Run Kruskal-Wallis H Test"):
                    perform_kruskal_wallis(df, group_col, val_col)

        elif nonpara_test == "Wilcoxon Signed-Rank Test":
            st.subheader("Wilcoxon Signed-Rank Test")
            st.markdown("Compares two related samples or repeated measurements on a single sample. It's the non-parametric equivalent of the paired t-test.")
            c1, c2 = st.columns(2)
            col1 = c1.selectbox("Select First Paired Variable:", numeric_cols, key="wilcox_col1")
            col2 = c2.selectbox("Select Second Paired Variable:", numeric_cols, key="wilcox_col2")
            if st.button("Run Wilcoxon Signed-Rank Test"):
                if col1 == col2:
                    st.error("Please select two different columns for the test.")
                else:
                    paired_data = df[[col1, col2]].dropna()
                    if len(paired_data) > 1:
                        perform_wilcoxon_signed_rank(paired_data[col1], paired_data[col2])
                    else:
                        st.warning("Not enough paired data (non-missing rows) between the selected columns.")
        
        elif nonpara_test == "Chi-Square Test":
            st.subheader("Chi-Square Test of Independence")
            st.markdown("Determines whether there is a significant association between two categorical variables.")
            c1, c2 = st.columns(2)
            cat1 = c1.selectbox("Select First Categorical Variable:", categorical_cols, key="chi_cat1")
            cat2 = c2.selectbox("Select Second Categorical Variable:", categorical_cols, key="chi_cat2")
            if st.button("Run Chi-Square Test"):
                if cat1 == cat2:
                    st.error("Please select two different categorical variables.")
                else:
                    perform_chi_square(df, cat1, cat2)

    with tab_assoc:
        st.header("Test Associations")
        assoc_test = st.selectbox("Select Analysis", ["Correlation", "Simple Linear Regression"])

        if assoc_test == "Correlation":
            if len(numeric_cols) > 1:
                cols = st.multiselect("Select numeric columns:", numeric_cols, default=numeric_cols[:2])
                method = st.selectbox("Method:", ["pearson", "spearman", "kendall"])
                if len(cols) > 1: 
                    perform_correlation_analysis(df, cols, method)
                else:
                    st.warning("Please select at least two numeric columns to perform a correlation analysis.")
        
        if assoc_test == "Simple Linear Regression":
            st.subheader("Simple Linear Regression")
            st.markdown("Analyzes the linear relationship between a single independent variable (X) and a dependent variable (Y).")

            if len(numeric_cols) < 2:
                st.warning("⚠️ Simple Linear Regression requires at least two numeric columns in the dataset (one for X and one for Y).")
            else:
                c1, c2 = st.columns(2)
                iv = c1.selectbox("Select Independent Variable (X):", numeric_cols, key="slr_iv")

                dv_options = [c for c in numeric_cols if c != iv]
                if not dv_options:
                    c2.warning("No other numeric columns are available to be the dependent variable.")
                else:
                    dv = c2.selectbox("Select Dependent Variable (Y):", dv_options, key="slr_dv")

                    if st.button("Run Regression"):
                        regression_df = df[[iv, dv]].dropna()

                        if len(regression_df) < 2:
                            st.error(f"Error: Not enough overlapping data between '{iv}' and '{dv}'. The regression requires at least two complete rows of data.")
                        else:
                            perform_linear_regression(regression_df, iv, dv)

if __name__ == "__main__":
    main()