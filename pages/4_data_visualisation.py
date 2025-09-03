import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
from io import BytesIO

def preview_data(df, max_rows=50, default_rows=5, key=None):
    '''preview data'''
    max_rows = int(max_rows)
    default_rows = int(default_rows)
    if key is None:
        key = f"preview_slider_{len(df.columns)}_{df.shape[0]}"
    num_rows = st.slider("Select number of rows", 0, max_rows, default_rows, key=key)
    df_rows = len(df)
    if num_rows > df_rows:
        st.warning(f"The number of rows you selected ({num_rows}) exceeds the available data ({df_rows}) rows.")
    else:
        st.dataframe(df.head(num_rows), hide_index=True)

def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    '''get data type of columns'''
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
    for col in df.columns:
        if col not in numeric_cols + categorical_cols + datetime_cols:
            try:
                if pd.to_datetime(df[col], errors='raise').notna().all():
                    datetime_cols.append(col)
            except (ValueError, TypeError):
                pass 
    return {
        "numeric": numeric_cols,
        "categorical": list(set(categorical_cols) - set(datetime_cols)),
        "datetime": datetime_cols
    }

def is_suitable_for_line(df: pd.DataFrame):
    '''check whether data is suitable for plotting line'''
    col_types = get_column_types(df)
    # A line chart needs at least one numeric y-axis and either a numeric or datetime x-axis
    if len(col_types["numeric"]) >= 1 and (len(col_types["numeric"]) >= 2 or len(col_types["datetime"]) >= 1):
        return True
    return False

def is_suitable_for_bar(df: pd.DataFrame):
    '''check whether data is suitable for plotting bar'''
    col_types = get_column_types(df)
    # A bar chart needs a categorical axis and a numeric axis
    if len(col_types["categorical"]) >= 1 and len(col_types["numeric"]) >= 1:
        return True
    return False

def is_suitable_for_scatter(df: pd.DataFrame):
    '''check whether data is suitable for plotting scatter'''
    col_types = get_column_types(df)
    # A scatter plot needs at least two numeric columns
    if len(col_types["numeric"]) >= 2:
        return True
    return False

def is_suitable_for_histogram(df: pd.DataFrame):
    '''check whether data is suitable for plotting histogram'''
    col_types = get_column_types(df)
    # A histogram needs at least one numeric column
    if len(col_types["numeric"]) >= 1:
        return True
    return False

def is_suitable_for_box(df: pd.DataFrame):
    '''check whether data is suitable for plotting box'''
    col_types = get_column_types(df)
    # A box plot needs at least one numeric column
    if len(col_types["numeric"]) >= 1:
        return True
    return False

def is_suitable_for_violin(df: pd.DataFrame):
    '''check whether data is suitable for plotting violin'''
    col_types = get_column_types(df)
    if len(col_types["numeric"]) == 0:
        return False
    for col in col_types["numeric"]:
        if df[col].nunique() < 3:
            return False
    return True

def is_suitable_for_pie(df: pd.DataFrame):
    '''check whether data is suitable for plotting pie'''
    col_types = get_column_types(df)
    # A pie chart needs a categorical column for slices and a numeric column for values
    if len(col_types["categorical"]) >= 1 and len(col_types["numeric"]) >= 1:
        return True
    return False

def is_suitable_for_stack(df: pd.DataFrame):
    '''check whether data is suitable for plotting stack plot'''
    col_types = get_column_types(df)
    # A stack plot is similar to a line chart but often with multiple y-values
    if len(col_types["numeric"]) >= 1 and (len(col_types["numeric"]) >= 2 or len(col_types["datetime"]) >= 1):
        return True
    return False

def select_axis_to_plot(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=""):
    '''allow user to select axis to plot'''
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("Select the column for the X-axis:", df_cols_option, index=0, key=f"x_axis_{chart}_{key_suffix}")
    with col_y:
        y_axis = st.selectbox("Select the column for the Y-axis:", df_cols_option, index=1, key=f"y_axis_{chart}_{key_suffix}")
    if x_axis == y_axis:
        st.warning("Please select different columns for the X and Y axes.")
    else:
        function = chart_function_dict[chart]
        function(chart_df, x_axis, y_axis)

def select_axis_histogram(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=""):
    '''allow user to select axis to plot histogram'''
    numeric_cols = [col for col in df_cols_option if pd.api.types.is_numeric_dtype(chart_df[col])]
    col1, col2 = st.columns(2)
    with col1:
        selected_columns = st.multiselect("Select column(s) to plot:", numeric_cols, default=numeric_cols[:1] if len(numeric_cols) > 0 else [], key=f"hist_cols_{key_suffix}")
        if not selected_columns:
            st.warning("Please select at least one column")
            return
    with col2:
        default_bins = 10
        if len(selected_columns) == 1 and pd.api.types.is_numeric_dtype(chart_df[selected_columns[0]]):
            col_data = chart_df[selected_columns[0]].dropna()
            iqr = col_data.quantile(0.75) - col_data.quantile(0.25)
            if iqr > 0:
                bin_width = (2 * iqr) / (len(col_data) ** (1/3))
                default_bins = max(5, min(50, int((col_data.max() - col_data.min()) / bin_width)))
        n_bins = st.slider("Number of bins:", min_value=5, max_value=50, value=default_bins, key=f"hist_bins_{key_suffix}")
    if len(selected_columns) > 1:
        display_mode = st.radio("Display mode:", options=["Single Axes", "Subplots"], index=0, key=f"hist_mode_{key_suffix}")
    else:
        display_mode = "Single Axes"
    function = chart_function_dict[chart]
    function(chart_df, selected_columns, n_bins, display_mode)

def select_axis_box(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=""):
    '''allow user to select axis to plot box plot'''
    numeric_cols = [col for col in df_cols_option if pd.api.types.is_numeric_dtype(chart_df[col])]
    selected_columns = st.multiselect("Select column(s) to plot:", numeric_cols, default=numeric_cols[:min(1, len(numeric_cols))], key=f"box_cols_{key_suffix}")
    if not selected_columns:
        st.warning("Please select at least one numeric olumn")
        return
    function = chart_function_dict[chart]
    function(chart_df, selected_columns)

def select_axis_violin(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=""):
    '''allow user to select axis to plot violin plot'''
    numeric_cols = [col for col in df_cols_option if pd.api.types.is_numeric_dtype(chart_df[col])]
    categorical_cols = [col for col in df_cols_option if isinstance(chart_df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(chart_df[col])]
    numeric, category = st.columns(2)
    with numeric:
        selected_num_cols = st.multiselect("Select column(s) to plot:", numeric_cols, default=numeric_cols[:min(1, len(numeric_cols))], key=f"violin_cols_{key_suffix}")
    with category:
        if categorical_cols:
            group_by = st.selectbox("Optional: Select grouping variable (categorical):", categorical_cols, index=None, key=f"violin_group_{key_suffix}")
        else:
            group_by = None
    if not selected_num_cols:
        st.warning("Please select at least one numeric column")
        return
    function = chart_function_dict[chart]
    function(chart_df, selected_num_cols, group_by)

def select_axis_pie(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=""):
    '''allow user to select axis to plot pie plot'''
    numeric_cols = [col for col in df_cols_option if pd.api.types.is_numeric_dtype(chart_df[col])]
    categorical_cols = [col for col in df_cols_option if isinstance(chart_df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(chart_df[col])]
    category, numeric = st.columns(2)
    with category:
        selected_cat_col = st.selectbox("Select category/label column:", categorical_cols, key=f"pie_cat_cols_{key_suffix}")
    with numeric:
        selected_num_col = st.selectbox("Select value column:", numeric_cols, key=f"pie_num_cols_{key_suffix}")
    if not selected_cat_col or not selected_num_col:
        st.warning("Please select both label and value columns")
        return
    if len(chart_df[selected_cat_col].dropna()) != len(chart_df[selected_num_col].dropna()):
        st.warning("Selected columns have different numbers of non-null values")
        return
    function = chart_function_dict[chart]
    function(chart_df, selected_cat_col, selected_num_col)
    
def select_axis_stack(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=""):
    '''allow user to select axis to plot stack plot'''
    x_options = [col for col in df_cols_option if pd.api.types.is_numeric_dtype(chart_df[col]) or pd.api.types.is_datetime64_any_dtype(chart_df[col])]
    y_options = [col for col in df_cols_option if pd.api.types.is_numeric_dtype(chart_df[col])]
    colx, coly = st.columns(2)
    with colx:
        x_col = st.selectbox("X-axis (time/numeric):", x_options, index=0, key=f"stack_x_{key_suffix}")
    with coly:
        y_cols = st.multiselect("Y-axis values:", y_options, default=y_options[:3] if len(y_options) > 2 else y_options, key=f"stack_y_{key_suffix}")
    if not x_col or not y_cols:
        st.warning("Please select both x-axis and at least one y-axis column")
    plot_type = st.radio("Stack plot type:", ["Area", "Bar"], index=0, key=f"stack_type_{key_suffix}")
    function = chart_function_dict[chart]
    function(chart_df, x_col, y_cols, plot_type)

def get_image_download_link(fig, filename, text):
    '''allow user to download and save plot'''
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(text, buf, file_name=filename, mime="image/png")

def plot_line(df: pd.DataFrame, x: str, y: str):
    '''plot line and allow customised options'''
    rotation = 0
    title = f"Line Chart: {y} vs {x}"
    x_label = x
    y_label = y
    marker_style = "o"
    line_style = "-"
    max_points = 100
    with st.expander("Customisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            rotation = st.slider("Rotate x-axis labels", 0, 90, 0, key=f"rotation_line_{x}_{y}")
            marker_style = st.selectbox("Marker style", ['none', 'o', 's', '^', 'D', 'v', 'p', '*'], index=1, key=f"marker_line_{x}_{y}")
            line_style = st.selectbox("Line style", ['-', '--', '-.', ':'], index=0, key=f"line_style_{x}_{y}")
            if len(df) > max_points:
                agg_method = st.selectbox("Reduce data points by", ["None", "Mean", "Median", "Max", "Min"], index=1, key=f"agg_method_{x}_{y}")
            else:
                agg_method = "None"
        with col2:
            title = st.text_input("Chart title", value=title, key=f"title_line_{x}_{y}")
            x_label = st.text_input("x-axis label", value=x, key=f"xlabel_line_{x}_{y}")
            y_label = st.text_input("y-aixs label", value=y, key=f"ylabel_line_{x}_{y}")
    if len(df) > max_points and agg_method == "None":
        st.warning(f"Displaying a large dataset ({len(df)} points). Consider choosing an aggregation method for better performance.")
    fig, ax = plt.subplots(figsize=(10,6))
    try:
        plot_df = df[[x, y]].copy().dropna(subset=[x, y])
        if plot_df.empty:
            st.warning("No valid data to plot after removing missing values.")
            return
        if pd.api.types.is_categorical_dtype(plot_df[x]) or pd.api.types.is_object_dtype(plot_df[x]):
            plot_df = plot_df.sort_values(x)
            x_values = plot_df[x].astype(str)
            y_values = plot_df[y]
        else:
            plot_df = plot_df.sort_values(x)
            x_values = plot_df[x]
            y_values = plot_df[y]
            if len(plot_df) > max_points and agg_method != "None":
                bins = min(max_points, len(plot_df))
                if pd.api.types.is_datetime64_any_dtype(plot_df[x]):
                    plot_df['bin'] = pd.cut(plot_df[x], bins=bins)
                    agg_df = plot_df.groupby('bin', observed=True).agg(y_agg=(y, agg_method.lower())).reset_index()
                    agg_df['x_agg'] = agg_df['bin'].apply(lambda b: b.mid).astype('datetime64[ns]')
                    agg_df = agg_df.sort_values('x_agg')
                    x_values = agg_df['x_agg']
                    y_values = agg_df['y_agg']
                else:
                    plot_df['bin'] = pd.cut(plot_df[x], bins=bins)
                    agg_df = plot_df.groupby('bin', observed=True).agg(
                        x_agg=(x, 'mean'),
                        y_agg=(y, agg_method.lower())
                    ).reset_index(drop=True)
                    agg_df = agg_df.sort_values('x_agg')
                    x_values = agg_df['x_agg']
                    y_values = agg_df['y_agg']
                st.info(f"Downsampled to {len(x_values)} points using {agg_method}.")
        ax.plot(x_values, y_values, marker=marker_style, linestyle=line_style, linewidth=1.5, alpha=0.8)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(title, fontsize=12, pad=20)
        if isinstance(df[x].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[x]):
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(x_values, rotation=rotation, ha='right' if rotation > 0 else 'center')
            if len(x_values) > 20:
                for i, label in enumerate(ax.xaxis.get_ticklabels()):
                    if i % 5 != 0:
                        label.set_visible(False)
        else:
            ax.tick_params(axis='x', rotation=rotation)
            if pd.api.types.is_datetime64_any_dtype(df[x]):
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if len(df) > 5000 and agg_method == "None":
            st.warning("Consider using aggregation for better visualization of trends")
        st.pyplot(fig)
        get_image_download_link(fig, f"line_chart_{x}_vs_{y}.png", "Download Line Chart")    
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.stop()

def plot_bar(df: pd.DataFrame, x: str, y: str):
    '''plot bar and allow customised options'''
    rotation = 0
    title = f"Bar Chart: {y} by {x}"
    x_label = x
    y_label = y
    bar_width = 0.8
    bar_color = "#1f77b4"
    with st.expander("Customisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            rotation = st.slider("Rotate x-axis labels", 0, 90, 0, key=f"rotation_bar_{x}_{y}")
            bar_width = st.slider("Bar width", 0.1, 1.0, 0.8, step=0.1, key=f"bar_width_{x}_{y}")
            use_same_color = st.checkbox("Use same color for all", value=True, key="bar_same_color")
            if use_same_color:
                bar_color = st.color_picker("Bar color", bar_color, key=f"bar_color_{x}_{y}")
            else:
                color_palette = st.selectbox("Color palette", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Pastel1', 'Set1', 'Set2', 'Set3', 'tab20'], index=5, key=f"color_palette_{x}_{y}")
        with col2:
            title = st.text_input("Chart title", value=title, key=f"bar_title_{x}_{y}")
            x_label = st.text_input("x-axis label", value=x, key=f"bar_xlabel_{x}_{y}")
            y_label = st.text_input("y-axis label", value=y, key=f"bar_ylabel_{x}_{y}")
    fig, ax = plt.subplots(figsize=(10,6))
    try:
        plot_data = df.groupby(x)[y].mean().reset_index()
        x_values = plot_data[x].astype(str)
        y_values = plot_data[y]
        if use_same_color:
            colors = [bar_color] * len(x_values)
        else:
            cmap = plt.get_cmap(color_palette)
            colors = cmap(np.linspace(0, 1, len(x_values)))
        bars = ax.bar(x_values, y_values, width=bar_width, color=colors,edgecolor='white', linewidth=0.5)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(x_values, rotation=rotation, ha='right' if rotation > 0 else 'center')
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, f"bar_chart_{x_label}_by_{y_label}.png", "Download Bar Chart")
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.stop()

def plot_scatter(df: pd.DataFrame, x: str, y: str):
    '''plot scatter and allow customised options'''
    title = f"Scatter Plot: {y} vs {x}"
    x_label = x
    y_label = y
    point_color = "#1f77b4"
    point_size = 50
    point_alpha = 0.7
    point_style = "o"
    show_trendline = False
    with st.expander("Customisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            point_color = st.color_picker("Point color", point_color, key=f"scatter_color_{x}_{y}")
            point_size = st.slider("Point size", 10, 200, 50, key=f"scatter_size_{x}_{y}")
            point_alpha = st.slider("Transparency", 0.1, 1.0, 0.7, key=f"scatter_alpha_{x}_{y}")
        with col2:
            title = st.text_input("Chart title", value=title, key=f"scatter_title_{x}_{y}")
            x_label = st.text_input("x-axis label", value=x, key=f"scatter_xlabel_{x}_{y}")
            y_label = st.text_input("y-axis label", value=y, key=f"scatter_ylabel_{x}_{y}")
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                show_trendline = st.checkbox("Show trendline", False, key=f"scatter_trend_{x}_{y}")
        point_style = st.selectbox("Marker style", ['o', 's', '^', 'D', 'v', 'p', '*', '+'], index=0, key=f"scatter_marker_{x}_{y}")
    fig, ax = plt.subplots(figsize=(10,6))
    try:
        ax.scatter(df[x], df[y], c=point_color, s=point_size, alpha=point_alpha, marker=point_style)
        if show_trendline:
            z = np.polyfit(df[x], df[y], 1)
            p = np.poly1d(z)
            ax.plot(df[x], p(df[x]), color="red", linestyle="--", label="Trendline")
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(title, fontsize=12, pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, f"scatter_plot_{x}_vs_{y}.png", "Download Scatter Plot")
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.stop()

def plot_histogram(df: pd.DataFrame, columns: list, bins: int, mode: str):
    '''plot histogram and allow customised options'''
    title = "Histogram"
    with st.expander("Customisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            hist_type = st.selectbox("Histogram type:", options=['bar', 'barstacked', 'step', 'stepfilled'], index=0, key="hist_type")
            hist_align = st.selectbox("Alignment:", options=['left', 'mid', 'right'], index=1, key="hist_align")
            hist_orientation = st.selectbox("Orientation", options=['vertical', 'horizontal'], index=0, key="hist_orientation")
            hist_alpha = st.slider("Transparency:", 0.1, 1.0, 0.7, key="hist_alpha")
        with col2:
            default_colors = [
                '#1f77b4', '#ff7f0e','#2ca02c',  '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            col_fill, col_stack = st.columns(2)
            with col_fill:
                fill_bar = st.checkbox("Fill bar with color", value=True, key="hist_fill")
            with col_stack:
                if len(columns) > 1 and mode == "Single Axes":
                    stack_bar = st.checkbox("Stack bars", value=False, key="hist_stacked")
                else:
                    stack_bar = False
            use_same_color = st.checkbox("Use same color for all", value=False, key="hist_same_color")
            if use_same_color:
                hist_color = st.color_picker("Select color for all histograms", "#1f77b4", key="hist_color_all")
                colors = [hist_color] * len(columns)
            else:
                colors = [st.color_picker(f"Color for {col}", default_colors[i % len(default_colors)], key=f"hist_color_{col}") for i, col in enumerate(columns)]
    if mode == "Subplots":
        n_cols = 2
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * len(columns)), squeeze=False)
        axes = axes.flatten()
        for ax in axes[len(columns):]:
            ax.set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax] * len(columns)
    try:
        if stack_bar and mode == "Single Axes":
            ax.hist(
                [df[col] for col in columns],
                bins=bins,
                color=colors,
                alpha=hist_alpha,
                align=hist_align,
                fill=fill_bar,
                orientation=hist_orientation,
                histtype=hist_type,
                label=columns,
                edgecolor='white' if fill_bar else colors
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            for i, (col, ax) in enumerate(zip(columns, axes)):
                ax.hist(
                    df[col],
                    bins=bins,
                    color=colors[i],
                    alpha=hist_alpha,
                    align=hist_align,
                    fill=fill_bar,
                    orientation=hist_orientation,
                    stacked=False,
                    histtype=hist_type,
                    label=col if (mode == "Single Axes") else None,
                    edgecolor='white' if fill_bar else colors[i]
                )
        for i, (col, ax) in enumerate(zip(columns, axes)):
            ax.set_ylabel("Frequency" if hist_orientation == 'vertical' else col)
            ax.set_xlabel(col if hist_orientation == 'vertical' else "Frequency")
            if mode == "Subplots":
                ax.set_title(col, fontsize=12, pad=20)
            elif i == 0:
                ax.set_title(title, fontsize=12, pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, f"histogram.png", "Download Histogram")
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.stop()

def plot_box(df: pd.DataFrame, columns: list):
    '''plot box and allow customised options'''
    title = "Boxplot"
    x_label = ""
    y_label = ""
    with st.expander("Customisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            box_orientation = st.selectbox("Orientation:", ['vertical', 'horizontal'], key="box_orientation")
            box_notch = st.checkbox("Notched boxplot", key="box_notch")
            show_means = st.checkbox("Show means", key="box_show_means")
            show_fliers = st.checkbox("Show outliers", value=True, key="box_show_fliers")
            x_label = st.text_input("x-axis label", value=x_label, key=f"xlabel_box")
            y_label = st.text_input("y-aixs label", value=y_label, key=f"ylabel_box")
        with col2:
            title = st.text_input("Plot title", value=title, key=f"title_box")
            default_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf'
            ]
            use_same_color = st.checkbox("Use same color for all", value=False, key="box_same_color")
            if use_same_color:
                box_color = st.color_picker("Select color for all histograms", "#1f77b4", key="box_color_all")
                colors = [box_color] * len(columns)
            else:
                colors = [
                    st.color_picker(
                        f"Color for {col}",
                        default_colors[i % len(default_colors)],
                        key=f"box_color_{col}"
                    )
                    for i, col in enumerate(columns)
                ]
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(
            df[columns].dropna(),
            patch_artist=True,
            vert=(box_orientation == 'vertical'),
            notch=box_notch,
            showmeans=show_means,
            meanline=True if show_means else False,
            showfliers=show_fliers
        )
        for box, color in zip(bp['boxes'], colors):
            box.set(facecolor=color, alpha=0.7)
        if box_orientation == 'vertical':
            ax.set_xticklabels(columns)
        else:
            ax.set_yticklabels(columns)
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, f"boxplot.png", "Download Boxplot")
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.stop()

def plot_violin(df: pd.DataFrame, val_col: list, group_by: str = None):
    '''plot violin and allow customised options'''
    title = f"Violin Plot of {', '.join(val_col)}"
    x_label = f"{group_by}" if group_by else ""
    y_label = "Value" if group_by else ""
    if group_by:
        title += f" grouped by {group_by}"
    with st.expander("Customisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            inner_style = st.selectbox("Inner style", ['box', 'quartile', 'point', 'stick', None], index=0, key="violin_inner")
            bw_method = st.selectbox("Smoothing (Bandwidth)", ['scott', 'silverman', 'custom'], index=0, key="violin_bw")
            if bw_method == 'custom':
                bw_adjust = st.slider("Bandwidth adjustment", 0.1, 2.0, 1.0, 0.1, key="violin_bw_adjust")
            else:
                bw_adjust = bw_method # Use the string 'scott' or 'silverman'
            v_width = st.slider("Violin width:", 0.1, 2.0, 0.8, 0.1, key="v_width")
            split_violins = st.checkbox("Split violins", value=False, key="violin_split")
            if split_violins and (not group_by or len(val_col) != 2):
                st.warning("Splitting violins is most effective when grouping by a category with exactly two value columns.")
        with col2:
            color_palette = st.selectbox("Color palette", ['Pastel1', 'Set2', 'tab10', 'viridis', 'plasma'], index=1, key="violin_color")
            title = st.text_input("Plot title", value=title, key="violin_title")
            x_label = st.text_input("x-axis label", value=x_label, key=f"xlabel_violin")
            y_label = st.text_input("y-axis label", value=y_label, key=f"ylabel_violin")
    try:
        cols_to_check = val_col + ([group_by] if group_by else [])
        plot_df = df[cols_to_check].dropna()
        if plot_df.empty:
            st.error("No valid data to plot after removing missing values.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        if group_by:
            long_df = plot_df.melt(id_vars=[group_by], value_vars=val_col, var_name='Variable', value_name='Value')
            sns.violinplot(
                data=long_df,
                x=group_by,
                y='Value',
                hue='Variable',
                split=split_violins,
                inner=inner_style,
                palette=color_palette,
                bw_method=bw_adjust,
                width=v_width,
                ax=ax
            )
        else:
            sns.violinplot(
                data=plot_df[val_col],
                inner=inner_style,
                palette=color_palette,
                bw_method=bw_adjust,
                width=v_width,
                ax=ax
            )
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.xticks(rotation=0)
        if group_by:
             ax.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, "violin_plot.png", "Download Violin Plot")
    except Exception as e:
        st.error(f"Error creating violin plot: {str(e)}")
        st.stop()

def plot_pie(df: pd.DataFrame, cat_col: str, val_col: str):
    '''plot pie and allow customised options'''
    title = f"Distribution by {cat_col}"
    with st.expander("Customisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            show_percent = st.checkbox("Show percentages", key="pie_show_percent")
            autopct = st.selectbox("Percentage format", ['%1.1f%%', '%1.0f%%', '%.2f%%'], index=0, key="pie_autopct") if show_percent else None
            if autopct:
                pct_distance = st.slider("Text distance", 0.0, 1.5, 0.6, 0.1, key="pie_pctdistance")
            show_labels = st.checkbox("Show labels", key="pie_show_labels")
            show_legend = st.checkbox("Show legend", key="pie_legend")
            title = st.text_input("Plot title", value=title, key=f"title_pie")
        with col2:
            explode_option = st.radio("Explode mode:", ["None", "All wedges", "Single wedge"], index=0, help="Separate slice(s) from center", key="pie_explore")
            if explode_option == "All wedges":
                explode_value = st.slider("Explode amount", 0.0, 0.2, 0.0, 0.01, key="pie-explore_value")
            elif explode_option == "Single wedge":
                selected_wedge = st.selectbox("Wedge to explode:", df[cat_col].unique(), key="pie_selected_wedge")
                explode_value = st.slider("Explode amount", 0.0, 0.5, 0.2, 0.01, key="pie-explore_value")
            start_angle = st.slider("Start angle", 0, 360, 90, help="Starting angle of first slice", key="pie_angle")
            color_palette = st.selectbox("Color palette", ['Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'hsv', 'rainbow'], index=5, key="pie_color")
    try:
        plot_df = df.groupby(cat_col)[val_col].sum().reset_index()
        if plot_df.empty or plot_df[val_col].sum() <= 0:
            st.error("No valid, positive data to plot for the pie chart.")
            return
        categories = plot_df[cat_col]
        if explode_option == "All wedges":
            explode = [explode_value] * len(plot_df)
        elif explode_option == "Single wedge":
            explode = [explode_value if cat == selected_wedge else 0 for cat in categories]
        else:
            explode = None
        colors = plt.get_cmap(color_palette)(np.linspace(0, 1, len(categories)))
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, *_ = ax.pie(
            plot_df[val_col],
            labels=plot_df[cat_col] if show_labels else None,
            autopct=autopct if show_percent else None,
            pctdistance=pct_distance if autopct else None,
            startangle=start_angle,
            explode=explode,
            colors=colors
        )
        if show_legend:
            ax.legend(wedges, plot_df[cat_col],title=cat_col,loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax.axis('equal')
        ax.set_title(title, fontsize=12, pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, f"piechart.png", "Download Pie Chart")
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.stop()

def plot_stack(df: pd.DataFrame, x: str, y: list, type: str):
    '''plot stack plot and allow customised options'''
    if x in y:
        st.warning("⚠️ Warning: The same column cannot be used for both X-axis and Y-axis. Please select different columns.")
        return
    title = f"Stacked {type} Plot"
    x_label = x
    y_label = ""
    with st.expander("Optimisation Options"):
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Transparency", 0.3, 1.0, 0.8, key="stack_alpha")
            if type == "Area":
                line_width = st.slider("Line width", 0, 3, 1, key="stack_width")
            else:
                bar_width = st.slider("Bar width", 0.1, 1.0, 0.8, key="stack_width")
            show_grid = st.checkbox("Show grid", True, key="stack_grid")
            show_legend = st.checkbox("Show legend", True, key="stack_legend")
            baseline_options = st.selectbox("Baseline", ['zero', 'sym', 'wiggle', 'weighted_wiggle'], index=0, key="stack_baseline")
        with col2:
            color_palette = st.selectbox("Color palette", ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'tab10', 'tab20', 'hsv', 'rainbow'], index=5, key="stack_color")
            title = st.text_input("Chart title", value=title, key=f"title_stack_{x}")
            x_label = st.text_input("x-axis label", value=x_label, key=f"xlabel_stack_{x}")
            y_label = st.text_input("y-aixs label", value=y_label, key=f"ylabel_stack_{x}")
    try:
        y = [col for col in y if col != x]
        plot_df = df[[x] + y].copy()
        if plot_df.isnull().values.any():
            st.warning("Warning: Data contains missing values. Rows with NA values will be dropped.")
        plot_df = plot_df.dropna()
        if plot_df.empty:
            st.error("No valid data after dropping null values")
            return
        x_values = np.array(plot_df[x]).flatten()
        if x_values.ndim > 1:
            x_values = x_values.ravel() 
        y_values = plot_df[y].values.T
        if pd.api.types.is_numeric_dtype(plot_df[x]) or pd.api.types.is_datetime64_any_dtype(plot_df[x]):
            sort_idx = np.argsort(x_values)
            x_values = x_values[sort_idx]
            y_values = y_values[:, sort_idx]
        colors = plt.get_cmap(color_palette)(np.linspace(0, 1, len(y)))
        fig, ax = plt.subplots(figsize=(10, 6))
        if type == "Area":
            ax.stackplot(
                x_values,
                y_values,
                labels=y,
                colors=colors,
                alpha=alpha,
                linewidth=line_width,
                baseline=baseline_options
            )
        else:
            bottom = np.zeros(len(x_values))
            for i, col in enumerate(y):
                ax.bar(
                    x_values,
                    plot_df[col].values,
                    bottom=bottom,
                    width=bar_width,
                    color=colors[i],
                    alpha=alpha,
                    linewidth=0.5,
                    label=col
                )
                bottom += plot_df[col].values
                if pd.api.types.is_datetime64_any_dtype(plot_df[x]):
                    fig.autofmt_xdate()
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if pd.api.types.is_datetime64_any_dtype(plot_df[x]):
            fig.autofmt_xdate()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if show_grid:
            ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        get_image_download_link(fig, f"stackplot.png", "Download Stack Plot")
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        st.stop()

def reset_session_state():
    st.session_state.chart_selection = None

def main():
    st.set_page_config(page_title="Visualise Data", layout="wide")

    if "initial_df" not in st.session_state:
        st.session_state.initial_df = None
    if "working_df" not in st.session_state:
        st.session_state.working_df = None

    left_col, right_col = st.columns([8,0.5])

    with left_col:
        st.header("Data Visualisation")
        st.caption("""**Uncover the story within your data.** Create charts to identify patterns, trends, and relationships, forming the foundation for rigorous statistical testing.""")

    st.markdown("---")
    
    if "initial_df" not in st.session_state or st.session_state.initial_df is None:
        st.warning("No data found. Please upload a dataset or use sample dataset on the 'Data Upload' page.")
        if st.button("Go to Upload Data", type='primary'):
            st.switch_page("pages/2_upload_data.py")
        return

    if 'working_df' not in st.session_state or st.session_state.working_df is None:
        st.session_state.working_df = st.session_state.initial_df.copy()

    with right_col:
        if st.button("Reset", use_container_width=True):
            reset_session_state()
            st.rerun()

    # preview data
    st.subheader("👀 Data Preview")
    st.write("How many rows of data would you like to view?")
    preview_data(st.session_state.working_df, key="preview_initial")

    st.markdown("---")

    st.subheader("Build Your Chart")
    chart_options = ["Line", "Bar", "Scatter", "Histogram", "Box", "Violin", "Pie", "Stack"]
    chart_checks = {
        "Line": is_suitable_for_line,
        "Bar": is_suitable_for_bar,
        "Scatter": is_suitable_for_scatter,
        "Histogram": is_suitable_for_histogram,
        "Box": is_suitable_for_box,
        "Violin": is_suitable_for_violin,
        "Pie": is_suitable_for_pie,
        "Stack": is_suitable_for_stack
    }

    chart_function_dict = {
        "Line": plot_line,
        "Bar": plot_bar,
        "Scatter": plot_scatter,
        "Histogram": plot_histogram,
        "Box": plot_box,
        "Violin": plot_violin,
        "Pie": plot_pie,
        "Stack": plot_stack
    }

    # only show options suitable for plotting
    valid_charts = [chart for chart in chart_options if chart_checks[chart](st.session_state.working_df)]

    plt.style.use('dark_background')
    chart_df = st.session_state.working_df.copy()
    df_cols_option = chart_df.columns.to_list()

    if valid_charts:
        chart_selection = st.pills("Select chart type(s)", valid_charts, selection_mode="multi", default=None)
        if chart_selection:
            st.session_state.chart_selection = chart_selection
            st.write(f"You selected: {', '.join(chart_selection)}")

            for i, chart in enumerate(chart_selection):
                st.markdown(f"### {chart} Chart")
                # plot charts
                if chart == "Line":
                    select_axis_to_plot(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
                elif chart == "Bar":
                    select_axis_to_plot(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
                elif chart == "Scatter":
                    select_axis_to_plot(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
                elif chart == "Histogram":
                    select_axis_histogram(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
                elif chart == "Box":
                    select_axis_box(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
                elif chart == "Violin":
                    select_axis_violin(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
                elif chart == "Pie":
                    select_axis_pie(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
                elif chart == "Stack":
                    select_axis_stack(chart_function_dict, chart, chart_df, df_cols_option, key_suffix=str(i))
        else:
            st.write("No chart type selected.")

    else:
        st.warning("⚠️ No suitable chart types found for the current dataset's column types.")

    st.markdown("---")

    st.markdown(f"**Continue Your Analysis**")
    st.write("**Have you discovered a potential pattern?** Proceed to statistical analysis to test its significance and robustness.")
    col_option1, col_option2, col_option3= st.columns(3)
    with col_option1:
        if st.button("🔍 Explore Data"):
            st.switch_page("pages/3_data_analysis.py")
    with col_option2:
        if st.button("📉 Statistical Analysis"):
            st.switch_page("pages/5_statistical_analysis.py")
    with col_option3:
        if st.button("🤖 Machine Learning"):
            st.switch_page("pages/6_ML.py")

if __name__ == "__main__":
    main()

