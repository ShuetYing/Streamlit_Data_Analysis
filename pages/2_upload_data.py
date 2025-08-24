import streamlit as st
import json
import pandas as pd
import io
import time
from typing import Optional
from sqlalchemy import create_engine 
from sqlalchemy.exc import SQLAlchemyError

def init_session_state():
    if 'pasted_data' not in st.session_state:
        st.session_state.pasted_data = ""
    if 'initial_df' not in st.session_state:
        st.session_state.initial_df = None
    if 'working_df' not in st.session_state:
        st.session_state.working_df = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'show_preview' not in st.session_state:
        st.session_state.show_preview = False
    if 'show_raw' not in st.session_state:
        st.session_state.show_raw = False
    if 'selected' not in st.session_state:
        st.session_state.selected = "paste_data"

def detect_text_format(pasted_data: str) -> str:

	try:
		json.loads(pasted_data)
		return "JSON"
	except ValueError:
		pass

	delimiters = {
		"CSV (comma-delimited)": ',',
		"TSV (tab-delimited)": '\t',
		"Semicolon-delimited": ';'
	}

	for name, delim in delimiters.items():
		try:
			df = pd.read_csv(io.StringIO(pasted_data), sep=delim)
			if len(df.columns) > 1:
				return name
		except:
			continue

	try:
		df = pd.read_csv(io.StringIO(pasted_data), sep=r'\s+', engine='python')
		if len(df.columns) > 1:
			return "Space-delimited text"
	except:
		pass

	return "Unknown format"

def validate_data(df):
	if df.empty:
		return "Empty dataframe"
	elif df.isna().all().all():
		return "All values are NA"
	return None

def process_data(pasted_data: str) -> tuple[Optional[pd.DataFrame], str]:
	format = detect_text_format(pasted_data)
	try:
		if format == "JSON":
			df = pd.read_json(io.StringIO(pasted_data))
		elif "delimited" in format:
			sep = ',' if "CSV" in format else '\t' if "TSV" in format else ';'
			df = pd.read_csv(io.StringIO(pasted_data), sep=sep, engine='python')
		elif format == "Space-delimited text":
			df = pd.read_csv(io.StringIO(pasted_data), sep=r'\s+', engine='python')
		else:
			df = None
		if df is not None:
			df_check = validate_data(df)
		return df, format, df_check
	except Exception as e:
		return None, str(e), None

def process_file(uploaded_file) -> tuple[Optional[pd.DataFrame], str]:
	try:
		if uploaded_file.name.endswith(".json"):
			df = pd.read_json(uploaded_file)
			format = "JSON"
		elif uploaded_file.name.endswith(".csv"):
			df = pd.read_csv(uploaded_file, sep=r'[,;]', engine='python')
			format = "CSV"
		elif uploaded_file.name.endswith((".tsv", ".txt")):
			df = pd.read_csv(uploaded_file, sep='\t')
			format = "TSV/TXT"
		elif uploaded_file.name.endswith(("xlsx", "xls")):
			df = pd.read_excel(uploaded_file)
			format = "Excel"
		else:
			df = None
			format = "Unsupported"
		if df is not None:
			df_check = validate_data(df)
		return df, format, df_check
	except Exception as e:
		return None, str(e), None

def db_connection_string(db_type, params) -> str:
	host = params.get("host")
	username = params.get("user")
	password = params.get("password")
	database = params.get("database")

	if db_type == "MySQL":
		import pymysql
		port = params.get("port", 3306)
		return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
	elif db_type == "PostgreSQL":
		import psycopg2
		port = params.get("port", 5432)
		return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
	elif db_type == "Microsoft SQL Server":
		driver = params.get("driver", "ODBC Driver 17 for SQL Server")
		return f"mssql+pyobdc://{username}:{password}@{host}/{database}?driver={driver.replace(' ', '+')}"
	elif db_type == "SQLite":
		return f"sqlite:///{database}"

@st.cache_resource
def db_connection(connection_string):
	try:
		engine = create_engine(connection_string)
		with engine.connect() as conn:
			conn.execute("SELECT 1")
		st.success("successfully connected to database!")
		return engine
	except SQLAlchemyError as e:
		st.error(f"Error connecting to the database: {e}")
		return None
	except Exception as e:
		st.error(f"Error connecting to database: {e}")
		return None
	
def reset_session_state():
	st.session_state.pasted_data = ""
	st.session_state.initial_df = None
	st.session_state.working_df = None
	st.session_state.uploaded_file = None
	st.session_state.show_preview = False
	st.session_state.show_raw = False

def main():
	st.set_page_config(page_title="Data Loader", layout="wide")

	init_session_state()

	st.header("Data Loader")
	st.caption("""**Ingest your data from multiple sources to begin your analysis.** Select the method that best suits your workflow to load a dataset for exploration, visualisation, and modeling.""")
	st.markdown("---")

	if "selected" not in st.session_state:
		st.session_state.selected = "paste_data"
	
	if st.button("Reset form"):
		reset_session_state()
		st.rerun()

	col1, col2 = st.columns([1.5, 4])

	with col1:
		if st.button("Paste data", 
			icon=":material/content_paste:", 
			key="paste_btn", 
			help="Paste data", 
			use_container_width=True,
			type="primary" if st.session_state.selected == "paste_data" else "secondary"):
			st.session_state.selected = "paste_data"
			reset_session_state()
			st.rerun()

		if st.button("Upload file", 
			icon=":material/upload:", 
			key="upload_btn", 
			help="Upload CSV, Excel, or JSON files", 		
			use_container_width=True,
			type="primary" if st.session_state.selected == "upload_file" else "secondary"):
			st.session_state.selected = "upload_file"
			reset_session_state()
			st.rerun()

		if st.button("Load from database", 
			icon=":material/database:",
			key="db_btn", 
			help="Connect to SQL database", 
			use_container_width=True,
			type="primary" if st.session_state.selected == "database" else "secondary"):
			st.session_state.selected = "database"
			reset_session_state()
			st.rerun()

	with col2:
		with st.container(border=True):
			if st.session_state.selected == "paste_data":
				pasted_data = st.text_area("Paste your data here (supports CSV, TSV, JSON, space-delimited):", 
					height=300, 
					placeholder="Paste your data here...",
					value=st.session_state.pasted_data)

				if st.button("Process pasted data", type="primary"):
					if pasted_data.strip():
						with st.spinner("Processing data..."):
							time.sleep(0.5)
							df, format, df_check = process_data(pasted_data)
							if format == "Unknown format":
								st.error("Could not detect data format! Please check your input.")
							elif df_check == "Empty dataframe":
								st.warning(df_check)
							elif df_check == "All values are NA":
								st.warning(df_check)
							elif df is not None:
								st.session_state.pasted_data = pasted_data
								st.session_state.initial_df = df
								st.session_state.working_df = df
								st.success(f"Detected format: {format}")
								st.markdown("---")
								st.write(f"**Rows loaded:** {len(df)}")
								st.write(f"**Columns:** {', '.join(df.columns)}")
							else:
								st.error(f"Error processing data: {format}")
					else:
						st.error("Please paste some data first!")

				if st.session_state.initial_df is not None:
					st.session_state.show_preview = st.checkbox("Show data preview", 
						value=st.session_state.show_preview)
					st.session_state.show_raw = st.checkbox("Show raw data",
						value=st.session_state.show_raw)

					if st.session_state.show_preview:
						st.dataframe(st.session_state.initial_df.head(), hide_index=True)
					if st.session_state.show_raw:
						st.code(st.session_state.pasted_data)
			
			elif st.session_state.selected == "upload_file":
				uploaded_file = st.file_uploader("Choose a file", 
					type = ["xlsx", "txt", "tsv", "csv", "json"])

				if st.button("Process uploaded file", type="primary"):
					if uploaded_file is not None:
						with st.spinner("Processing file..."):
							time.sleep(0.5)
							df, format, df_check = process_file(uploaded_file)
							st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size/1024:.2f} KB)")
							if format == "Unsupported":
								st.error("Unsupported file format!")
							elif df_check == "Empty dataframe":
								st.warning(df_check)
							elif df_check == "All values are NA":
								st.warning(df_check)
							elif df is not None:
								st.session_state.uploaded_file = uploaded_file
								st.session_state.initial_df = df
								st.session_state.working_df = df
								st.success(f"Successfully processed as {format}!")
							else:
								st.error(f"Error processing file: {format}")
					else:
						st.error("Please upload file!")

				if st.session_state.initial_df is not None:
					st.session_state.show_preview = st.checkbox("Show data preview", 
						value=st.session_state.show_preview)
					
					if st.session_state.show_preview:
						st.dataframe(st.session_state.initial_df.head(), hide_index=True)				
			
			elif st.session_state.selected == "database":
				db_type = st.selectbox("Database type", ["MySQL", "PostgreSQL", "Microsoft SQL Server", "SQLite"], 
						index=None, placeholder="Select database type")
				
				with st.form("database_connection"):
					col_db1, col_db2 = st.columns(2)

					with col_db1:
						host = st.text_input("Host")
						db_name = st.text_input("Database name")

					with col_db2:
						if db_type == "SQLite":
							username = st.text_input("Username", disabled=True, help="Username not required for SQLite")
							passwd = st.text_input("Password", type="password", disabled=True, help="Password not required for SQLite")
						else:
							username = st.text_input("Username")
							passwd = st.text_input("Password", type="password")
					if db_type == "SQLite" or db_type == "Microsoft SQL Server":
						port = st.text_input("Port", disabled=True, help="Port not required")
					else:
						port = st.text_input("Port", placeholder="Use dafault port if none is entered")

					driver = st.text_input("Driver", placeholder="Use default driver if none is entered") if db_type == "Microsoft SQL Server" else None
					query = st.text_area("SQL query", height=100)
				
					if st.form_submit_button("Connect & Query", type="primary"):
						if not db_type:
							st.error("Please select a database type!")
						else:
							missing = False
							if db_type == "SQLite":
								missing_fields = []
								input_values = {
									"Host": host,
									"Database name": db_name,
									"SQL query": query
								}
								for label, value in input_values.items():
									if not value.strip():
										missing_fields.append(label)
								if missing_fields:
									missing = True
									st.error(f"{' and '.join(missing_fields)} cannot be empty!")
							else:
								missing_fields = []
								input_values = {
									"Host": host,
									"Username": username,
									"Database name": db_name,
									"Password": passwd,
									"SQL query": query
								}
								for label, value in input_values.items():
									if not value.strip():
										missing_fields.append(label)
								if missing_fields:
									missing = True
									st.error(f"{' , '.join(missing_fields)} cannot be empty!")

							if not missing:	
								try:
									with st.spinner(f"Connecting to {db_type} database..."):
										time.sleep(0.5)
										connection_params = {
											'host': host,
											'user': username,
											'password': passwd,
											'database': db_name,
											'driver': driver if driver else None,
											'port': int(port) if port else None
										}
						
										connection_string = db_connection_string(db_type, connection_params)
										engine = db_connection(connection_string)

										if engine is not None:
											try:
												with engine.connect() as conn:
													df = pd.read_sql(query, con=conn)
													if not df.empty:
														st.session_state.initial_df = df
														st.session_state.working_df = df
														st.success("Query executed successfully!")
													else:
														st.warning("Query returned empty results!")
											except SQLAlchemyError as e:
												st.error(f"Failed to execute query due to SQLAlchemy error: {e}")
											except Exception as e:
												st.error(f"An unexpected error occurred while executing the query: {e}")
								except Exception as e:
									st.error(f"Operation failed: {e}")
								
								if st.session_state.initial_df is not None:
									st.session_state.show_preview = st.checkbox("Show data preview", 
										value=st.session_state.show_preview)
					
									if st.session_state.show_preview:
										st.dataframe(st.session_state.initial_df.head(), hide_index=True)	

		if st.session_state.initial_df is not None and not st.session_state.initial_df.empty:
			st.markdown(f"**Continue Your Analysis**")
			st.write("Your dataset is ready. Proceed to explore, visualise, or model your data.")
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
