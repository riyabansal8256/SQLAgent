from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import sqlite3
from groq import Groq
import pandas as pd
import tempfile
import plotly.express as px
import plotly.graph_objects as go

# Configure Groq API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()

# Function to create database from uploaded Excel file
def excel_to_sqlite(excel_file, table_name="UPLOADED_DATA"):
    """Convert uploaded Excel file to SQLite database"""
    # Read Excel file
    df = pd.read_excel(excel_file)
    
    # Create a temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db_path = temp_db.name
    temp_db.close()
    
    # Connect to SQLite database and create table
    conn = sqlite3.connect(temp_db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    return temp_db_path, df.columns.tolist(), df

# Function to retrieve query from the database
def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    return rows

# Function to generate dynamic prompt based on table schema
def generate_prompt(table_name, columns):
    column_list = ", ".join(columns)
    prompt = f"""
    You are an expert in converting English questions to SQL query!
    The SQL database has the name {table_name} and has the following columns - {column_list}
    
    Important: 
    - The table name is {table_name}
    - Column names are case-sensitive and exactly as shown: {column_list}
    - Do not include ``` or 'sql' in your output
    - Return only the SQL query
    
    Examples:
    - "How many records are there?" → SELECT COUNT(*) FROM {table_name};
    - "Show all data" → SELECT * FROM {table_name};
    """
    return prompt

# Function to create various types of charts
def create_chart(df, chart_type, x_col=None, y_col=None, color_col=None):
    """Create different types of charts based on the data and user selection"""
    
    if chart_type == "Bar Chart":
        if x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
        else:
            # Default: show count of first column
            fig = px.bar(df[df.columns[0]].value_counts().reset_index(), 
                        x='index', y=df.columns[0], 
                        title=f"Count of {df.columns[0]}")
    
    elif chart_type == "Line Chart":
        if x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                         title=f"{y_col} over {x_col}", markers=True)
        else:
            fig = px.line(df, y=df.select_dtypes(include=['number']).columns[0] if len(df.select_dtypes(include=['number']).columns) > 0 else df.columns[0],
                         title="Line Chart")
    
    elif chart_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=x_col or numeric_cols[0], 
                           y=y_col or numeric_cols[1], 
                           color=color_col,
                           title=f"Scatter Plot: {x_col or numeric_cols[0]} vs {y_col or numeric_cols[1]}")
        else:
            st.warning("Need at least 2 numeric columns for scatter plot")
            return None
    
    elif chart_type == "Pie Chart":
        if x_col:
            value_counts = df[x_col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, 
                        title=f"Distribution of {x_col}")
        else:
            # Default: use first column
            value_counts = df[df.columns[0]].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Distribution of {df.columns[0]}")
    
    elif chart_type == "Box Plot":
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig = px.box(df, y=y_col or numeric_cols[0], x=x_col,
                        title=f"Box Plot of {y_col or numeric_cols[0]}")
        else:
            st.warning("Need numeric columns for box plot")
            return None
    
    elif chart_type == "Histogram":
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig = px.histogram(df, x=x_col or numeric_cols[0], 
                             color=color_col,
                             title=f"Distribution of {x_col or numeric_cols[0]}")
        else:
            st.warning("Need numeric columns for histogram")
            return None
    
    elif chart_type == "Heatmap":
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title="Correlation Heatmap")
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap")
            return None
    
    else:
        return None
    
    # Update layout for better appearance
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        showlegend=True,
        height=500
    )
    
    return fig

# Streamlit App
st.set_page_config(page_title="SQL Query Generator with Visualization", layout="wide")
st.header("Natural Language to SQL Query Generator with Visualization")

# Sidebar for file upload
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Process the uploaded file
        try:
            db_path, columns, df = excel_to_sqlite(uploaded_file, "UPLOADED_DATA")
            st.session_state['db_path'] = db_path
            st.session_state['table_name'] = "UPLOADED_DATA"
            st.session_state['columns'] = columns
            st.session_state['df'] = df
            
            st.success("File uploaded successfully!")
            st.write("**Columns detected:**")
            for col in columns:
                st.write(f"- {col}")
            
            # Show preview of data
            st.write("**Data Preview:**")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Option to use sample data
    st.divider()
    if st.button("Use Sample Student Data"):
        # Create sample database
        conn = sqlite3.connect("sample_student.db")
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS STUDENT (
            NAME TEXT,
            CLASS TEXT,
            SECTION TEXT,
            MARKS INTEGER
        );
        """)
        
        # Insert sample data
        cursor.execute("DELETE FROM STUDENT")
        students = [
            ('Krish', 'Data Science', 'A', 90),
            ('Sudhanshu', 'Data Science', 'B', 100),
            ('Darius', 'Data Science', 'A', 86),
            ('Vikash', 'DEVOPS', 'A', 50),
            ('Dipesh', 'DEVOPS', 'A', 35)
        ]
        cursor.executemany("INSERT INTO STUDENT VALUES (?, ?, ?, ?)", students)
        conn.commit()
        conn.close()
        
        st.session_state['db_path'] = "sample_student.db"
        st.session_state['table_name'] = "STUDENT"
        st.session_state['columns'] = ['NAME', 'CLASS', 'SECTION', 'MARKS']
        
        # Create DataFrame for preview
        sample_df = pd.DataFrame(students, columns=['NAME', 'CLASS', 'SECTION', 'MARKS'])
        st.session_state['df'] = sample_df
        
        st.success("Sample data loaded!")
        st.write("**Columns:** NAME, CLASS, SECTION, MARKS")
        st.dataframe(sample_df)

# Main area
if 'db_path' in st.session_state:
    st.subheader(f"Current Table: {st.session_state['table_name']}")
    st.write(f"Columns: {', '.join(st.session_state['columns'])}")
    
    # Query input
    question = st.text_input("Enter your question:", key="input", 
                           placeholder="e.g., How many records are there?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Generate Query", type="primary")
    
    # Process query
    if submit and question:
        try:
            # Generate dynamic prompt
            prompt = generate_prompt(st.session_state['table_name'], 
                                   st.session_state['columns'])
            
            # Get SQL query from Groq
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Display generated query
            st.subheader("Generated SQL Query:")
            st.code(sql_query, language='sql')
            
            # Execute query
            try:
                data = read_sql_query(sql_query, st.session_state['db_path'])
                st.subheader("Query Results:")
                
                if data:
                    # Convert to DataFrame for better display
                    # Get column names from the query
                    conn = sqlite3.connect(st.session_state['db_path'])
                    cursor = conn.cursor()
                    cursor.execute(sql_query)
                    col_names = [description[0] for description in cursor.description]
                    conn.close()
                    
                    result_df = pd.DataFrame(data, columns=col_names)
                    st.dataframe(result_df)
                    
                    # Store result in session state for visualization
                    st.session_state['result_df'] = result_df
                    st.session_state['query_executed'] = True
                    
                else:
                    st.info("Query executed successfully but returned no results.")
                    st.session_state['query_executed'] = False
                    
            except sqlite3.Error as e:
                st.error(f"Database error: {str(e)}")
                st.write("Please check if the SQL query is correct.")
                st.session_state['query_executed'] = False
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state['query_executed'] = False
    
    # Visualization Section
    if st.session_state.get('query_executed', False) and 'result_df' in st.session_state:
        st.divider()
        st.subheader("📊 Data Visualization")
        
        result_df = st.session_state['result_df']
        
        # Check if we have data to visualize
        if len(result_df) > 0:
            viz_col1, viz_col2 = st.columns([3, 2])
            
            with viz_col2:
                st.write("**Visualization Options**")
                
                # Chart type selection
                chart_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", 
                             "Box Plot", "Histogram", "Heatmap"]
                selected_chart = st.selectbox("Select Chart Type", chart_types)
                
                # Column selection for charts
                columns = result_df.columns.tolist()
                numeric_columns = result_df.select_dtypes(include=['number']).columns.tolist()
                
                x_axis = None
                y_axis = None
                color_by = None
                
                if selected_chart in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                    x_axis = st.selectbox("X-Axis", columns, key="x_axis")
                    if numeric_columns:
                        y_axis = st.selectbox("Y-Axis", numeric_columns, key="y_axis")
                    color_by = st.selectbox("Color By (optional)", [None] + columns, key="color")
                
                elif selected_chart == "Pie Chart":
                    x_axis = st.selectbox("Category", columns, key="pie_category")
                
                elif selected_chart in ["Box Plot", "Histogram"]:
                    if numeric_columns:
                        y_axis = st.selectbox("Value Column", numeric_columns, key="value_col")
                    x_axis = st.selectbox("Group By (optional)", [None] + columns, key="group_by")
                    if selected_chart == "Histogram":
                        color_by = st.selectbox("Color By (optional)", [None] + columns, key="hist_color")
                
                # Generate chart button
                if st.button("Generate Chart", type="secondary"):
                    with viz_col1:
                        try:
                            fig = create_chart(result_df, selected_chart, x_axis, y_axis, color_by)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Option to download chart
                                if st.button("Download Chart as Image"):
                                    fig.write_image("chart.png")
                                    with open("chart.png", "rb") as file:
                                        st.download_button(
                                            label="Download PNG",
                                            data=file,
                                            file_name="sql_result_chart.png",
                                            mime="image/png"
                                        )
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")
            
            with viz_col1:
                # Show data summary
                st.write("**Data Summary**")
                st.write(f"Total rows: {len(result_df)}")
                st.write(f"Total columns: {len(result_df.columns)}")
                
                # Show basic statistics for numeric columns
                if numeric_columns:
                    st.write("**Numeric Column Statistics:**")
                    st.dataframe(result_df[numeric_columns].describe())
        else:
            st.info("No data available for visualization")
            
else:
    st.info("👈 Please upload an Excel file or use sample data from the sidebar to get started!")
    
    # Show example format
    with st.expander("Excel File Format Example"):
        st.write("Your Excel file should have headers in the first row. Here's an example:")
        example_df = pd.DataFrame({
            'Product': ['Laptop', 'Mouse', 'Keyboard'],
            'Category': ['Electronics', 'Accessories', 'Accessories'],
            'Price': [999, 25, 45],
            'Stock': [50, 200, 150]
        })
        st.dataframe(example_df)

# Clean up temporary files on app close
if st.session_state.get('db_path') and 'sample' not in st.session_state['db_path']:
    import atexit
    atexit.register(lambda: os.unlink(st.session_state['db_path']) if os.path.exists(st.session_state['db_path']) else None)
