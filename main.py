import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os
from io import StringIO, BytesIO

# Initialize ChatGroq with API key from environment
llm = ChatGroq(model_name='llama3-70b-8192', api_key="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")

def read_file(uploaded_file):
    """
    Read the uploaded file based on its type (Excel or CSV).
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def process_data(df, query):
    """
    Process the DataFrame using SmartDataframe and ChatGroq.
    """
    try:
        data = df
        smart_df = SmartDataframe(data, config={'llm': llm})
        return smart_df.chat(query)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def main():
    st.title('ClearVision- By Rajesh')
    
    # Add a brief description
    st.write("Upload your data file and ask questions to get insights!")
    
    uploaded_file = st.file_uploader("Upload a file (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = read_file(uploaded_file)
        
        if df is not None:
            # Display data preview and available columns
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            st.write("### Available Columns:")
            st.write(", ".join(df.columns.tolist()))
            
            # Query section with example
            query = st.text_input(
                "Enter your question",
                placeholder="Example: 'Show a pie chart for sales distribution' or 'Calculate the average revenue'"
            )
            
            if st.button('Analyze Data'):
                if query:
                    with st.spinner('Processing...'):
                        result = process_data(df, query)
                        
                        # Display results
                        st.write("### Results")
                        if isinstance(result, (int, float)):
                            st.write(result)
                        elif isinstance(result, str) and result.endswith('.png'):
                            st.image(result, use_column_width=True)
                        elif isinstance(result, str):
                            st.write(result)
                        else:
                            try:
                                st.image(result, use_column_width=True)
                            except:
                                st.write(result)
                else:
                    st.warning("Please enter a question before analyzing.")

if __name__ == "__main__":
    main()
