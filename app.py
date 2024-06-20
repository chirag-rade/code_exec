import streamlit as st
import json
from Agents import run_multiple_turns

# Streamlit app
st.title("Notebook Evaluation App")

# File uploader
uploaded_file = st.file_uploader("Upload a .ipynb file", type="ipynb")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("uploaded_notebook.ipynb", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Parameters for the evaluation
    happy_cases_n = st.number_input("Number of happy cases", min_value=1, max_value=10, value=1)
    edge_cases_n = st.number_input("Number of edge cases", min_value=1, max_value=10, value=1)
    recursion_limit = st.number_input("Recursion limit", min_value=1, value=50)
    n_workers = st.number_input("Number of workers", min_value=1, value=4)
    n_workers_inner_thread = st.number_input("Number of inner thread workers", min_value=1, value=4)

    # Run the evaluation
    if st.button("Run Evaluation"):
        st.write("Running evaluation, please wait...")
        eval = run_multiple_turns(
            notebook_path="uploaded_notebook.ipynb",
            happy_cases_n=happy_cases_n,
            edge_cases_n=edge_cases_n,
            recursion_limit=recursion_limit,
            n_workers=n_workers,
            n_workers_inner_thread=n_workers_inner_thread
        )

        # Display the evaluation results
        st.write("------------------------EVALUATION------------------------")
        st.json(eval)

        # Save the evaluation results to a JSON file
        with open('final_evaluation.json', 'w') as f:
            json.dump(eval, f, indent=4)
        st.write("Evaluation results saved to final_evaluation.json")