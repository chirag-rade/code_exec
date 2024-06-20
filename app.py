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

    # Run the evaluation
    if st.button("Run Evaluation"):
        st.write("Running evaluation, please wait...")
        try:
            eval = run_multiple_turns(
                notebook_path="uploaded_notebook.ipynb"
            )
            # Save the evaluation results to a JSON file
            with open('final_evaluation.json', 'w') as f:
                json.dump(eval, f, indent=4)
            st.write("Evaluation results saved to final_evaluation.json")
        except Exception as e:
            # st.write(f"An error occurred during evaluation: {str(e)}")
            st.write("The code cannot be executed")
        try:
            eval = eval[0]
        except (IndexError, KeyError, TypeError) as e:
            st.write("Code cannot run, encountered syntax error.")
            st.stop()

        # Display the evaluation results
        st.write("------------------------EVALUATION------------------------")
        
        # Display original code
        st.subheader("Original Code")
        st.code(eval['code'])

        # Display testing code and results
        for i, test in enumerate(eval['tests']):
            st.subheader(f"Test {i + 1}")
            # st.code(test)
            st.write("Testing Code")
            st.code(eval['testcoderesult'][i]['code'])
            st.write("Testing Code Result")
            st.code(eval['testcoderesult'][i]['response']['run']['out'])
            
            # Display error if it exists
            if not eval['testcoderesult'][i]['response']['run']:
                st.write("Error")
                st.code(eval['testcoderesult'][i]['response']['run']['error'])
            
            st.write("Result")
            st.write(eval['results'][i]['result'])
            st.write("Comment")
            st.write(eval['results'][i]['comment'])

        # # Save the evaluation results to a JSON file
        # with open('final_evaluation.json', 'w') as f:
        #     json.dump(eval, f, indent=4)
        # st.write("Evaluation results saved to final_evaluation.json")