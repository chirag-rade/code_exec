from langchain_core.tools import tool
import os
import io
import contextlib
from typing import Annotated

@tool
def python_repl(code: Annotated[str, "The python code to execute."]):
    """Use this to execute python code when needed. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user and you.
    """
    # Create StringIO objects to capture stdout and stderr
    stdout = io.StringIO()
    stderr = io.StringIO()

    # Use context managers to redirect stdout and stderr to our StringIO objects
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            # Use exec to execute the code
            exec(code, locals())
            result = stdout.getvalue()
            error = stderr.getvalue()
        except Exception as e:
            # If an error occurs during execution, return the error message
            return f"Failed to execute. Error: {repr(e)}"

    # If no errors occurred, return the output
    if error:
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}\nStderr: {error}"
    else:
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"



@tool
def write_file(name: Annotated[str, "The name of the file."], content: Annotated[str, "The content of the file."]):
    """This tool writes a file with the given name and content to the Playground directory."""
    # Define the directory
    directory = "Playground"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the file path
    file_path = os.path.join(directory, name)
    
    # Write the content to the file
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        status = "File created successfully"
    except Exception as e:
        status = f"Failed to create file. Error: {repr(e)}"
    
    # Return the name and status of creation
    return {"full_path": file_path, "status": status}

@tool
def create_directory(name: Annotated[str, "The name of the directory."]):
    """This tool creates a directory with the given name in the Playground directory."""
    # Define the parent directory
    parent_directory = "Playground"
    
    # Create the parent directory if it doesn't exist
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    
    # Define the directory path
    directory_path = os.path.join(parent_directory, name)
    
    # Create the directory
    try:
        os.makedirs(directory_path)
        status = "Directory created successfully"
    except Exception as e:
        status = f"Failed to create directory. Error: {repr(e)}"
    
    # Return the name and status of creation
    return {"full_path": directory_path, "status": status}


@tool
def run_js_code(js_code: Annotated[str, "The JavaScript code to run."]):
    
    """This tool runs the provided JavaScript code and returns the stdout."""
    try:
        result = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
        return {"result": result.stdout, "status": "JavaScript code executed successfully"}
    except subprocess.CalledProcessError as e:
        return {"result": None, "status": f"Failed to execute JavaScript code. Error: {repr(e)}"}

all_tools = [
            python_repl,
            write_file,
            run_js_code,
            create_directory
        ]
