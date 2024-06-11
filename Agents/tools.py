
from langchain.pydantic_v1 import BaseModel, Field
import os
import subprocess
from typing import Annotated
from langchain_core.tools import tool
import contextlib, io, subprocess
from enum import Enum


# @tool
# def python_repl(code: Annotated[str, "The python code to execute."], 
#                 #installs: Annotated[list[str], "The list of packages to install if any is external package is needed"],
#               ):
    
#     """Use this to execute python code when needed. If you want to see the output of a value,
#     you should print it out with `print(...)`. This is visible to the user and you.
#     """


#     url = "http://localhost:70/run_code"

#     headers = {
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "code": code,
#         "language": "python",
#         #"pip_install": installs
#     }

#     # Make the POST request
#     response = requests.post(url, headers=headers, data=json.dumps(payload))

#     # If the request was successful, return the response
#     if response.status_code == 200:
#         return response.json()['result']
#     else:
#         return f"Failed to execute. Error: {response.text}"
@tool
def code_exec(code: Annotated[str, "The code to execute."], language: Annotated[str, "The language of the code (python or javascript)"]):
    """Use this to execute code when needed. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user and you.
    """
    if language == "python":
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
    
    
    elif language == "javascript":
        try:
            result = subprocess.run(
                ["node", "-e", code], capture_output=True, text=True, check=True
            )
            return {
                "result": result.stdout,
                "status": "JavaScript code executed successfully",
            }
        except subprocess.CalledProcessError as e:
            return {
                "result": None,
                "status": f"Failed to execute JavaScript code. Error: {repr(e)}",
            }
    else:
        return f"Unsupported language: {language}"

@tool
def write_file(
    file_name: Annotated[str, "The name of the file."],
    content: Annotated[str, "The content of the file."],
):
    """This tool writes a file with the given name and content to the Playground directory."""
    # Define the directory
    directory = "Playground"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path
    file_path = os.path.join(directory, file_name)

    # Write the content to the file
    try:
        with open(file_path, "w") as file:
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


class TestResult(Enum):
    PASSED = "Passed"
    FAILED = "Failed"

class TestResults(BaseModel):
    """Represents the outcome of a test."""
    result: str = Field(
        ..., description="The result of the test, either PASSED or FAILED."
    )
    comment: str = Field(
        ..., description="Any comments or notes about the test result."
    )

class CodeExec(BaseModel):
    #__name__ = "code_exec"
    """Run python and javascript Code"""

    code: str = Field(description="The code to be executed")
    language: str =  Field(description="The language of the code(python or javascript).")



class WriteFile(BaseModel):
    #__name__ = "write_file"
    """Write File"""

    file_name: str = Field(description="The name of the file.")
    content: str = Field(description="The content of the file.")

class CreateDir(BaseModel):
    # __name__ = "create_directory"
    """Create Directory"""

    name: str = Field(description="The name of the directory.")





    
all_schema= [CodeExec, TestResults, WriteFile, CreateDir]
all_tools = [code_exec, write_file, create_directory]


tool_name_schema_map = {
    "CodeExec":"code_exec",
    "WriteFile":"write_file",
    "CreateDir":"create_directory",
}

tool_schema_name_map = {
    "code_exec": "CodeExec",
    "write_file": "WriteFile",
    "create_directory": "CreateDir",
}

