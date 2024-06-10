from flask import Flask, request, jsonify
import contextlib, io, subprocess, sys

def execute_code(code, installs =[], language="python",):
    """Use this to execute python code when needed. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user and you.
    """
    
    # for install in installs:
    #     try:
    #         subprocess.check_call([sys.executable, "-m", "pip", "install", install])
    #     except subprocess.CalledProcessError as e:
    #         return f"Failed to install package. Error: {repr(e)}"
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

app = Flask(__name__)

@app.route('/run_code', methods=['POST'])
def run_code():
    data = request.get_json()
    code = data.get('code')
    language = data.get('language')
   
    
    print(f"Language: {language}")
    
    result = execute_code(code, language)
    
    result = {'result': result,}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
