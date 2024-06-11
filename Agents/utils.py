import nbformat, json
import requests

def load_notebook(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    return nb


def router(state):
    # This is the router
    messages = state["messages"]
    sender = state["sender"]
    last_message = messages[-1]
    
    if "function_call" in last_message.additional_kwargs:
        return "tool_node" #irrespective of the sender
    else: return "END"
    
    
    
def create_json_for_code_api(code):
    code = "def test_code():\n    " + code.replace("\n", "\n    ")
    code += "\n\n# R E A D M E\n# DO NOT CHANGE the code below, we use it to grade your submission. If changed your submission will be failed automatically.\nif __name__ == '__main__':  \n    test_code()"
    return json.dumps({"code": code, "input": "None"})



def make_code_exec_request(code, language):
    url = "https://code-exec-server-staging.turing.com/api/languages/{language}".format(language=language)
    json_body = create_json_for_code_api(code)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json_body, headers=headers)
    return response.json()
