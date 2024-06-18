import nbformat, json
import requests
from llm_as_evaluator_client.client import LLMasEvaluator
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
from Models.models import LLMModel
from enum import Enum
from typing import Annotated, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import operator

load_dotenv()

LOCAL_BASE_URL = "http://127.0.0.1:8887"
REMOTE_BASE_URL = "https://llm-as-evaluator.turing.com"
llm_as_evaluator = LLMasEvaluator(REMOTE_BASE_URL, "public-engagement", "beta-test")

def split_notebook_turns(notebook_path):
    with open(notebook_path, "r") as f:
        notebook_contents = f.read()
         
    parsed_notebooks = llm_as_evaluator.parse_notebooks([notebook_contents])
    conversations = []
    for notebook in parsed_notebooks:
        conversation = notebook.get('conversation', [])
        turn = []
        for i in range(len(conversation)):
            if i == 0 or conversation[i]['role'] != conversation[i-1]['role']:
                if turn:
                    conversations.append(turn)
                turn = [{'role': conversation[i]['role'], 'content': conversation[i]['content']}]
            else:
                turn.append({'role': conversation[i]['role'], 'content': conversation[i]['content']})
        if turn:
            conversations.append(turn)
    return conversations

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







supported_languages = [
    "python3",
    "php",
    "nodejs",
    "kotlin",
    "java",
    "go",
    "csharp",
    "cpp",
    "ruby",
    "swift"
]
