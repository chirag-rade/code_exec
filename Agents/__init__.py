from dotenv import load_dotenv
from rich import print as md
from rich.markdown import Markdown
from rich.text import Text

load_dotenv()
import json
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from Models.models import LLMModel
import concurrent.futures
import random
from .agents import *
from .prompts import *
from .schemas import *
from .tools import *
from .utils import *
from rich.console import Console
console = Console()

def display(content, color="red", id=None):
    console.print(f"###{id}### " + content, style=color)
  

def tool_node(state):
    """This runs tools in the graph

    It takes in an agent action and calls that tool and returns the result."""
    
    messages = state["messages"]
    last_message = messages[-1]
    tool_input = last_message.additional_kwargs["function_call"]["arguments"]
    tool_input = json.loads(tool_input)
    #print("DEBUG", tool_input), type(tool_input)
    # We can pass single-arg inputs by value
    # if len(tool_input) == 1:
    #     tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    tool_name = tool_name_schema_map[tool_name]
    
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_schema_name_map[tool_name]} response: {str(response)}", name=tool_schema_name_map[action.tool]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}



def code_extractor_node(state):
    x = HumanMessage(json.dumps(code_extractor(state)))
    return {
        "messages": [x],
        "sender": "code_extractor",
    }


def happy_pather_node(state):
    x = HumanMessage(json.dumps(happy_pather(state)))
    return {
        "messages": [x],
        "sender": "happy_pather",
    }


def edge_caser_node(state):
    x = HumanMessage(json.dumps(edge_caser(state)))
    return {
        "messages": [x],
        "sender": "edge_caser",
    }


def issue_finder_node(state):
    x = HumanMessage(json.dumps(issue_finder(state)))
    return {
        "messages": [x],
        "sender": "issue_finder",
    }


def combine_tests_node(state, not_graph=False):
    import copy

    testers_input = copy.deepcopy(state["messages"])
    happy_output = happy_pather(testers_input)
    edge_output = edge_caser(testers_input)
    issues = issue_finder(testers_input)
    combined_output = {
        "happy": happy_output["happy_paths"],
        "edge": edge_output["edge_cases"],
        "issues": issues["issues"],
        "code": happy_output["code"],
    }
    if not_graph:
        return combined_output
    return {
        "messages": [HumanMessage(json.dumps(combined_output))],
        "sender": "testers",
    }


def testers_graph(notebook, custom_input = None):
    workflow1 = StateGraph(AgentState)
    workflow1.add_node("code_extractor", code_extractor_node)
    workflow1.add_node("testers", combine_tests_node)
    workflow1.add_edge("code_extractor", "testers")
    workflow1.add_edge("testers", END)
    workflow1.set_entry_point("code_extractor")
    graph = workflow1.compile()

    if custom_input:
        messages_in = custom_input
    else:
        messages_in = [
            HumanMessage(content="Here is the conversation {}".format(notebook))
        ]
   
    input_message = {
        "chat_history": [],
        "messages": messages_in,
        "user_config": {},
    }

    for s in graph.stream(input_message, {"recursion_limit": 20}):
        agent = list(s.keys())[0]
        content = s[agent]["messages"][-1].content
        if agent == "testers":
            h = json.loads(content)
            tests_content = (
                f"Happy: {h['happy']}\n\nEdge: {h['edge']}\n\nIssues: {h['issues']}"
            )
            tests_content = Markdown(tests_content)
            md(tests_content)
        elif agent != "tool_node":
            # check if it is trying to call a function/tool
            if "function_call" in s[agent]["messages"][-1].additional_kwargs:
                function_being_called = s[agent]["messages"][-1].additional_kwargs[
                    "function_call"
                ]["name"]
                args = s[agent]["messages"][-1].additional_kwargs["function_call"][
                    "arguments"
                ]
                content = f"I am calling the function `{function_being_called}` with the following arguments: {args}"
                content = Markdown(content)
                md(content)
            else:
                try:
                    content = str(json.loads(content)["response"])
                except:
                    pass
                content = Markdown(content)
                md(content)
        else:
            content = Markdown(content)
            md(content)

    return h


def code_runner_graph(test, code, recursion_limit, mdcolor=None, id=None):

    workflow2 = StateGraph(AgentState)
    workflow2.add_node("code_runner", code_runner)
    workflow2.add_node("tool_node", tool_node)
    workflow2.add_conditional_edges(
        "code_runner",
        router,
        {"END": END, "tool_node": "tool_node"},
    )
    workflow2.add_conditional_edges(
        "tool_node",
        lambda x: x["sender"],
        {"code_runner": "code_runner"},
    )

    workflow2.set_entry_point(
        "code_runner",
    )
    graph2 = workflow2.compile()

    messages_in = [
        HumanMessage(
            content="HERE IS THE CODE:\n\n ```{code}``` \n\n  INSTRUCTIONS:\n\n Write the test code to test for this: \n {test}".format(
                code=code, test=test
            )
        )
    ]

    input_message = {
        "chat_history": [],
        "messages": messages_in,
        "user_config": {},
    }

    for s in graph2.stream(input_message, {"recursion_limit": recursion_limit}):
        #print("AGENT:", s)
        agent = list(s.keys())[0]
        content = s[agent]["messages"][-1].content
        if agent != "tool_node":
            # check if it is trying to call a function/tool
            if "function_call" in s[agent]["messages"][-1].additional_kwargs:
                function_being_called = s[agent]["messages"][-1].additional_kwargs[
                    "function_call"
                ]["name"]
                args = s[agent]["messages"][-1].additional_kwargs["function_call"][
                    "arguments"
                ]
                #content = f"I am calling the function `{function_being_called}` with the following arguments: {args}"
                content = f"I am calling the function `{function_being_called}`"
                # content = Markdown(content)
                # md(Markdown(content, style=mdcolor))
                display(content, mdcolor, id)
            else:
                try:
                    content = str(json.loads(content)["response"])
                except:
                    pass
                # content = Markdown(content)
                # md(Markdown(content, style=mdcolor))
                display(content, mdcolor, id)
        else:
            
            # content = Markdown(content)
            # md(Markdown(content, style=mdcolor))
            display(content, mdcolor, id)

    return s[agent]["messages"][-1].content



def execute_code_in_parallel(focus, code, recursion_limit,id, n_workers_inner_thread):
        with ThreadPoolExecutor(max_workers=n_workers_inner_thread) as executor:
            colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white']
            used_colors = set()
            futures = []
            unique_id = 0
            
            for t in focus:
                if len(used_colors) == len(colors):
                    used_colors.clear()
                available_colors = list(set(colors) - used_colors)
                mdcolor = random.choice(available_colors)
                used_colors.add(mdcolor)
                
                futures.append(executor.submit(code_runner_graph, t, code, recursion_limit=recursion_limit, mdcolor=mdcolor, id=f"{id}->TEST{unique_id}")) #actual execution
                unique_id += 1
            
            final_results = []
            for future in concurrent.futures.as_completed(futures):
                final_results.append(future.result())
                
        return final_results

def run(
    notebook_path,
    happy_cases_n=2,
    edge_cases_n=2,
    recursion_limit=30,
    loaded_notebook=None,
    n_workers = None
):
    if happy_cases_n > 10 or edge_cases_n > 10:
        raise ValueError("Number of happy cases and edge cases each should not exceed 10.")
    if loaded_notebook:
        notebook = loaded_notebook
    else:
        notebook = load_notebook(notebook_path)
    test_cases = testers_graph(notebook)

    happy_paths = test_cases["happy"]
    edge_cases = test_cases["edge"]
    focus = edge_cases[:edge_cases_n] + happy_paths[:happy_cases_n]
    code = test_cases["code"]
    final_results = []

    
    execute_code_in_parallel(focus, code, recursion_limit, id="")

    single_message = HumanMessage(
    f"Here is the actual Conversation: \n {notebook} \n"
    f"Here is the main code extracted: \n {code} \n"
    f"I decided to run the following Tests on the code: \n {happy_paths} \n"
    f"Here are the results: \n {final_results} \n"
    f"These are also some issues i see, i might be wrong though:\n {test_cases['issues']} \n"
    "Now  please give a final evaluation  of  the entire conversation using the provided schema"
    "Your Evaluation is for both the code provided and other aspect of the conversation"
)   


    ans = evaluator([single_message])

    return ans


def run_multiple_turns(
    notebook_path,
    happy_cases_n=2,
    edge_cases_n=2,
    recursion_limit=30,
    loaded_notebook=None,
    n_workers = 4,
    n_workers_inner_thread = 4
):
    if happy_cases_n > 10 or edge_cases_n > 10:
        raise ValueError("Number of happy cases and edge cases each should not exceed 10.")
    if loaded_notebook:
        notebook = loaded_notebook
    else:
        notebook = load_notebook(notebook_path)
    
    turns = split_notebook_turns(notebook_path) #adjust to work for loaded notebook also
    final_output = []
    from concurrent.futures import ThreadPoolExecutor

    def process_turn(result, turn,  id, n_workers_inner_thread):
        # print(f"#-{id}-#"*30)
        # print("DEBUG:", result)
        
        code = result["code"]
        dependencies = result["dependencies"]
        language = result["language"]
        can_be_tested  = result["can_be_tested"]

        messages_in_testers = {"messages": [
            HumanMessage(content="Here is the code to be tested: {}. It is written in {} and has the following dependencies: {}.".format(code, language, dependencies))
        ]}

        tests = combine_tests_node(messages_in_testers, not_graph=True)
        happy_paths = tests["happy"]
        edge_cases = tests["edge"]
        focus = edge_cases[:edge_cases_n] + happy_paths[:happy_cases_n]

        # now parallelly run test with codia
        results = execute_code_in_parallel(focus, code, recursion_limit,  id, n_workers_inner_thread)

        # generate final eval for turn
        single_message = HumanMessage(
            f"Here is the actual Conversation: \n {notebook} \n"
            f"Here is the main code extracted: \n {code} \n"
            f"I decided to run the following Tests on the code: \n {happy_paths} \n"
            f"Here are the results: \n {results} \n"
            f"These are also some issues i see, i might be wrong though:\n {tests['issues']} \n"
            "Now  please give a final evaluation  of  the entire conversation using the provided schema"
            "Your Evaluation is for both the code provided and other aspect of the conversation"
        )

        ans = evaluator([single_message])

        return {
            "turn": turn,
            "code": code,
            "dependencies": dependencies,
            "can_be_tested" :can_be_tested,
            "language": language,
            "tests": focus,
            "eval": ans
        }

    
    def process_turn_wrapper(turn, i, n_workers_inner_thread):
            colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white']
            mdcolor = colors[i % len(colors)]
            result = turn_classifier()([HumanMessage(json.dumps(turn))])
            should_be_tested = all([result["has_code"], result["complete_code"], result["can_be_tested"]])
            
            if should_be_tested:
                return process_turn(result, turn, f"TURN:{i}", n_workers_inner_thread)
            else:
                display(" SKIPPING TURN, NO CODE OR CAN'T BE TESTED", mdcolor, f"TURN:{i}")
                return None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        final_output = list(executor.map(lambda i_turn: process_turn_wrapper(i_turn[1], i_turn[0], n_workers_inner_thread), enumerate(turns)))

        final_output = [outcome for outcome in final_output if outcome is not None]
        return final_output
                
    