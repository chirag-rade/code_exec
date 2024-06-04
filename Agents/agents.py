from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from Models.models import LLMModel
from .schemas import *
from .tools import *
from langgraph.prebuilt.tool_executor import ToolExecutor
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
load_dotenv()
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#Base Prompt Template   
def get_base_prompt_template(name=None, custom_instructions=None, agents=None):
    prompt_template = ChatPromptTemplate.from_messages(
    [
    (
    "system",
    "Your name is {name} and you are working along side other agents as listed below "
    "Agents: \n {agents} "
    "{custom_instructions} "
    ""
    )
    ]
    )
    
    prompt_template = prompt_template.partial(name=name)
    prompt_template = prompt_template.partial(agents=agents)
    prompt_template = prompt_template.partial(custom_instructions=custom_instructions if custom_instructions else "")
    return prompt_template



#CODE EXTRACTORS
code_extractor_prompt_template = get_base_prompt_template(
    name="code_extractor",
    custom_instructions="Your role is to extract the code provided by the AI assistant from the conversation in the notebook. "
                        "Once you've extracted the code, forward it to the agent named testers without making any changes. ",
    agents="testers"
)

code_extractor = LLMModel(
    provider="openai_api",
    model="gpt-4o",
    use_history=False,
    #use_tool=False,
    output_schema=StandardResponse.model_json_schema(),
    prompt_template=code_extractor_prompt_template
)


# Happy Pather Agent
happy_pather_prompt_template = get_base_prompt_template(
    name="happy_pather aka testers",
    custom_instructions="Your role is to generate and list (Just 5) happy path  test cases that should be tested (not test code). "
                        "Once you've created the test cases, share them with another agent called codia. ",
    agents="codia"
)

happy_pather = LLMModel(
    provider="openai_api",
    model="gpt-4o",
    use_history=False,
    #use_tool=False,
    output_schema=HappyPaths.model_json_schema(),
    prompt_template=happy_pather_prompt_template
)

# Edge Caser Agent
edge_caser_prompt_template = get_base_prompt_template(
    name="edge_caser aka testers",
    custom_instructions="Your role is to generate and list edge case test cases (Just 5) that should be tested (not test code) for the provided code. "
                        "Once you've created the test cases, share them with another agent called codia. ",
                      
    agents="codia"
)

edge_caser = LLMModel(
    provider="openai_api",
    model="gpt-4o",
    use_history=False,
    #use_tool=False,
    output_schema=EdgeCases.model_json_schema(),
    prompt_template=edge_caser_prompt_template
)



# Issue Finder Agent
issue_finder_prompt_template = get_base_prompt_template(
    name="issue_finder",
    custom_instructions="Your role is to identify and list issues with the provided code. "
                        "Once you've identified the issues, share them with another agent called issue_verify. ",
    agents="issue_verify"
)

issue_finder = LLMModel(
    provider="openai_api",
    model="gpt-4o",
    use_history=False,
    #use_tool=False,
    output_schema=Issues.model_json_schema(),
    prompt_template=issue_finder_prompt_template
)



code_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "As a seasoned Python and JavaScript developer, you are known as code_runner. Your role involves the following steps:"
            "1. You will be given a piece of code along with a test scenario, which could be a happy path or an edge case."
            "2. Your task is to write the test code and combine it with the original code to form a complete script."
            "3. If the test requires dummy files such as JSON or CSV use the write_file tool to create them and you can use create_directory tool to create a directory"
            "4. Execute the complete code and test using the python_repl tool. This approach eliminates the need for any unit test framework."
            "5. If you encounter any issues after invoking any tool, feel free to make necessary corrections and retry."
            "NOTE: Do whatever you need to do and only give your final comment when done"
            "Remember, always write a complete code that can be executed as a standalone script. Do not modify the original code being tested."
            "NOTE: make use of  print within your code to output information for better observation and debugging."
            "NOTE: Avoid using 'if __name__ == '__main__' as this will prevent the code from running."
            "Finally, report if the test passed and any other comment you have using this schema:\n {schema} and nothing else"
            
            

            
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

parser = PydanticOutputParser(pydantic_object=TestResults)
standard_test = parser.get_format_instructions()
code_prompt = code_prompt.partial(schema = standard_test )
llm3 = ChatOpenAI(model="gpt-4o", 
                  # model_kwargs = {"response_format":{"type": "json_object"}}
                 )
functions = [format_tool_to_openai_function(t) for t in all_tools]
model3 = code_prompt | llm3.bind_functions(functions)

def code_runner(state):
    out = model3.invoke(state)
    if "function_call" in out.additional_kwargs:
        message = [AIMessage(content=out.content, additional_kwargs=out.additional_kwargs)]
    else:
        message = [AIMessage(content=out.content)]
    #print("TYPE:", out)
    
    return {
        "messages":message,
        "sender": "code_runner",
    }


tool_executor = ToolExecutor(all_tools)