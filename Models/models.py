from dotenv import  load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    HumanMessage,
)
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
import json
import os
from abc import abstractmethod
from typing import Any
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.output_parsers import (
    ToolsOutputParser as AntrhropicToolsOutputParser,
)
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_fireworks import Fireworks
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAI
import uuid
import tiktoken
from typing import Any
from . import schemas

load_dotenv()

openai_encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(obj: Any) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(openai_encoding.encode(str(obj)))
    return num_tokens


class LLMModel:
    def __init__(self, provider, model, 
                 config: dict = {},
                 tools: list = [],
                 output_schema: dict = None, 
                 input_schema = None, 
                 name: str = None,
                 prompt_template = None,
                 chat_history: list = None,
                 as_evaluator:bool =  False,
                 use_history: bool = False):
        
        
        self.test = False
        self.provider = provider
        self.model = model
        self.tools = tools
        self.config = config
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.name = name
        self.as_evaluator = as_evaluator
        self.prompt_template = prompt_template if prompt_template is not None else ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{format_instructions}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.chat_history = chat_history if chat_history else []
        self.id = str(uuid.uuid4())
        self.chain = None
        self.retries = self.config.get("retries", 2)
        self.use_history = use_history
        
    def init(self):
        self.chain = self.create_chain()
        
    def __call__(self, messages):
        if self.as_evaluator:
            #print(type(messages))
            if not isinstance(messages, self.input_schema):
                print("ERROR: INVALID SCHEMA")
                return None
            self.create_prompt_template(messages) #use the instruction in the input to recreate the template
            messages = [HumanMessage(content=str(messages.conversation))]
        
        
        
        self.chain = self.create_chain()
        IN_ = self.prepare_messages(messages)
        response =  self.chain.invoke(IN_)
        rx =  self.add_to_history_and_prepare_response(response)
        if self.provider in ["anthropic_api"]:
            try:
                response_json = json.loads(rx.content)
                return rx
            except:
                try:
                    if self.test: raise Exception("Invalid") # for debugging purposes
                    json_start = rx.content.index('{')
                    json_end = rx.content.rindex('}')
                    json_str = rx.content[json_start:json_end+1]
                    json_obj = json.loads( json_str)
                    json_obj.update({"other_content": rx.content[:json_start] + rx.content[json_end+1:]})
                    final_rx = json.dumps(json_obj)
                    return AIMessage(content=final_rx)
                except:
                    self.chat_history.append(
                        HumanMessage(content="Its's Important you reply as json as described, Please try again")
                    )
                    rx = self(self.chat_history)
                    return rx

        else:
            return rx   
         
    def create_chain(self):
        functions = [format_tool_to_openai_function(t) for t in self.tools]
        llm = self.create_model()
        pydantic_parser = PydanticOutputParser(pydantic_object=self.output_schema)
        format_instructions = pydantic_parser.get_format_instructions()
        self.prompt = self.prompt_template.partial(format_instructions=format_instructions)
        if len(self.tools) is not 0:
            return self.prompt | llm | llm.bind_functions(functions)
        else:
            return self.prompt | llm
        
        
    
    def prepare_messages(self, messages: list):
        
      
        self.chat_history.extend(messages)
       
            
        #Convert to model specific format
        if self.provider in ["openai_api", "anthropic_api"] :
            return self.chat_history
        
        elif self.provider in ["fireworks_api"] : #uses dict messages
            new_messages =[]
            for message in self.chat_history:
                if isinstance(message, HumanMessage):
                    new_messages.append(
                        {"content":message.content, "role":"user"}
                    )
                elif isinstance(message, AIMessage):
                    new_messages.append(
                        {"content":message.content, "role":"assistant"}
                    )
                elif isinstance(message, SystemMessage):
                    new_messages.append(
                        {"content":message.content, "role":"system"}
                    )
                    
                elif isinstance(message, FunctionMessage):
                    new_messages.append(
                        {"content":message.content, "role":"function"}
                    )
            return {"messages":new_messages}
            
    
            
    
    def add_to_history_and_prepare_response(self, response):
        #get response
        if self.use_history == True: 
            self.chat_history.append(response)
        else:
            self.chat_history = [] #clear the history if not to be saved
        return response
         
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider}, model={self.model}, name={self.name}-{self.id})"
    
    def create_model(self):
        if self.provider == "openai_api":
            mk = self.config.get("model_kwargs", {})
            mk.update({"response_format":{"type": "json_object"}})
            return ChatOpenAI(model=self.model, **self.config.get("params",{}), streaming=False, model_kwargs =mk)
        elif self.provider == "groq_api":
            mk = self.config.get("model_kwargs", {})
            mk.update({"response_format":{"type": "json_object"}})
            return ChatGroq(model=self.model, **self.config.get("params",{}), streaming=False, model_kwargs =mk)
        elif self.provider == "anthropic_api":
            mk = self.config.get("model_kwargs", {})
            return ChatAnthropic(model=self.model, **self.config.get("params",{}), streaming=False, model_kwargs =mk)
        elif self.provider == "fireworks_api":
            mk = self.config.get("model_kwargs", {})
            mk.update({"response_format":{"type": "json_object"}})
            if self.model in [
                "dbrx-instruct",
                "mixtral-8x22b-instruct",
                "mixtral-8x7b-instruct",
                "llama-v3-70b-instruct",
                "llama-v3-8b-instruct",
                "llama-v2-70b-chat",
            ]:
                return ChatOpenAI(
                    api_key=os.environ["FIREWORKS_API_KEY"],
                    base_url="https://api.fireworks.ai/inference/v1",
                    model_kwargs =mk,
                    model="accounts/fireworks/models/" + self.model,
                    **self.config.get("params",{}),
                    streaming=False,
                )
            else:
                return OpenAI(
                    api_key=os.environ["FIREWORKS_API_KEY"],
                    base_url="https://api.fireworks.ai/inference/v1",
                    model="accounts/fireworks/models/" + self.model,
                    **self.config.get("params",{}),
                    streaming=False,
                )
        else:
            raise NotImplementedError("Model creation for the given provider is not implemented.")

   
    def get_total_tokens(self):
        all_messages = self.prompt.format_messages(messages=self.chat_history)
        if self.provider in ["fireworks_api", "openai_api"] and len(self.chat_history) > 0:
            return self.chat_history[-1].response_metadata['token_usage']['total_tokens'], all_messages
        
        elif self.provider in ["anthropic_api"] and len(self.chat_history) > 0:
            return  (self.chat_history[-1].response_metadata['usage']['input_tokens'] + 
                    self.chat_history[-1].response_metadata['usage']['output_tokens']), all_messages
        
        else:
            return num_tokens_from_string([message.content for message in all_messages]), all_messages
        
        
    def create_prompt_template(self, eval):
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class Evaluator, whose job is to evaluate the quality of the given aspect(s):\n{aspect}.\n\n"
                    "You are to focus on only the  responses from {role} in the conversation. i.e it is crucial that you evaluate the {role} messages/responses only.\n\n"
                    "Think step by step to figure out the correct evaluation(s) and provide you final response in this given format:\n"
                    "{format_instructions}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
            )
        self.prompt_template = self.prompt_template.partial(aspect=eval.quality_aspect)
        self.prompt_template = self.prompt_template.partial(role=eval.role.value)
   
                    
    
    def evaluate(self, evaluation_input):
        self.create_prompt_template(evaluation_input)
        task = [HumanMessage(content=str(evaluation_input.conversation))]
        result = self(task)
        return result.content
        
        
        

    