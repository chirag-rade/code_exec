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
import copy
from Models.schemas import AspectorEvaluatorInputSchema, FeedbackISC, FeedbackBasic,  QualityAspect, AspectorRole
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder

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
                 output_schema: dict = FeedbackBasic, 
                 input_schema = None, 
                 name: str = None,
                 prompt_template:  ChatPromptTemplate = None,
                 try_to_parse:bool = True,
                 chat_history: list = None,
                 as_evaluator:bool =  False,
                 use_history: bool = False):
        
        self.try_to_parse = try_to_parse
        self.test = False
        self.provider = provider
        self.model = model
        self.tools = tools
        self.config = config
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.name = name
        self.as_evaluator = as_evaluator
        self.prompt_template =  self.correct_prompt_template(prompt_template)
        self.chat_history = chat_history if chat_history else []
        self.id = str(uuid.uuid4())
        self.chain = None
        self.retry = self.config.get("retry", 3)
        self.use_history = use_history
        self.first_message = True
        
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
        
      
        
        
        #here i check if you pass a dict
        if isinstance(messages, dict):
            #does it have messages in it?
            if "messages" in messages.keys():
                IN_   = self.prepare_messages(messages["messages"])
            else: #no messages just inputs, we add messages
                messages.update({"messages":[]})
                IN_ = messages
        elif isinstance(messages, list): #this  user pass a list of langchain messages
            IN_ = self.prepare_messages(messages)
            
        #----------
        
        if self.first_message == True:
            print("INFO:", "First Message")
            self.initial_message = self.prompt_template.invoke(IN_)
            self.first_message = False
    
        #---------
    
        response =  self.chain.invoke(IN_)
        rx =  self.add_to_history_and_prepare_response(response)
        if self.try_to_parse and self.output_schema !=None: #user wants answer as json
            if self.provider in ["anthropic_api"]:
                try:
                    response_json = json.loads(rx.content)
                    return response_json 
                except Exception as e:
                    try:
                        if self.test: raise Exception("Invalid") # for debugging purposes
                        json_start = rx.content.index('{')
                        json_end = rx.content.rindex('}')
                        json_str = rx.content[json_start:json_end+1]
                        json_obj = json.loads( json_str)
                        json_obj.update({"other_content": rx.content[:json_start] + rx.content[json_end+1:]})
                        return json_obj
                    except Exception as e: 
                        #try for a specific number of time
                        self.chat_history.append(
                            HumanMessage(content="Its's Important you reply as json as described, Please try again")
                        )
                        return self(self.chat_history)
                        

            elif self.provider in ["fireworks_api", "openai_api"]:
                return json.loads(rx.content) if self.output_schema != None else rx.content
        else:
            # if self.output_schema == FeedbackBasic:
            #     print("wow")
            #     return json.loads(rx.content)["response"]
            # else:
                return rx.content
            
        
         
    def create_chain(self):
        functions = [format_tool_to_openai_function(t) for t in self.tools]
        llm = self.create_model()
        
        if self.output_schema != None : #aslong as there is an output schema then give the instructions
            pydantic_parser = PydanticOutputParser(pydantic_object=self.output_schema)
            format_instructions = pydantic_parser.get_format_instructions()
            self.prompt_template = self.prompt_template.partial(format_instructions=format_instructions)
        else: #when there is no output schema, we still need to fill up the template
            self.prompt_template = self.prompt_template.partial(format_instructions="")
            
            

 
        if len(self.tools) is not 0:
            return self.prompt_template | llm | llm.bind_functions(functions)
        else:
            return self.prompt_template | llm
          
    def prepare_messages(self, messages: list):
        self.chat_history.extend(messages)
        #Convert to model specific format
        if self.provider in ["openai_api", "anthropic_api"] :
            return {"messages":self.chat_history}
        
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
            if self.try_to_parse and self.output_schema != None: mk.update({"response_format":{"type": "json_object"}}) 
            return ChatOpenAI(model=self.model, **self.config.get("params",{}), streaming=False, model_kwargs =mk)
        elif self.provider == "groq_api":
            mk = self.config.get("model_kwargs", {})
            if self.try_to_parse and self.output_schema != None: mk.update({"response_format":{"type": "json_object"}}) 
            return ChatGroq(model=self.model, **self.config.get("params",{}), streaming=False, model_kwargs =mk)
        elif self.provider == "anthropic_api":
            mk = self.config.get("model_kwargs", {})
            return ChatAnthropic(model=self.model, **self.config.get("params",{}), streaming=False, model_kwargs =mk)
        elif self.provider == "fireworks_api":
            mk = self.config.get("model_kwargs", {})
            if self.try_to_parse and self.output_schema != None: mk.update({"response_format":{"type": "json_object"}}) 
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
        all_messages = self.initial_message.messages + self.chat_history
        if self.provider in ["fireworks_api", "openai_api"] and len(self.chat_history) > 0:
            token = {
                "in":self.chat_history[-1].response_metadata['token_usage']['prompt_tokens'],
                "out": self.chat_history[-1].response_metadata['token_usage']["completion_tokens"]
                }
            return token
        
        elif self.provider in ["anthropic_api"] and len(self.chat_history) > 0:
            return  {
                "in": self.chat_history[-1].response_metadata['usage']['input_tokens'],
                "out":  self.chat_history[-1].response_metadata['usage']['output_tokens']
                }
        
        else:
            all_ai = ''
            all_human_and_sys = ''
            return num_tokens_from_string([message.content for message in all_messages])
    
    
    def correct_prompt_template(self, old_prompt_template):
        new_prompt_template = copy.deepcopy(old_prompt_template)
        #if no prompt_template is given
        if not new_prompt_template:
            return ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "{format_instructions}" #if self.try_to_parse and self.output_schema != None else ""
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
        
        else:
            #if given but no format_instruction in system or not system at all
            if "format_instructions" not in new_prompt_template.input_variables: # and self.try_to_parse
                #find the last system message and insert format_instructions after  (better to come last)
                sys_bool = [isinstance(msg_type, SystemMessagePromptTemplate) for msg_type in new_prompt_template.messages]
                if True in sys_bool:
                    #find the last True:
                    last_occurrence_index = len(sys_bool) - 1 - sys_bool[::-1].index(True)
                    # now insert the format_instructions into the last system message
                    # Copy the old one:
                    old_system_template = new_prompt_template.messages[0].dict()['prompt']["template"]
                    new_system_template = "\n" + old_system_template + "\n{format_instructions}"
                    # re-create and replace SystemMessagePromptTemplate
                    new_prompt_template.messages[last_occurrence_index ] = SystemMessagePromptTemplate.from_template(new_system_template)
                else:
                    # when no system template, create one and insert on top
                    system_message_template = SystemMessagePromptTemplate.from_template("{format_instructions}")
                    new_prompt_template.messages.insert(0, system_message_template)
            
                
        #add messages placeholder if nt already exists
        msg_pl_bool =[isinstance(msg_type, MessagesPlaceholder) for msg_type in new_prompt_template]
        if  msg_pl_bool:
            #verify that the variable name is messages
            for msg in new_prompt_template.messages:
                if isinstance(msg, MessagesPlaceholder):
                    if msg.variable_name == "messages":
                        return new_prompt_template 
            new_prompt_template.append(MessagesPlaceholder(variable_name="messages"))
        else:
            new_prompt_template.append(MessagesPlaceholder(variable_name="messages"))
        
        return new_prompt_template
        
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
        
        
        

    