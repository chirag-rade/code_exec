from dotenv import  load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain_core.output_parsers.json import parse_partial_json
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
from langchain_core.output_parsers.json import parse_json_markdown

load_dotenv()

openai_encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(obj: Any) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(openai_encoding.encode(str(obj)))
    return num_tokens


def process_messages(messages, provider, model):
    if  provider == "openai_api" or model+provider == "llama-v3-70b-instruct"+"fireworks_api":
        print("DEBUG:", "...API token count")
        current_input = ""
        inputs = []
        outputs = []
        input_token = 0
        output_token = 0
        
        for message in messages:
            if isinstance(message, AIMessage):
                input_token += message.response_metadata['token_usage']['prompt_tokens'] 
                output_token += message.response_metadata['token_usage']['completion_tokens']
                
                inputs.append(current_input)  # Save the current state of input_accumulator before adding AI content
                outputs.append(message.content)
                # Reset current_input to include all previous messages for the next AI message
                current_input += message.content
            else:
                current_input += message.content

        all_messages_concat = {"in": inputs, "out": outputs}
        all_in = all_messages_concat['in']
        all_out = all_messages_concat['out']
        return {"in": input_token,"out":output_token}, all_messages_concat
    
    elif provider in ["anthropic_api"]:
        print("DEBUG:", "...API token count")
        current_input = ""
        inputs = []
        outputs = []
        input_token = 0
        output_token = 0
        
        for message in messages:
            if isinstance(message, AIMessage):
                input_token += message.response_metadata['usage']['input_tokens']
                output_token += message.response_metadata['usage']['output_tokens']
                
                inputs.append(current_input)  # Save the current state of input_accumulator before adding AI content
                outputs.append(message.content)
                # Reset current_input to include all previous messages for the next AI message
                current_input += message.content
            else:
                current_input += message.content

        all_messages_concat = {"in": inputs, "out": outputs}
        all_in = all_messages_concat['in']
        all_out = all_messages_concat['out']
        return {"in": input_token,"out":output_token}, all_messages_concat
        
    
    else:
        print("DEBUG:", "...Manual token count")
        current_input = ""
        inputs = []
        outputs = []
        
        for message in messages:
            if isinstance(message, AIMessage):
                inputs.append(current_input)  # Save the current state of input_accumulator before adding AI content
                outputs.append(message.content)
                # Reset current_input to include all previous messages for the next AI message
                current_input += message.content
            else:
                current_input += message.content

        all_messages_concat = {"in": inputs, "out": outputs}
        all_in = all_messages_concat['in']
        all_out = all_messages_concat['out']
        return {"in":num_tokens_from_string("".join(all_in)) ,"out": num_tokens_from_string("".join(all_out))}, all_messages_concat



def CustomJSONParser(ai_message) -> dict:
    if isinstance(ai_message, str):
        s = ai_message
    else:
        s = ai_message.content
    r = parse_json_markdown(s)
    if r is None:
        # Extract JSON from text wrapped in triple backticks
        try:
            start_idx = s.index("```json") + len("```json")
        except ValueError:
            try:
                start_idx = s.index("```") + len("```")
            except ValueError:
                start_idx = 0

        end_idx = s.rindex("```", start_idx) if "```" in s[start_idx:] else len(s)

        # Extract the JSON string
        json_str = s[start_idx:end_idx].strip()
        r = parse_json_markdown(json_str)
    return r

class LLMModel:
    def __init__(self, provider, model, 
                 config: dict = {},
                 tools: list = [],
                 output_schema: dict = FeedbackBasic, 
                 input_schema = None, 
                 name: str = None,
                 prompt_template:  ChatPromptTemplate = None,
                 try_to_parse:bool = True,
                 chat_history: list = [],
                 as_evaluator:bool =  False,
                 use_history: bool = True):
        
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
        self.chat_history_untouched = chat_history if chat_history else []
        self.id = str(uuid.uuid4())
        self.chain = None
        self.retry = self.config.get("retry", 3)
        self.retry_with_history = self.config.get("retry_with_history", False)
        self.use_history = use_history
        self.first_message = True
        self.initial_message  = []
    
        
    def init(self):
        self.chain = self.create_chain()
        
        
    def old_parser(self, rx):
        if self.provider in ["anthropic_api"]:
                    try:
                        response_json = json.loads(rx.content)
                        return response_json 
                    except Exception as e:
                        if self.test: raise Exception("Invalid") # for debugging purposes
                        json_start = rx.content.index('{')
                        json_end = rx.content.rindex('}')
                        json_str = rx.content[json_start:json_end+1]
                        json_obj = json.loads( json_str)
                        json_obj.update({"other_content": rx.content[:json_start] + rx.content[json_end+1:]})
                        return json_obj
                        
                            

        elif self.provider in ["fireworks_api", "openai_api"]:
                    return json.loads(rx.content) if self.output_schema != None else rx.content
            
        
    def __call__(self, messages, retrying=False):
        
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
                if not retrying:IN_   = self.prepare_messages(messages["messages"])
            else: #no messages just inputs, we add messages
                messages.update({"messages":[]})
                IN_ = messages
        elif isinstance(messages, list): #this  user pass a list of langchain messages
            IN_ = self.prepare_messages(messages)
            
        #----------
        
        if self.first_message == True:
            #print("INFO:", "First Message")
            self.initial_message = self.prompt_template.invoke(IN_)
            self.first_message = False
            
        #---------
    
        response =  self.chain.invoke(IN_)
        rx =  self.add_to_history_and_prepare_response(response)
        
        #print("DEBUG: ", rx)
        if self.try_to_parse and self.output_schema !=None: #user wants answer as json
            
            try:
                #raise Exception("test exception raised")
                result = CustomJSONParser(rx)
                self.validate_output_schema(result)
                return result
            except Exception as e:
                try:
                    #raise Exception("test exception raised")
                    result = self.old_parser(rx)
                    self.validate_output_schema(result)
                    return result
                except Exception as e:
                    self.retry -= 1
                    if self.retry > 0:
                        print("retrying...", "retry remaining ", self.retry)
                        if self.retry_with_history:
                            self.chat_history.append(HumanMessage(content="You failed to output a valid/parsable JSON. Please try again"))
                            self.chat_history_untouched.append(HumanMessage(content="You failed to output a valid/parsable JSON. Please try again"))
                        elif not self.retry_with_history and len(self.chat_history) > 0:  # remove last response from history - this will affect main chat history record
                            self.chat_history.pop(-1)  # Correctly remove the last item
                        return self(messages, retrying=True)  # Recursively call __call__ with the original messages
                    self.retry = self.config.get('retry', 3)
                    raise Exception("JSON Parsing Retry exceeded")
            
        else:
            return rx if isinstance(rx, str) else rx.content
    
    
    def validate_output_schema(self, output):
        if not isinstance(self.output_schema, dict) and self.try_to_parse:
            try:
                print("Validating output schema.....")
                self.output_schema.model_validate(output)
            except:
                raise Exception("Validation of output schema failed")
        
    
    def get_chat_history(self):
        return  self.initial_message.messages + self.chat_history_untouched[1:]
        
         
    def create_chain(self):
        functions = [format_tool_to_openai_function(t) for t in self.tools]
        llm = self.create_model()
        
        if self.output_schema != None : #aslong as there is an output schema then give the instructions
            if isinstance(self.output_schema, dict):
                format_instructions = self.output_schema
            else:
                pydantic_parser = PydanticOutputParser(pydantic_object=self.output_schema)
                format_instructions = pydantic_parser.get_format_instructions()
            
            self.prompt_template = self.prompt_template.partial(format_instructions="IN JSON FORMAT: " + str(format_instructions))
        else: #when there is no output schema, we still need to fill up the template
            self.prompt_template = self.prompt_template.partial(format_instructions="")
            
            

 
        if len(self.tools) is not 0:
            return self.prompt_template | llm | llm.bind_functions(functions)
        else:
            return self.prompt_template | llm
          
    def prepare_messages(self, messages: list):
        self.chat_history.extend(messages)
        self.chat_history_untouched.extend(messages)
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
        #ensure response is an AImessage
        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
        #get response
        if self.use_history == True: 
            self.chat_history.append(response)
            self.chat_history_untouched.append(response)
        else:
            self.chat_history = [] #clear the history if not to be saved
            self.chat_history_untouched = []
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
        return  process_messages(self.get_chat_history(), self.provider, self.model)
   
    
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
        
        
        

    