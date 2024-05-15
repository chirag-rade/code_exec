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
    
    
    try:
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
    except Exception as e:
        json_start = s.index('{')
        json_end = s.rindex('}')
        json_str = s[json_start:json_end+1]
        json_obj = json.loads(json_str)
        json_obj.update({"other_content": s[:json_start] + s[json_end+1:]})
        return json_obj

class LLMModel:
    def __init__(self, provider, model, 
                 config: dict = {},
                 output_schema: dict = FeedbackBasic, 
                 input_schema = None, 
                 name: str = None,
                 use_tool: bool = True,
                 prompt_template:  ChatPromptTemplate = None,
                 try_to_parse:bool = True,
                 chat_history: list = [],
                 as_evaluator:bool =  False,
                 use_history: bool = True):
        
        self.output_type = None
        self.try_to_parse = try_to_parse
        self.test = False
        self.use_tool = use_tool
        self.provider = provider
        self.model = model
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
         

    def is_tool_call(self,message):
        try:
            if self.provider == "openai_api":
                a = json.loads(message.additional_kwargs["tool_calls"][0]["function"]["arguments"])
                return a
            elif self.provider == "anthropic_api":
                for content in message.content:
                    if content["type"] == "tool_use":
                        a = content['input']
                return a
        except Exception as e:
            return False
    
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
            original_messages = copy.deepcopy(messages)
            #does it have messages in it?
            if "messages" in messages.keys():
                formated_message = self.prepare_messages(messages["messages"])
                messages.update(formated_message)
                IN_= messages
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
        
        # if retrying:
        #     print("retrying params:")
        #     print(messages)
        #     print(IN_)
        #     print("retrying params end:")
    
        response =  self.chain.invoke(IN_)
        rx =  self.add_to_history_and_prepare_response(response)
        # check if ita a tool call:
        tool_calling = self.is_tool_call(rx)
        if tool_calling and self.use_tool:
            #extract the args
            rx =  tool_calling
       
       
        #print("DEBUG: ", rx)
        if self.try_to_parse and self.output_schema !=None: #user wants answer as json
            
            try:
                #raise Exception("test exception raised")
                result = CustomJSONParser(json.dumps(rx)) if tool_calling else CustomJSONParser(rx)
                self.validate_output_schema(result)
                return result
            except Exception as e:
                #print("ERROR:", e)
                if self.retry > 0:
                    print("retrying...", "retry remaining ", self.retry)
                    if self.retry_with_history:
                        if self.provider in ["fireworks_api"] or not self.use_tool:
                            retry_message = HumanMessage(content="You failed to output a valid/parsable JSON. Please try again")
                        else:
                            retry_message = HumanMessage(content="You failed to call the function/tool properly. Please try again")
                        
                            
                        self.chat_history.append(retry_message)
                        self.chat_history_untouched.append(retry_message)
                    elif not self.retry_with_history and len(self.chat_history) > 0:  # remove last response from history - this will affect main chat history record
                        self.chat_history.pop(-1)  # Correctly remove the last item
                    self.retry -= 1
                    return self(original_messages, retrying=True)  # Recursively call __call__ with the original messages
                self.retry = self.config.get('retry', 3)
                raise Exception("JSON Parsing Retry exceeded")
            
        else:
            return rx if isinstance(rx, str) else rx.content
    
    
    def validate_output_schema(self, output):
        if not isinstance(self.output_schema, dict) and self.try_to_parse:
            print("Validating output schema.....")
            self.output_schema.model_validate(output)
           
    
    def get_chat_history(self):
        return  self.initial_message.messages + self.chat_history_untouched[1:]
        
         
    def create_chain(self):
        
        #GET LLM
        llm = self.create_model()
        
  
        #FORMAT INSTRUCTIONS
        if self.output_schema != None : #aslong as there is an output schema then give the instructions
            if isinstance(self.output_schema, dict):
                if not self.use_tool or self.provider == "groq_api" or self.provider == "fireworks_api": #these providers dont suppport function calling
                    format_instructions = f'''Ensure your output is given in a JSON format. Here is the description of the 
                    JSON parameters including the type and other info:\n
                    {self.output_schema["properties"]}
                    '''  
                else:
                    ##MAJOR UPDATE FOR USING TOOL
                    if self.provider == "openai_api":
                        format_instructions = f"once you are done with the task, call the given function {self.output_schema["title"]} to save the evaluation. Your final response should always be JSON"
                        llm = llm.bind_tools(
                                            [self.output_schema],
                                            tool_choice={
                                                "type": "function",
                                                "function": {"name": self.output_schema["title"]},
                                            },
                        )
                        
                    elif self.provider == "anthropic_api":
                        format_instructions = f"Please, output only JSON nothing else, use the tool {self.output_schema["title"]} "
                        # needs at least one user message
                        human_bool = [isinstance(msg_type, HumanMessagePromptTemplate) for msg_type in self.prompt_template.messages]
                        if True in human_bool:
                            pass
                        else:
                            self.prompt.messages.append(HumanMessagePromptTemplate.from_template("Follow instructions and output JSON."))
                        
                        llm = llm.bind_tools([self.output_schema])
                    
                  
            else:
                pydantic_parser = PydanticOutputParser(pydantic_object=self.output_schema)
                format_instructions = pydantic_parser.get_format_instructions()
                format_instructions =  format_instructions
            
            self.prompt_template = self.prompt_template.partial(format_instructions=format_instructions)
        
        
        else: #when there is no output schema, we still need to fill up the template
            self.prompt_template = self.prompt_template.partial(format_instructions="")
            
            
        #CHAIN CREATE
        #print("FORMAT INSTRUCTIONS:", format_instructions)
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
        if  isinstance(response, str):
            response = AIMessage(content=str(response))
        elif isinstance(response, dict) and "content" in response.keys(): #fireworks
            response = AIMessage(content=response["content"])
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
            if self.try_to_parse and self.output_schema != None and self.model in [
                "gpt-3.5-turbo-1106",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-4o"
            ]:
                mk.update({"response_format":{"type": "json_object"}}) 
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
            if self.try_to_parse and self.output_schema != None:
                mk.update({"response_format":{"type": "json_object",  "schema": self.output_schema}}) if isinstance(self.output_schema,dict) else self.output_schema.model_json_schema(),
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
    
    # ------------- FUNCTION CALLING FUNCTIONS -------------
    

    def prepare_for_tool_use(self, prompt, model):
        provider = self.llm_config["provider"]
        if provider == "openai_api":
            return (
                prompt,
                model.bind_tools(
                    [self.output_schema],
                    tool_choice={
                        "type": "function",
                        "function": {"name": self.output_schema["title"]},
                    },
                ),
                "tool",
            )
        elif provider == "anthropic_api":
            # needs at least one user message
            if len(prompt.messages) < 2:
                prompt.messages.append(
                    HumanMessagePromptTemplate.from_template(
                        "Follow instructions and output JSON."
                    )
                )

            return (
                prompt,
                model.bind_tools(
                    [self.output_schema],
                ),
                "tool",
            )
        elif provider == "groq_api" or provider == "fireworks_api":
            prompt = self._instruct_for_json_output(prompt)
            # fireworks has both chat and non chat, chat supports schema inside but so so, and non chat does not support even json
            return prompt, model, "json"
        else:
            raise NotImplementedError(
                "Model creation for the given provider is not implemented."
            )

    def evaluate_old(self, input_data, config, input_validation=True, parse=True):
        input_data = {**input_data, **config}
        if input_validation:
            self.validate_input(input_data)
        prompt = self.create_prompt(input_data)
        model = self.create_model()
        prompt, model, output_parser, output_type = self.prepare_for_tool_use(
            prompt, model
        )
        if parse:
            chain = prompt | model | output_parser
        else:
            chain = prompt | model
        chain = chain.with_retry(stop_after_attempt=self.config.get("retries", 3))
        result = None
        if parse:
            if output_type == "tool":
                for _ in range(self.config.get("retries", 4)):
                    result = chain.invoke(input_data)
                    if result:
                        break
            else:
                result = chain.invoke(input_data)
            if output_type == "tool":
                return result[0]["args"]
            elif output_type == "json":
                return result
            else:
                raise NotImplemented(f"Wrong output type in evaluate {output_type}")
        else:
            return {"RAW_OUTPUT": str(chain.invoke(input_data))}
        
        
        

    