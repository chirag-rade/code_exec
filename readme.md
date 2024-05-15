# How to Use Models
**PLEASE SEE `readme.ipynb` FOR DETAILED USE CASES**


## Overview
The LLMModel with the package is a Python class designed to interact with various language model APIs such as OpenAI, Anthropic, and Fireworks. It handles message processing, token counting, and response generation based on the provided configuration and schemas. The class is versatile, supporting different providers and capable of parsing responses according to specified schemas.

## Initilization of the Model


- **provider** : The backend service provider, such as `openai_api`, `anthropic_api` or `fireworks_api`. Default is `None`.
- **model** : The specific model identifier used by the provider. Default is `None`.
- **config** (Optional): A dictionary to configure retries and other provider-specific options. Default is `{}`.
- **tools** (Optional): A list of tools or functions to be applied in the processing chain. Default is `[]` i.e no tools.
- **output_schema** (Optional): The Pydantic schema used to validate and parse the model's output. Default is `FeedbackBasic`.
- **input_schema** (Optional): The expected pydantic input schema. This is also used to validate the input before sending it to the model. Default is `None`. If none is given, then output is returned as string without parsing.
- **name** (Optional): A name identifier for the model instance. Default is `None`.
- **use_tool** (Optional):This will use tool/function method to generate the output based on the schema. Default is True.
- **prompt_template** (Optional): A langchain ChatPromptTemaplete to modify system message and add extra inputs. Default is `None`.
- **try_to_parse** (Optional): Boolean flag to attempt parsing the model's output using the output_schema. Default is `True`.
- **chat_history** (Optional): This can be used to load chat history from a database to continue a conversation `[]`.
- **as_evaluator** (Optional): Boolean flag to indicate if the model is used for evaluation purposes. Default is `False`.
- **use_history** (Optional): Boolean flag to determine if the conversation history should be used in requests. Default is `True`.


## Model Methods


- **get_chat_history()**: Retrieves the full chat history, including untouched messages.
- **get_total_tokens()**: Calculates the total number of tokens used in the conversation history.
- **self(messages)**: Processes messages through the model, handling retries and parsing as configured.


**PLEASE SEE README.IPYNB FOR DETAILED USE CASES**
