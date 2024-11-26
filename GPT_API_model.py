from openai import OpenAI
import json
import requests
import base64
import os
from dotenv import load_dotenv
from database_retrieval import RetrievalModel
from function_calling import handle_function_calling
class GPT_APIModel:
    def __init__(self, api_key, model_name= None):
        self.client = OpenAI(api_key=api_key)
        self.tools = [
                        {
                            "type": "function",
                            "function": {
                                        "name": "finding_repair_method",
                                        "description": "Get the machine name and model number and error description keyword to query and find similar fixed errors in repair history. Call this function when you need statistics on the operating history of the requested machine., for example when a customer asks 'Hãy cho tôi biết về các cách xử lý lỗi này đã được thực hiện trong quá khứ'. If you do not know about the machine information, ask the user.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "machine_name": {
                                                    "type": "string",
                                                    "description": "The name of the machine",
                                                },
                                                "machine_id": {
                                                    "type": "string",
                                                    "description": "The model number of the machine",
                                                },
                                                "error_description": {
                                                    "type": "string",
                                                    "description": "Several keywords that describe the error",
                                                },
                                            },
                                            "required": ["machine_name", "model_number", "error_description"],
                                            "additionalProperties": False,
                                            },
                                        },
                            

                                        
                        },
                        
                        {
                            "type": "function",
                            "function": {
                                        "name": "recommend_maintenance",
                                        "description": "Get machine id of the machine that need predicting maintenance date. Call this function when you need to recommend maintenance for a machine, If you do not know about the machine information, ask the user.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "machine_id": {
                                            "type": "string",
                                            "description": "The id of the machine",
                                        },
                                    },
                                    "required": ['machine_id'],
                                    "additionalProperties": False,
                                    }
                                }
                        },  
                        {
                                "type": "function",
                                "function": {
                                            "name": "covariate_effects_on_machine",
                                            "description": "Get machine id of the machine that need to find relations between machine's risks. Call this function when you need to recommend maintenance for a machine, If you do not know about the machine information, ask the user.",
                                            "parameters": {
                                                "type": "object",
                                                "properties": {
                                                    "machine_id": {
                                                        "type": "string",
                                                        "description": "The id of the machine",
                                                    },
                                                },
                                                        "required": ['machine_id'],
                                                        "additionalProperties": False,
                                                }
                                            }
                        },  
          
                
                    ]
        self.chat_history = [{
                            "role": "system",
                            "content": "You are an intelligent and helpful assistant specializing in supporting operations at Denso Vietnam, a leading export manufacturing enterprise that supplies automotive components. Your primary role is to assist with factory operations, provide insights into production processes, help optimize workflows, and answer technical queries related to automotive parts manufacturing and assembly. Ensure your responses are clear, concise, and tailored to the needs of factory personnel." 
                            },]
        if model_name is None:
            self.MODEL = 'ft:gpt-4o-2024-08-06:personal::AXT1Yomi'
            
        self.retrieval_model = RetrievalModel()
        
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return f"data:image/png;base64,{encode_image(image_path)}" #base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_response(self, prompt, image_link= None):
        # update user prompt
        self.chat_history.append({
            "role": "user",
            "content": prompt
        })
        # add RAG queried information
        try: 
            query_prompt = self.retrieval_model.augment_query(prompt)
        except Exception as e:
            print("Error: ", e)
            return 
            
        ## create message
        message = self.chat_history.copy()
        message.append({
            "role": "user",
            "content": query_prompt
        })
        if image_link is not None:
            if not image_link.startswith("data:image/png;base64"):
                image = self.encode_image(image_link)
            else:
                image = image_link
            message.append({ "role": "user", "content": [
                            {
                            "type": "image_url",
                            "image_url": {
                                "url":  image
                            }
                            }
                        ]
                        })
        
        ## generate response
        response = self.client.chat.completions.create(
                model= self.MODEL,
                messages=message,
                tools=self.tools,
                n =1 
            )
    
        answer = self.process_response(prompt, response)
        
        if len(self.chat_history) > 3:
            self.reduce_context_length()
        
        return answer
            
    def process_response(self, prompt, response):
        """ 
        Process the response from the model. This function checks the finish_reason of the response and handles it accordingly.
        
        Args:
        - prompt: The prompt that was used to generate the response
        - response: The response from the model
        """
        
        # Check if the conversation was too long for the context window
        if response.choices[0].finish_reason  == "length":
            print("Error: The conversation was too long for the context window.")
            return self.handle_length_error(prompt)
            
        # Check if the model's output included copyright material (or similar)
        if response.choices[0].finish_reason == "content_filter":
            print("Error: The content was filtered due to policy violations.")
            return self.handle_content_filter_error(response)
        
        # Check if the model has made a tool_call. This is the case either if the "finish_reason" is "tool_calls" or if the "finish_reason" is "stop" and our API request had forced a function call
        if response.choices[0].finish_reason == "tool_calls":
            
            print("Model made a tool call.")
            return self.handle_tool_call(response)
             
        # Else finish_reason is "stop", in which case the model was just responding directly to the user
        elif response.choices[0].finish_reason == "stop":
            # Handle the normal stop case
            print("Model responded directly to the user.")
            return self.handle_normal_response(response)
            
        # Catch any other case, this is unexpected
        else:
            print("Unexpected finish_reason:", response.choices[0].finish_reason)
            return self.handle_unexpected_case(response)
            
    def handle_normal_response(self, response):
        self.chat_history.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    
    def reduce_context_length(self, force_reduce= False):
        if force_reduce:
            self.chat_history = [self.chat_history[0]] + self.chat_history[len(self.chat_history) - 2 :]
        else:
            self.chat_history = [self.chat_history[0]] + self.chat_history[3:]
    
    def handle_length_error(self, prompt):
        print("Error: The conversation was too long for the context window. Processing the prompt again.")
        self.reduce_context_length(force_reduce= True)
        return self.generate_response(prompt)
    
    def handle_content_filter_error(self, response):
        print("Error: The content was filtered due to policy violations.")
        return "Error: The content was filtered due to policy violations. Please try again with a different prompt."
    
    def handle_tool_call(self, response):
        # Extract the tool call from the response
        # tool_call = response.choices[0].message.tool_calls[0]
        call_function = response.choices[0].message.tool_calls[0].function.name
        arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        function_call_result = handle_function_calling(call_function, arguments)
        function_call_result_message = {"role": "tool",
                                "content": function_call_result,
                                "tool_call_id": response.choices[0].message.tool_calls[0].id}
        message = self.chat_history.copy()
        message.append(response.choices[0].message)
        message.append(function_call_result_message)
        
        new_response = self.client.chat.completions.create(
            model= self.MODEL,
            messages= message,
        )
        
        self.chat_history.append({
            "role": "assitant",
            "content": new_response.choices[0].message.content
        })
        
        return new_response.choices[0].message.content
    
    def handle_unexpected_case(self, response):
        print("Unexpected finish_reason:", response['choices'][0]['message']['finish_reason'])
        
        return f"Unexpected finish_reason: {response['choices'][0]['message']['finish_reason']}"
        
    def get_chat_history(self):
        return self.chat_history


    
def main():
    load_dotenv() 
    api_key = os.getenv("GPT_API")
    model = GPT_APIModel(api_key)
    while True:
        prompt = input("Insert question: ")
        if prompt == "exit":
            break
        image_link = input("Insert image link(skip if not include image): ")
        image_link = None if image_link.strip() == "" else image_link
        response = model.generate_response(prompt, image_link= image_link)
        print(response)
    
if __name__ == "__main__":
    main()