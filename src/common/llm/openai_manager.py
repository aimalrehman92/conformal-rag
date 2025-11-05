import json
import os
import openai
from dotenv import load_dotenv
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from typing import List
from langchain.schema import Document
from io import BytesIO
  

class AssistantTools:
    def __init__(self, type: str = "file_search"):
        #File Search is a built-in RAG tool to process and search through files
        #For all openai assistant type, check https://platform.openai.com/docs/assistants/tools for different types
        self.type = type 
    def to_dict(self):
        return {"type": self.type}

class CreateAssistantReqDTO:
    def __init__(self, name: str, instructionkey: str):
        self.name = name
        self.description = f"assistant for instruction: {instructionkey}"
        self.tools = [AssistantTools().to_dict()]
        self.model = "gpt-4o"
        with open('instructions.json', 'r') as json_file:
            data = json.load(json_file)
            self.instructions = data[instructionkey]


class DocumentManager:
    # interact with OpenAI file management API to upload files to openAI vector store
    def __init__(self, name, client):
        self.client = client
        self.vector_store = self.client.beta.vector_stores.create(name = name)

    def upload(self, file_paths: List[str]):
        file_streams = [open(path, "rb") for path in file_paths]
        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
          vector_store_id=self.vector_store.id, files=file_streams
        )
        print(file_batch.status)
        return file_batch

    def get_vector_store_id(self):
        return self.vector_store.id

class EventHandler(AssistantEventHandler):    
  @override
  def on_text_created(self, text) -> None:
    print(f"\nassistant > ", end="", flush=True)
      
  @override
  def on_text_delta(self, delta, snapshot):
    print(delta.value, end="", flush=True)
      
  def on_tool_call_created(self, tool_call):
    print(f"\nassistant > {tool_call.type}\n", flush=True)
  
  def on_tool_call_delta(self, delta, snapshot):
    if delta.type == 'code_interpreter':
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)
 
# Then, we use the `stream` SDK helper 
# with the `EventHandler` class to create the Run 
# and stream the response.

class OpenAIManager:
    def __init__(self):
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.client = OpenAI()

    def create_openai_embeddings(self, texts, model="text-embedding-3-large"):
        if not texts:
            raise ValueError("Texts are not processed. Call process_pdf() first.")
        
        embeddings_list = []
        for i, text in texts:
            text = text.replace("\n", " ")
            res = self.client.embeddings.create(input=[text], model=model)
            embeddings_list.append(res.data[0].embedding)
            print(f'{i+1}/{len(texts)} embeddings done')

        return embeddings_list
    
    def initialize_assistant(self, createAssistantDTO):
        self.assistant = self.client.beta.assistants.create(
          name=createAssistantDTO.name,
          instructions=createAssistantDTO.instructions,
          tools=createAssistantDTO.tools,
          model=createAssistantDTO.model,
        )
        self.docManager = DocumentManager(createAssistantDTO.name, self.client)
    
    def upload_documents(self, paths):
        #upload array of path
        #TODO: path check
        print(self.docManager.upload(paths))
        self.assistant = self.client.beta.assistants.update(
          assistant_id=self.assistant.id,
          tool_resources={"file_search": {"vector_store_ids": [self.docManager.get_vector_store_id()]}},
        )
    
    def new_conversation(self):
        #create a new conversation thread and return thread id
        thread = self.client.beta.threads.create()
        return thread

    def get_response(self, thread, userInput):
        try:
            message = self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=userInput
            )
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        with self.client.beta.threads.runs.stream(
          thread_id=thread.id,
          assistant_id=self.assistant.id,
          event_handler=EventHandler(),
        ) as stream:
          stream.until_done()
        
def main():
  #sample usage
  openaiManager = OpenAIManager()
  assistantDTO = CreateAssistantReqDTO("readPaper", "readpaper")
  openaiManager.initialize_assistant(assistantDTO)
  doc1path = os.path.join(os.getcwd(), "documents","2024_Corrective_RAGv2.pdf")
  #print(doc1path)
  openaiManager.upload_documents([doc1path])
  thread = openaiManager.new_conversation()
  openaiManager.get_response(thread, "What is the paper about?")

if __name__ == "__main__":
    print("Running openai_manager.py")
    main()