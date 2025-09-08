# %%
from dotenv import load_dotenv
import os
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
exa_api_key = os.getenv("EXA_API_KEY")

# %%
from langchain_groq import ChatGroq

# Replace with your actual Groq API key
groq_api_key = groq_api_key

llm = ChatGroq(
model="llama-3.3-70b-versatile",
api_key=groq_api_key,
temperature=0.7
)

# %%
from exa_py import Exa
from pydantic import BaseModel, Field
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],   # you can restrict later e.g. ["https://x.com"]
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# Define schema for structured output
class FactCheck(BaseModel):
    verdict: str = Field(..., description="Estimated truth verdict (e.g. 'True', 'False', 'Uncertain')")
    response: str = Field(..., description="Concise summary of the fact check based on evidence")
    sources: List[str] = Field(..., description="List of source URLs or references used")
class FactCheckRequest(BaseModel):
    claim: str


# Init Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0
)

# Example pipeline
Claim = "There is a video of a fish poking it's head out of water, opening it's mouth to eat egg yolk. Is this natural fish behaviour?"


exa = Exa(api_key=exa_api_key)
Exa_answer = ""

# =================================================

@app.get("/")
def home():
    return {"message": "API is live"}



from fastapi.responses import StreamingResponse

@app.post("/factcheck/stream")
def factcheck_stream(req: FactCheckRequest):
    def generate():
        yield "⏳ Starting fact-check...\n"
        exa_answer = ""

        # Stream chunks as they arrive
        for chunk in exa.stream_answer("is it true that " + req.claim):
            if chunk.content:
                exa_answer += chunk.content
                yield chunk.content + "\n"

        # After Exa finishes, call the LLM
        structured_llm = llm.with_structured_output(FactCheck)
        fact_check = structured_llm.invoke(f"Question: {req.claim}\nEvidence:\n{exa_answer}")
        yield f"\n✅ Final Verdict:\n{fact_check}\n"

    return StreamingResponse(generate(), media_type="text/plain")
    
'''
# Then, use stream_answer with your original query
result = exa.stream_answer(
   "is it true that " + Claim,
)


# Process the streaming response
for chunk in result:
   if chunk.content:
       Exa_answer  += chunk.content

# Create structured output chain
structured_llm = llm.with_structured_output(FactCheck)

input_text = f"Question: is it true that {Claim}\nEvidence:\n{Exa_answer}"
fact_check = structured_llm.invoke(input_text)


# %%
print(fact_check)
'''
