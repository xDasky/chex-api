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



exa = Exa(api_key=exa_api_key)
Exa_answer = ""

# =================================================

@app.get("/")
def home():
    return {"message": "API is live"}

CheckCount = 0

from fastapi.responses import StreamingResponse
from fastapi.responses import Response

@app.head("/")
def healthcheck():
    return Response(status_code=200)



@app.post("/factcheck/stream")
def factcheck_stream(req: FactCheckRequest):
    def generate():
        global CheckCount

        claim_text = (req.claim or "").strip()
        if not claim_text:
            yield "⚠️ No text to be fact checked\n"
            return

        yield "⏳ Starting fact-check...\n"
        exa_answer = ""
        print("Fact Checking " + claim_text)
        CheckCount += 1

        try:
            # ✅ Stream chunks from Exa
            for chunk in exa.stream_answer("is it true that " + claim_text):
                if chunk.content:
                    exa_answer += chunk.content
                    yield chunk.content + "\n"

        except requests.exceptions.ChunkedEncodingError:
            yield "\n[Stream ended unexpectedly]\n"
            return  # don’t continue to LLM if stream broke
        except Exception as e:
            yield f"\n[Error while streaming: {str(e)}]\n"
            return

        # ✅ After Exa finishes, call the LLM
        try:
            structured_llm = llm.with_structured_output(FactCheck)
            fact_check = structured_llm.invoke(
                f"Question: {claim_text}\nEvidence:\n{exa_answer}"
            )
            yield f"\n✅ Final Verdict:\n{fact_check}\n"
        except Exception as e:
            yield f"\n[Error while generating verdict: {str(e)}]\n"

        print("done fact check")
        print(CheckCount)

    return StreamingResponse(generate(), media_type="text/plain")
    
