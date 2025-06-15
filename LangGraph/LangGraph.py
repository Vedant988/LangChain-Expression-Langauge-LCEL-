import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDPRuvKv-giXi8eLzTrKb_wJHxnCkzIk6k"

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model=genai.GenerativeModel('gemini-1.5-flash')

class MyState(dict):
    pass

def ask_gemini(state):
    question=state["question"]
    response=model.generate_content(question)
    return {"question":question,"answer":response.text}

graph = StateGraph(MyState)
graph.add_node("Gemini", RunnableLambda(ask_gemini))
graph.set_entry_point("Gemini")
graph.set_finish_point("Gemini")

app = graph.compile()

result = app.invoke({"question": "What is LangGraph in LangChain?"})
print(result)
