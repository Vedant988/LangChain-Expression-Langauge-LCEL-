import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')
api_key = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=api_key)
model=genai.GenerativeModel('gemini-1.5-flash')


class MyState(TypedDict):
    question:str
    answer:Optional[str]

def ask_gemini(state: MyState)->dict:
    question=state.get("question")
    print(f"Node 'ask_gemini' received question:{question}")

    try:
        response=model.generate_content(question)
        
        if response.parts:
            answer=response.parts[0].text
        elif response.text:
            answer=response.text
        else:
            answer="Could not find text in the response."
            print(f"Gemini response was empty or blocked. Full response: {response}")

    except Exception as e:
        print(f"Gemini Error: {e}")
        answer="Failed to generate response due to an error."
    return{"answer":answer}


graph_builder=StateGraph(MyState)
graph_builder.add_node("Gemini",ask_gemini)
graph_builder.set_entry_point("Gemini")
graph_builder.add_edge("Gemini",END)
app = graph_builder.compile()

if __name__ == "__main__":
    input_data = {"question":"where is IIIT nagpur located in which part of Nagpur?"}    
    result = app.invoke(input_data)
    
    print("\nFinal Result:")
    print(result)