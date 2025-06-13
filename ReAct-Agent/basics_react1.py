import os
import re
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"]=""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model=genai.GenerativeModel('gemini-1.5-flash')

def calculator(input_expr:str)->str:
    try:
        input_expr=input_expr.strip()
        print("input_expr:", input_expr)
        result=eval(input_expr, {"__builtins__": {}})
        print(f"result: {result}")
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def react_agent_repl(question):
    history=[]
    while True:
        ptr=0
        prompt=f"""
            You are a ReAct-style AI assistant. Follow this loop:

            Format:
            Thought: Reason about what to do next.
            Action: "Calculator(expression)" or "Finish(final_answer)"
            Observation: Result of the action.

            Available Tools:
            - Calculator(expression)

            Only respond in the format above.

            Question: {question}
        """
        for step in history:
            prompt += f"\nThought: {step['thought']}\nAction: {step['action']}\nObservation: {step['observation']}"
        prompt += "\nThought:"
        print(f"Prompt {ptr}: {prompt}")

        thought_response=model.generate_content(prompt)
        thought=thought_response.text.strip().split("\n")[0].replace("Thought: ", "")
        print(f"\nThought: {thought}")
        print(f"prompT {ptr}: {prompt}")

        action_prompt=prompt + f"{thought}\nAction:"
        action_response=model.generate_content(action_prompt)
        action_line=action_response.text.strip().split("\n")[0].replace("Action: ", "")
        print(f"🔧 Action: {action_line}")

        if action_line.startswith("Finish("):
            final_answer=action_line[len("Finish("):-1]
            print(f"Final Answer: {final_answer}")
            break

        match=re.match(r"(\w+)\((.*)\)", action_line.strip())
        if match:
            tool_name=match.group(1)
            tool_input=match.group(2)

            if tool_name == "Calculator":
                result=calculator(tool_input)
            else:
                result="Unknown tool"

            print(f"Observation: {result}")
            history.append({
                "thought": thought,
                "action": action_line,
                "observation": result
            })
        else:
            print("Invalid action. Ending...")
            break

react_agent_repl("What is ((7 + 3) * 4 / 4)?")
