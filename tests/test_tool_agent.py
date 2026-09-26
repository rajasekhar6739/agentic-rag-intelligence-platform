from agents.tool_agent import ToolAgent


agent = ToolAgent()


request = """
Calculate the percentage increase from
100 ms to 150 ms.
"""


result = agent.run(request)


print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(result["answer"])


print("\n==============================")
print("ITERATIONS")
print("==============================")

print(result["iterations"])


print("\n==============================")
print("EXECUTION TRACE")
print("==============================")


for event in result["execution_trace"]:

    print("\nIteration:")
    print(event["iteration"])

    print("Tool:")
    print(event["tool"])

    print("Arguments:")
    print(event["arguments"])

    print("Result:")
    print(event["result"])