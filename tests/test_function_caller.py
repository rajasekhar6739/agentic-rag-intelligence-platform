from agents.function_caller import FunctionCaller


agent = FunctionCaller()


result = agent.run(
    "Calculate 1250 multiplied by 18 percent."
)


print("\n==============================")
print("FUNCTION CALLING")
print("==============================")


print("Answer:")
print(result["answer"])


print("\nTool calls:")

for call in result["tool_calls"]:
    print(call)