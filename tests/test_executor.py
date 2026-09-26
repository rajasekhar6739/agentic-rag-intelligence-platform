from agents.tool_executor import ToolExecutor


executor = ToolExecutor()


print("\n==============================")
print("AVAILABLE TOOLS")
print("==============================")


print(
    executor.available_tools()
)


print("\n==============================")
print("TOOL EXECUTION")
print("==============================")


result = executor.execute(
    "calculator",
    {
        "expression": "250 * 18 / 100"
    }
)


print(result)