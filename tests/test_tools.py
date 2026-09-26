from agents.tool_registry import ToolRegistry
from agents.tool_executor import ToolExecutor


print("\n==============================")
print("AURA TOOL SYSTEM TEST")
print("==============================")


# =========================================================
# Example tools
# =========================================================

def calculator(a, b):
    return a + b


def greet(name):
    return f"Hello, {name}!"


# =========================================================
# Registry
# =========================================================

registry = ToolRegistry()

registry.register(
    name="calculator",
    tool=calculator,
    description="Adds two numbers."
)

registry.register(
    name="greet",
    tool=greet,
    description="Greets a user."
)


print("\nREGISTERED TOOLS")
print(registry.names())


# =========================================================
# Executor
# =========================================================

executor = ToolExecutor(
    registry
)


# =========================================================
# Calculator
# =========================================================

result = executor.execute(
    "calculator",
    {
        "a": 10,
        "b": 20
    }
)

print("\nCALCULATOR")
print(result)


# =========================================================
# Greeting
# =========================================================

result = executor.execute(
    "greet",
    {
        "name": "AURA"
    }
)

print("\nGREETING")
print(result)


# =========================================================
# Unknown tool
# =========================================================

result = executor.execute(
    "unknown_tool",
    {}
)

print("\nUNKNOWN TOOL")
print(result)


# =========================================================
# Tool descriptions
# =========================================================

print("\nTOOL DESCRIPTIONS")

for item in registry.descriptions():
    print(item)


print("\n==============================")
print("TOOL SYSTEM READY")
print("==============================")