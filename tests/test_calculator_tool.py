from tools.calculator import CalculatorTool


tool = CalculatorTool()


result = tool.run(
    "125 * 8 / 100"
)


print("\n==============================")
print("CALCULATOR TOOL")
print("==============================")


print(result)