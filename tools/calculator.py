class CalculatorTool:

    name = "calculator"

    description = (
        "Perform basic mathematical calculations."
    )

    def run(self, expression: str) -> dict:

        allowed = set(
            "0123456789+-*/(). %"
        )

        if not expression:
            return {
                "success": False,
                "error": "Empty expression"
            }

        if any(
            char not in allowed
            for char in expression
        ):
            return {
                "success": False,
                "error": "Invalid expression"
            }

        try:

            result = eval(
                expression,
                {
                    "__builtins__": {}
                },
                {}
            )

            return {
                "success": True,
                "expression": expression,
                "result": result
            }

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc)
            }