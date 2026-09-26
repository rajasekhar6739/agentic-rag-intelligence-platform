import ast
import operator


class CalculatorTool:

    """
    Safe arithmetic calculator.

    Does NOT use eval().
    """

    OPERATORS = {

        ast.Add: operator.add,

        ast.Sub: operator.sub,

        ast.Mult: operator.mul,

        ast.Div: operator.truediv,

        ast.Mod: operator.mod,

        ast.Pow: operator.pow
    }

    # =========================================================
    # PUBLIC TOOL
    # =========================================================

    def __call__(
        self,
        expression: str
    ):

        expression = (
            expression
            .strip()
        )

        if not expression:

            raise ValueError(
                "Expression cannot be empty."
            )

        # Remove common natural-language prefixes.

        prefixes = [
            "calculate",
            "what is",
            "compute"
        ]

        lowered = expression.lower()

        for prefix in prefixes:

            if lowered.startswith(prefix):

                expression = expression[
                    len(prefix):
                ].strip()

                break

        tree = ast.parse(
            expression,
            mode="eval"
        )

        return self._evaluate(
            tree.body
        )

    # =========================================================
    # SAFE AST EVALUATION
    # =========================================================

    def _evaluate(
        self,
        node
    ):

        if isinstance(
            node,
            ast.Constant
        ):

            if isinstance(
                node.value,
                (int, float)
            ):

                return node.value

            raise ValueError(
                "Only numeric values are allowed."
            )

        if isinstance(
            node,
            ast.BinOp
        ):

            left = self._evaluate(
                node.left
            )

            right = self._evaluate(
                node.right
            )

            operation = self.OPERATORS.get(
                type(node.op)
            )

            if operation is None:

                raise ValueError(
                    "Unsupported operator."
                )

            return operation(
                left,
                right
            )

        if isinstance(
            node,
            ast.UnaryOp
        ):

            value = self._evaluate(
                node.operand
            )

            if isinstance(
                node.op,
                ast.USub
            ):

                return -value

            if isinstance(
                node.op,
                ast.UAdd
            ):

                return value

            raise ValueError(
                "Unsupported unary operator."
            )

        raise ValueError(
            "Invalid mathematical expression."
        )