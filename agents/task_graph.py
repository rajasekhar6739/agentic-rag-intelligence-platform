from __future__ import annotations

from typing import Any, Dict, List, Optional


class TaskGraph:
    """Dependency-aware execution graph."""

    def __init__(self, executor: Optional[Any] = None):
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.executor = executor

    def add_task(
        self,
        task_id: int,
        tool: str,
        description: str = "",
        arguments: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[int]] = None,
    ):
        task = {
            "id": int(task_id),
            "tool": str(tool),
            "description": description or "",
            "arguments": dict(arguments or {}),
            "depends_on": [
                int(x) for x in (depends_on or [])
            ],
        }

        self.tasks[int(task_id)] = task
        return task

    def add_tasks(self, tasks):
        for task in tasks:
            self.add_task(
                task_id=task["id"],
                tool=task["tool"],
                description=task.get("description", ""),
                arguments=task.get("arguments", {}),
                depends_on=task.get("depends_on", []),
            )
        return self

    def get_task(self, task_id):
        return self.tasks.get(int(task_id))

    def list_tasks(self):
        return list(self.tasks.values())

    def clear(self):
        self.tasks.clear()
        return self

    def _normalize_tasks(self, tasks=None):
        source = self.tasks if tasks is None else tasks

        if callable(source):
            source = self.tasks

        if isinstance(source, dict):
            values = list(source.values())
        elif isinstance(source, (list, tuple, set)):
            values = list(source)
        else:
            values = []

        normalized = {}

        for task in values:
            if not isinstance(task, dict):
                continue

            task_id = task.get("id")
            if task_id is None:
                continue

            task_id = int(task_id)

            dependencies = task.get("depends_on", []) or []

            normalized[task_id] = {
                "id": task_id,
                "tool": task.get("tool"),
                "description": task.get("description", ""),
                "arguments": dict(
                    task.get("arguments", {}) or {}
                ),
                "depends_on": [
                    int(x)
                    for x in dependencies
                    if not isinstance(x, dict)
                ],
            }

        return normalized

    def execution_order(self, tasks=None):
        task_map = self._normalize_tasks(tasks)

        order = []
        visited = set()
        visiting = set()

        def visit(task_id):
            if task_id in visited:
                return

            if task_id in visiting:
                raise ValueError(
                    f"Circular dependency detected at task {task_id}"
                )

            if task_id not in task_map:
                raise ValueError(
                    f"Unknown dependency: {task_id}"
                )

            visiting.add(task_id)

            for dependency in task_map[task_id]["depends_on"]:
                visit(dependency)

            visiting.remove(task_id)
            visited.add(task_id)
            order.append(task_id)

        for task_id in task_map:
            visit(task_id)

        return order

    def _execute_single(self, task, previous_results):
        if self.executor is None:
            return {
                "success": False,
                "tool": task["tool"],
                "error": "No task executor configured",
            }

        tool = task["tool"]
        arguments = dict(task.get("arguments", {}))

        dependencies = task.get("depends_on", [])

        dependency_results = {
            dep_id: previous_results.get(dep_id)
            for dep_id in dependencies
        }

        if dependency_results:
            arguments["_dependency_results"] = dependency_results

        try:
            if hasattr(self.executor, "execute"):
                return self.executor.execute(
                    tool,
                    arguments,
                )

            if callable(self.executor):
                return self.executor(
                    tool,
                    arguments,
                )

            return {
                "success": False,
                "tool": tool,
                "error": "Invalid task executor",
            }

        except Exception as exc:
            return {
                "success": False,
                "tool": tool,
                "error": str(exc),
            }

    def execute(
        self,
        tasks=None,
        executor=None,
        user_query=None,
    ):
        # Compatibility:
        # execute(fake_executor, "query")
        if callable(tasks):
            if executor is None or isinstance(executor, str):
                user_query = executor
                executor = tasks
                tasks = None

        if executor is not None:
            self.executor = executor

        task_map = self._normalize_tasks(tasks)
        order = self.execution_order(task_map)

        results = {}

        for task_id in order:
            task = task_map[task_id]

            print(f"\nEXECUTING: {task_id}")
            print(f"ARGUMENTS: {task}")

            result = self._execute_single(
                task,
                results,
            )

            if result is None:
                result = {
                    "success": False,
                    "tool": task["tool"],
                    "error": "Executor returned None",
                }

            results[task_id] = result

        success = all(
            isinstance(results.get(task_id), dict)
            and results[task_id].get("success", False)
            for task_id in order
        )

        return {
            "success": success,
            "execution_order": order,
            "results": results,
        }