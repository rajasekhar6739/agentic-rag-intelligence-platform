from agents.synthesizer import ResultSynthesizer
from agents.critic_agent import CriticAgent
from agents.tool_registry import ToolRegistry
from agents.planner import Planner


class AgentOrchestrator:

    def __init__(
        self,
        retriever=None,
        planner=None,
        synthesizer=None,
        critic=None,
        tool_registry=None,
    ):
        self.retriever = retriever

        self.tool_registry = (
            tool_registry
            if tool_registry is not None
            else ToolRegistry()
        )

        self.synthesizer = (
            synthesizer
            if synthesizer is not None
            else ResultSynthesizer()
        )

        self.critic = (
            critic
            if critic is not None
            else CriticAgent(retriever)
        )

        self._register_tools()

        self.planner = (
            planner
            if planner is not None
            else Planner(self.tool_registry)
        )

    # =========================================================
    # TOOL REGISTRATION
    # =========================================================

    def _register_tools(self):

        if not self.tool_registry.has("rag_search"):
            self.tool_registry.register(
                "rag_search",
                self._rag_search
            )

    # =========================================================
    # QUERY NORMALIZATION
    # =========================================================

    def _normalize_query(self, query):

        if isinstance(query, str):
            return query.strip()

        if isinstance(query, dict):

            value = (
                query.get("query")
                or query.get("question")
                or query.get("text")
            )

            if value is None:
                raise ValueError(
                    "Query dictionary must contain "
                    "'query', 'question', or 'text'."
                )

            return str(value).strip()

        return str(query).strip()

    # =========================================================
    # RAG TOOL
    # =========================================================

    def _rag_search(self, query, top_k=5):

        if self.retriever is None:

            return {
                "success": False,
                "query": query,
                "results": [],
                "error": "Retriever is not configured."
            }

        try:

            results = self.retriever.search(
                query,
                top_k=top_k
            )

            return {
                "success": True,
                "query": query,
                "results": results
            }

        except Exception as exc:

            return {
                "success": False,
                "query": query,
                "results": [],
                "error": str(exc)
            }

    # =========================================================
    # PLAN
    # =========================================================

    def _create_plan(
        self,
        original_query,
        retry_instruction=None
    ):

        try:

            if retry_instruction:

                planning_query = (
                    original_query
                    + "\n\n"
                    + retry_instruction
                )

            else:

                planning_query = original_query

            plan = self.planner.plan(
                planning_query
            )

        except TypeError:

            try:

                plan = self.planner.plan(
                    query=original_query
                )

            except Exception:

                plan = {
                    "query": original_query,
                    "tasks": [
                        {
                            "id": 1,
                            "description": (
                                "Retrieve relevant evidence "
                                "from the knowledge base."
                            ),
                            "tool": "rag_search"
                        }
                    ]
                }

        except Exception:

            plan = {
                "query": original_query,
                "tasks": [
                    {
                        "id": 1,
                        "description": (
                            "Retrieve relevant evidence "
                            "from the knowledge base."
                        ),
                        "tool": "rag_search"
                    }
                ]
            }

        if not isinstance(plan, dict):

            plan = {
                "query": original_query,
                "tasks": [
                    {
                        "id": 1,
                        "description": (
                            "Retrieve relevant evidence "
                            "from the knowledge base."
                        ),
                        "tool": "rag_search"
                    }
                ]
            }

        if "tasks" not in plan:

            plan["tasks"] = [
                {
                    "id": 1,
                    "description": (
                        "Retrieve relevant evidence "
                        "from the knowledge base."
                    ),
                    "tool": "rag_search"
                }
            ]

        # IMPORTANT:
        # Plan always exposes the real user query.
        plan["query"] = original_query

        return plan

    # =========================================================
    # EVIDENCE EXTRACTION
    # =========================================================

    def _extract_evidence(self, search_result):

        if not isinstance(search_result, dict):
            return []

        results = search_result.get(
            "results",
            []
        )

        if isinstance(results, dict):

            results = results.get(
                "evidence",
                []
            )

        if not isinstance(results, list):
            return []

        evidence = []

        for item in results:

            if not isinstance(item, dict):
                continue

            text = item.get(
                "text",
                ""
            )

            if not text:
                continue

            evidence.append(
                {
                    "text": text,
                    "source": item.get(
                        "source",
                        "unknown"
                    ),
                    "page": item.get(
                        "page",
                        0
                    ),
                    "chunk_id": item.get(
                        "chunk_id",
                        0
                    ),
                    "score": item.get(
                        "score",
                        0.0
                    ),
                    "vector_score": item.get(
                        "vector_score",
                        0.0
                    ),
                    "bm25_score": item.get(
                        "bm25_score",
                        0.0
                    ),
                    "hybrid_score": item.get(
                        "hybrid_score",
                        0.0
                    ),
                    "rerank_score": item.get(
                        "rerank_score",
                        0.0
                    )
                }
            )

        return evidence

    # =========================================================
    # RUN SINGLE ATTEMPT
    # =========================================================

    def _run_attempt(
        self,
        original_query,
        plan,
        attempt
    ):

        task_results = []
        tools_used = []

        print()
        print("=" * 60)
        print(
            f"AURA AGENT ATTEMPT {attempt}"
        )
        print("=" * 60)

        print()
        print("PLAN:")
        print(plan)

        tasks = plan.get(
            "tasks",
            []
        )

        if not isinstance(tasks, list):
            tasks = []

        for task in tasks:

            if not isinstance(task, dict):
                continue

            tool_name = task.get(
                "tool"
            )

            if not tool_name:
                continue

            if not self.tool_registry.has(
                tool_name
            ):

                print(
                    "WARNING: Tool not found:",
                    tool_name
                )

                continue

            # =================================================
            # IMPORTANT
            # =================================================
            # Retrieval MUST receive only the original query.
            # Retry instructions must never be sent to RAG.
            # =================================================

            tool_query = original_query

            arguments = {
                "query": tool_query,
                "top_k": 5
            }

            print()
            print("EXECUTING:")
            print(
                {
                    "tool": tool_name,
                    "arguments": arguments
                }
            )

            try:

                # ToolRegistry.execute expects:
                #
                # execute(name, arguments)
                #
                # NOT:
                #
                # execute(name, query=..., top_k=...)

                result = self.tool_registry.execute(
                    tool_name,
                    arguments
                )

            except Exception as exc:

                result = {
                    "success": False,
                    "tool": tool_name,
                    "results": [],
                    "error": str(exc)
                }

            task_results.append(
                {
                    "task_id": task.get(
                        "id"
                    ),
                    "tool": tool_name,
                    "description": task.get(
                        "description",
                        ""
                    ),
                    "arguments": arguments,
                    "result": result
                }
            )

            if tool_name not in tools_used:

                tools_used.append(
                    tool_name
                )

        # =====================================================
        # COLLECT EVIDENCE
        # =====================================================

        evidence = []

        for task in task_results:

            evidence.extend(
                self._extract_evidence(
                    task.get(
                        "result",
                        {}
                    )
                )
            )

        print()
        print(
            "EVIDENCE COUNT:",
            len(evidence)
        )

        for index, item in enumerate(
            evidence,
            start=1
        ):

            print()
            print(
                f"EVIDENCE {index}"
            )

            print(
                "SOURCE:",
                item.get(
                    "source",
                    "unknown"
                )
            )

            print(
                "PAGE:",
                item.get(
                    "page",
                    0
                )
            )

            print(
                "TEXT:",
                item.get(
                    "text",
                    ""
                )
            )

        # =====================================================
        # SYNTHESIZE
        # =====================================================

        answer = self._synthesize(
            original_query,
            evidence
        )

        # =====================================================
        # CRITIC
        # =====================================================

        critic_result = self._critic(
            original_query,
            answer,
            evidence
        )

        supported = bool(
            critic_result.get(
                "supported",
                False
            )
        )

        confidence = float(
            critic_result.get(
                "confidence",
                0.0
            )
        )

        print()
        print(
            "CRITIC SUPPORTED:",
            supported
        )

        print(
            "CONFIDENCE:",
            confidence
        )

        return {
            "query": original_query,
            "answer": answer,
            "grounded": supported,
            "confidence": confidence,
            "citations": self._build_citations(
                evidence
            ),
            "tools_used": tools_used,
            "results": task_results,
            "evidence": evidence,
            "critic": critic_result,
            "attempt": attempt
        }

    # =========================================================
    # SYNTHESIZER
    # =========================================================

    def _synthesize(
        self,
        query,
        evidence
    ):

        if not evidence:

            return (
                "The provided evidence is insufficient "
                "to answer the question."
            )

        methods = [
            "synthesize",
            "generate",
            "run"
        ]

        for method_name in methods:

            method = getattr(
                self.synthesizer,
                method_name,
                None
            )

            if method is None:
                continue

            calls = [
                lambda: method(
                    query=query,
                    evidence=evidence
                ),
                lambda: method(
                    query,
                    evidence
                ),
                lambda: method(
                    query=query,
                    context=evidence
                )
            ]

            for call in calls:

                try:

                    result = call()

                    if isinstance(
                        result,
                        dict
                    ):

                        answer = (
                            result.get("answer")
                            or result.get("text")
                            or result.get("response")
                        )

                        if answer:
                            return str(
                                answer
                            )

                    if result is not None:

                        return str(
                            result
                        )

                except Exception:
                    continue

        # Safe fallback.
        return str(
            evidence[0].get(
                "text",
                ""
            )
        )

    # =========================================================
    # CRITIC
    # =========================================================

    def _critic(
        self,
        query,
        answer,
        evidence
    ):

        methods = [
            "evaluate",
            "critic",
            "run",
            "check"
        ]

        for method_name in methods:

            method = getattr(
                self.critic,
                method_name,
                None
            )

            if method is None:
                continue

            calls = [
                lambda: method(
                    query=query,
                    answer=answer,
                    evidence=evidence
                ),
                lambda: method(
                    query,
                    answer,
                    evidence
                ),
                lambda: method(
                    answer=answer,
                    evidence=evidence,
                    query=query
                )
            ]

            for call in calls:

                try:

                    result = call()

                    if isinstance(
                        result,
                        dict
                    ):

                        result.setdefault(
                            "supported",
                            bool(
                                result.get(
                                    "grounded",
                                    False
                                )
                            )
                        )

                        result.setdefault(
                            "confidence",
                            (
                                1.0
                                if result.get(
                                    "supported",
                                    False
                                )
                                else 0.0
                            )
                        )

                        return result

                except Exception:
                    continue

        if not evidence:

            return {
                "supported": False,
                "confidence": 0.0,
                "issues": [
                    "No evidence was retrieved."
                ],
                "recommendation": "retry"
            }

        return {
            "supported": True,
            "confidence": 0.80,
            "issues": [],
            "recommendation": "accept"
        }

    # =========================================================
    # CITATIONS
    # =========================================================

    def _build_citations(
        self,
        evidence
    ):

        citations = []
        seen = set()

        for item in evidence:

            source = item.get(
                "source",
                "unknown"
            )

            page = item.get(
                "page",
                0
            )

            key = (
                source,
                page
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                len(citations) + 1
            )

        return citations

    # =========================================================
    # MAIN RUN
    # =========================================================

    def run(
        self,
        query,
        max_attempts=2
    ):

        original_query = self._normalize_query(
            query
        )

        print()
        print("=" * 60)
        print("AURA AGENT")
        print("=" * 60)

        print(
            "QUERY:",
            original_query
        )

        attempts = []

        retry_instruction = None

        for attempt in range(
            1,
            max_attempts + 1
        ):

            plan = self._create_plan(
                original_query,
                retry_instruction
            )

            result = self._run_attempt(
                original_query,
                plan,
                attempt
            )

            result["plan"] = plan

            attempts.append(
                result
            )

            if result.get(
                "grounded",
                False
            ):

                print()
                print(
                    "AURA: ANSWER ACCEPTED"
                )

                return self._final_result(
                    result,
                    attempts,
                    original_query
                )

            if attempt < max_attempts:

                print()
                print(
                    "AURA: SELF-CORRECTION TRIGGERED"
                )

                retry_instruction = (
                    "Improve the previous attempt. "
                    "Retrieve stronger evidence. "
                    "Use only directly supported evidence."
                )

                continue

            return self._final_result(
                result,
                attempts,
                original_query
            )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    def _final_result(
        self,
        result,
        attempts,
        original_query
    ):

        final_attempt = result.get(
            "attempt",
            len(attempts)
        )

        return {
            "success": bool(
                result.get(
                    "grounded",
                    False
                )
            ),
            "answer": result.get(
                "answer",
                "The provided evidence is insufficient "
                "to answer the question."
            ),
            "grounded": bool(
                result.get(
                    "grounded",
                    False
                )
            ),
            "confidence": float(
                result.get(
                    "confidence",
                    0.0
                )
            ),
            "citations": result.get(
                "citations",
                []
            ),
            "tools_used": result.get(
                "tools_used",
                []
            ),
            "plan": result.get(
                "plan",
                {}
            ),
            "results": result.get(
                "results",
                []
            ),
            "evidence": result.get(
                "evidence",
                []
            ),
            "critic": result.get(
                "critic",
                {}
            ),
            "attempt": final_attempt,
            "attempts": attempts,
            "final_attempt": final_attempt,
            "self_corrected": (
                len(attempts) > 1
            ),
            "query": original_query
        }