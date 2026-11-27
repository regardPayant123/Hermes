"""LangGraph orchestration for LLM selection, history, and fallbacks."""

from collections import deque
from typing import Any, Iterable, TypedDict

from engines.tools.interface import ToolRegistry

from .llm.interface import ChatMessage, LLMProvider, LLMResponse


class _GraphState(TypedDict, total=False):
    messages: list[ChatMessage]
    temperature: float
    max_tokens: int
    response: LLMResponse | None
    failures: list[str]
    active_provider: int
    tool_rounds: int


class LLMOrchestrator:
    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: Iterable[LLMProvider] = (),
        *,
        system_prompt: str | None = None,
        history_limit: int = 20,
        tools: ToolRegistry | None = None,
        max_tool_rounds: int = 4,
    ):
        if history_limit <= 0:
            raise ValueError("history_limit must be greater than zero")
        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be greater than zero")
        self.primary = primary
        self.fallbacks = tuple(fallbacks)
        self.system_prompt = system_prompt
        self.history: deque[ChatMessage] = deque(maxlen=history_limit)
        self.tools = tools or ToolRegistry()
        self.max_tool_rounds = max_tool_rounds
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph is required for LLM orchestration; install it with: pip install langgraph"
            ) from exc

        providers = (self.primary, *self.fallbacks)
        graph = StateGraph(_GraphState)

        for index, provider in enumerate(providers):
            node_name = f"provider_{index}"

            async def invoke_provider(
                state: _GraphState,
                *,
                selected_provider: LLMProvider = provider,
                selected_index: int = index,
            ) -> dict[str, Any]:
                try:
                    options: dict[str, Any] = {
                        "temperature": state["temperature"],
                        "max_tokens": state["max_tokens"],
                    }
                    tool_schemas = self.tools.schemas()
                    if tool_schemas:
                        options["tools"] = tool_schemas
                    response = await selected_provider.generate(state["messages"], **options)
                    return {"response": response, "active_provider": selected_index}
                except Exception as exc:
                    failure = f"{selected_provider.__class__.__name__}: {exc}"
                    return {
                        "response": None,
                        "active_provider": selected_index,
                        "failures": [*state.get("failures", []), failure],
                    }

            graph.add_node(node_name, invoke_provider)

        async def execute_tools(state: _GraphState) -> dict[str, Any]:
            response = state["response"]
            if response is None:
                return {}
            if state["tool_rounds"] >= self.max_tool_rounds:
                return {
                    "response": LLMResponse(
                        text="Limite d'appels aux outils atteinte.",
                        model=response.model,
                        finish_reason="tool_limit",
                    )
                }

            assistant_message = ChatMessage(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls,
            )
            results = [await self.tools.execute(call) for call in response.tool_calls]
            return {
                "messages": [*state["messages"], assistant_message, *results],
                "response": None,
                "tool_rounds": state["tool_rounds"] + 1,
            }

        graph.add_node("execute_tools", execute_tools)
        graph.add_edge(START, "provider_0")
        for index in range(len(providers)):
            node_name = f"provider_{index}"
            next_node = f"provider_{index + 1}" if index + 1 < len(providers) else END

            def route_provider(state: _GraphState, fallback=next_node):
                response = state.get("response")
                if response is None:
                    return fallback
                if response.tool_calls:
                    return "execute_tools"
                return END

            graph.add_conditional_edges(
                node_name,
                route_provider,
                {
                    END: END,
                    "execute_tools": "execute_tools",
                    **({next_node: next_node} if next_node != END else {}),
                },
            )

        graph.add_conditional_edges(
            "execute_tools",
            lambda state: f"provider_{state['active_provider']}",
            {f"provider_{index}": f"provider_{index}" for index in range(len(providers))},
        )

        return graph.compile()

    def clear_history(self) -> None:
        self.history.clear()

    def set_primary(self, provider: LLMProvider) -> None:
        self.primary = provider
        self._graph = self._build_graph()

    def _messages(self, user_message: str) -> list[ChatMessage]:
        messages = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.extend(self.history)
        messages.append(ChatMessage(role="user", content=user_message))
        return messages

    async def run(
        self,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        if not user_message.strip():
            raise ValueError("user_message cannot be empty")

        state = await self._graph.ainvoke(
            {
                "messages": self._messages(user_message),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response": None,
                "failures": [],
                "active_provider": 0,
                "tool_rounds": 0,
            }
        )
        response = state.get("response")
        if response is None:
            failures = state.get("failures", [])
            raise RuntimeError("All LLM providers failed: " + " | ".join(failures))

        history_start = len(self.history) + (1 if self.system_prompt else 0)
        self.history.extend(state["messages"][history_start:])
        self.history.append(ChatMessage(role="assistant", content=response.text))
        return response
