# agents/reasoning_agent.py
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from agents.base_agent import AsyncAgent
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat

# from utils.activity_logger import ActivityLogger  # ← your class

from vector_store.rag.activity_retriever import ActivityRetriever
from vector_store.rag.paper_retriever import PaperRetriever
from vector_store.rag.wiki_retriever import WikiRetriever
from vector_store.rag.multi_retriever import MultiSourceRetriever



class ReasoningAgent(AsyncAgent):
    def __init__(self, vlm, event_bus, llm_config, max_workers: int = 2):
        super().__init__("ReasoningAgent", vlm, event_bus)
        self.llm_config = llm_config
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 🧠 Multi-source RAG setup
        self.multi_retriever = MultiSourceRetriever([
            ("activity_log", ActivityRetriever()),
            ("research_papers", PaperRetriever()),
            ("local_wikipedia", WikiRetriever())
        ])

        event_bus.subscribe("memory_summary", self.handle_summary)
        event_bus.subscribe("user_goal", self.handle_goal)
        event_bus.subscribe("system_query", self.handle_query)

    def _run_autogen_with_rag(self, task_description: str):
        """Reason using multiple RAG sources."""
        try:
            # Retrieve across all memory sources
            retrieved = self.multi_retriever.search(task_description, n_results=5)
            context_text = "\n".join([
                f"[{r['source']}] {r['document']} (meta: {r['metadata']})"
                for r in retrieved
            ])

            enriched_prompt = f"""
You are a reasoning assistant with access to multi-source memory.

User Query / Goal:
{task_description}

Relevant Context:
{context_text or 'No related information found.'}

Think carefully, synthesize across these sources, and produce a coherent reasoning or plan.
"""

            planner = AssistantAgent("Planner", llm_config=self.llm_config)
            thinker = AssistantAgent("Thinker", llm_config=self.llm_config)
            critic = AssistantAgent("Critic", llm_config=self.llm_config)
            user = UserProxyAgent("System")

            group = SelectorGroupChat([user, planner, thinker, critic])
            result = group.start(enriched_prompt)

            asyncio.run_coroutine_threadsafe(
                self.event_bus.publish("proactive_insight", {
                    "text": f"Reasoning result: {result}",
                    "timestamp": datetime.now().isoformat()
                }),
                asyncio.get_event_loop()
            )

        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                self.event_bus.publish("proactive_insight", {
                    "text": f"Reasoning failed: {str(e)}"
                }),
                asyncio.get_event_loop()
            )
    async def run(self):
        while True:
            await asyncio.sleep(3600)
