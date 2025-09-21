import asyncio
from tempfile import TemporaryDirectory

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import DuckDuckGoSearchRun

from providers.local_openAI import model_vlm

working_directory = TemporaryDirectory(dir=r'f:/models/')
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)  # If you don't provide a root_dir, operations will default to the current working directory
l = toolkit.get_tools()


async def main() -> None:
    tool = [LangChainToolAdapter(i) for i in l]
    tool = [*tool, LangChainToolAdapter(DuckDuckGoSearchRun())]
    model_client = model_function_callling
    agent = AssistantAgent(
        "assistant",
        # tools=[tool],
        tools=tool,
        model_client=model_client,
        system_message="You are an AI assistant with web search and file handling capabilities. Your task is to perform web searches and save results systematically.",
    )
    await Console(
        agent.on_messages_stream(
            [TextMessage(content="1- Search the web for Allama Iqbal.\n2- Create a new file locally with name iqbal.txt.\n3- write the resulting text of search in iqbal.txt", source="user")], CancellationToken()
            # [TextMessage(content="""please list the directory contents""", source="user")], CancellationToken()
        )
    )


asyncio.run(main())
