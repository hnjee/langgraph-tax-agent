# uv add langchain-mcp-adapters

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage


load_dotenv()

llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')

async def main():
    client = MultiServerMCPClient(
        {
            "my_tools": {
                "command": "python",
                "args": ["./server.py"],
                "transport": "stdio",
            }
        }
    ) 
    # MCP Tool → BaseTool 변환
    mcp_tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(mcp_tools)
    tool_node = ToolNode(mcp_tools)

    # ===== 그래프 전체가 여기 안에 있어야 해요 =====
    def agent(state: MessagesState):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node('agent', agent)
    graph_builder.add_node('tools', tool_node)
    graph_builder.add_edge(START, 'agent')
    graph_builder.add_conditional_edges('agent', tools_condition)
    graph_builder.add_edge('tools', 'agent')
    graph = graph_builder.compile()

    # ===== 실행도 여기 안에 =====
    #query = "3이랑 5 더해줘"
    #query = "서울 혜화동의 떡볶이 맛집은?"
    query = "3 곱하기 19는?"
    async for chunk in graph.astream(
        {'messages': [HumanMessage(query)]},
        stream_mode='values'
    ):
        chunk['messages'][-1].pretty_print()

asyncio.run(main())

'''
실행: python client.py

MCP 서버를 따로 먼저 실행할 필요 없음
-> MultiServerMCPClient가 클라이언트 실행할 때 자동으로 서버를 띄워주기 때문 

agent.py 실행
    → MultiServerMCPClient가 python math_server.py 자동 실행
    → STDIO로 연결
    → Tool 가져옴
    → 에이전트 실행
    → async with 끝나면 서버도 자동 종료
'''