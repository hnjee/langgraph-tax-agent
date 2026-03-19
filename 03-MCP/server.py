# @tool to MCP Server with FastMCP
# uv add mcp fastmcp
from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("my_tools")

#name → 함수 이름 (add), description → docstring (숫자 a와 b를 더합니다.), parameters → 타입 힌트 (a: int, b: int)
@mcp.tool()
def add(a: int, b: int):
    """숫자 a와 b를 더합니다."""
    return a + b


@mcp.tool()
def multiply(a: int, b:int):
    """숫자 a와 b를 곱합니다."""
    return a * b

@mcp.tool()
def web_search(query: str):
    """웹을 검색합니다."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return str(results)

#이 파일을 직접 실행할 때만 서버를 시작해라, 다른 파일에서 이 파일을 import해서 사용할 때는 서버 시작X
if __name__ == "__main__": 
    mcp.run()

# Inspector라는 테스트 도구가 JS로 만들어짐
# 따라서 실행을 위해서는 컴퓨터에 Node.js 설치 필요 -> brew install node
# npx @modelcontextprotocol/inspector python math_server.py
