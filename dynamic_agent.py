import requests
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI


def fetch_tools():
    return requests.get("http://localhost:8000/tools").json()

tools_data = fetch_tools()
tools = []

for t in tools_data:
    def make_func(endpoint):
        return lambda x: requests.post(endpoint, json=x).json()
    tools.append(Tool(
        name=t["name"],
        func=make_func(t["endpoint"]),
        description=t["description"]
    ))

llm = ChatOpenAI()
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

# Example usage
query = {
    "symbol": "AAPL",
    "side": "buy",
    "qty": 1,
    "type": "market"
}
print(agent.run(f"Place a market order to buy 1 share of AAPL."))