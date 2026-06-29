from agent_graph import graph

png = graph.get_graph().draw_mermaid_png()

with open("langgraph_graph.png", "wb") as f:
    f.write(png)

print("Creato langgraph_graph.png")
