import logging
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()

logging.getLogger("langfuse").setLevel(logging.ERROR)
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langfuse.langchain import CallbackHandler
from chains_ollama import generate_chain, reflect_chain


langfuse_handler = CallbackHandler()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state: MessageGraph):
    return {
        "messages": [
            generate_chain.invoke({
                "messages": state["messages"]
            }, config={"callbacks": [langfuse_handler]})
        ]
    }


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke(
        {
            "messages": state["messages"]
        }
    )
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph):
    """Condition for LLM to continue or not."""

    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder = StateGraph(state_schema=MessageGraph)
# Add nodes
builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# Add conditions for nodes
builder.add_conditional_edges(
    GENERATE,
    should_continue,
    path_map={END:END, REFLECT:REFLECT},
)
# Add edges
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("Hello Langgraph!")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs, config={"callbacks": [langfuse_handler]})
    print(response)
