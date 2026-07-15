from typing import TypedDict, Annotated
from pydantic import BaseModel
import operator
import os

Groq_API_KEY = os.environ.get("GROQ_API_KEY")
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send


# -----------------------------
# 1. Data Models
# -----------------------------

class Task(BaseModel):
    id: int
    title: str
    description: str


class Plan(BaseModel):
    blog_title: str
    tasks: list[Task]


# -----------------------------
# 2. State Schema
# -----------------------------

class StateSchema(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[list, operator.add]
    final: str



# Worker ko bheje jane wale payload ki state
class WorkerState(TypedDict):
    task: Task
    topic: str
    plan: Plan



# -----------------------------
# 3. LLM Setup
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


planner_llm = llm.with_structured_output(Plan)

# -----------------------------
# 4. Orchestrator / Planner Node
# -----------------------------

def orchestrator_node(state: StateSchema):

    plan = planner_llm.invoke(
        [
            SystemMessage(
                content="""
                You are a blog planning assistant.
                Create a structured blog plan with 5-7 sections.
                Each section should have:
                - id
                - title
                - description
                """
            ),
            HumanMessage(
                content=state["topic"]
            )
        ]
    )

    return {
        "plan": plan
    }



# -----------------------------
# 5. Fan-out Function
# -----------------------------

def fan_out(state: StateSchema):

    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "plan": state["plan"]
            }
        )
        for task in state["plan"].tasks
    ]



# -----------------------------
# 6. Worker Node
# -----------------------------

def worker_node(state: WorkerState):

    task = state["task"]
    topic = state["topic"]
    plan = state["plan"]


    response = llm.invoke(
        [
            SystemMessage(
                content="Write one clean markdown section."
            ),

            HumanMessage(
                content=f"""
                Write one blog section.

                Overall Topic:
                {topic}


                Section Title:
                {task.title}


                Section Description:
                {task.description}


                Full Blog Plan:
                {plan}
                """
            )
        ]
    )


    return {
        "sections": [
            response.content
        ]
    }



# -----------------------------
# 7. Reducer Node
# -----------------------------

def reducer_node(state: StateSchema):

    body = "\n\n".join(
        state["sections"]
    )


    final_blog = (
        f"# {state['plan'].blog_title}\n\n"
        f"{body}"
    )


    # Save markdown file
    with open(
        "blog.md",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(final_blog)


    return {
        "final": final_blog
    }



# -----------------------------
# 8. Build Graph
# -----------------------------

graph = StateGraph(StateSchema)



graph.add_node(
    "orchestrator",
    orchestrator_node
)


graph.add_node(
    "worker",
    worker_node
)


graph.add_node(
    "reducer",
    reducer_node
)



# START -> Planner

graph.add_edge(
    START,
    "orchestrator"
)



# Planner -> Fan-out -> Workers

graph.add_conditional_edges(
    "orchestrator",
    fan_out
)



# Workers -> Reducer

graph.add_edge(
    "worker",
    "reducer"
)



# Reducer -> END

graph.add_edge(
    "reducer",
    END
)



# Compile

app = graph.compile()



# -----------------------------
# 9. Run
# -----------------------------

result = app.invoke(
    {
        "topic": "Scope of Software Engineering",
        "sections": [],
        "final": ""
    }
)


print(result["final"])