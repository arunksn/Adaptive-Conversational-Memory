import json
from pathlib import Path

from src.retrieval.graph_retriever import GraphRetriever
from src.storage.graph_store import GraphStore

from scripts.populate_graph_store import (
    build_procedure,
)


PROCEDURES_PATH = Path(
    "data/memories/procedures.json"
)


def build_retriever():

    with open(
        PROCEDURES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        procedure_data = json.load(file)

    store = GraphStore()

    for data in procedure_data:

        procedure = build_procedure(
            data
        )

        store.add_procedure(
            procedure
        )

    return GraphRetriever(
        store
    )


def test_retriever_initializes():

    retriever = build_retriever()

    assert retriever is not None


def test_project_evaluation_procedure_is_retrievable():

    retriever = build_retriever()

    procedure = retriever.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    assert (
        procedure.name
        == "Project Evaluation Workflow"
    )


def test_memory_retrieval_procedure_is_retrievable():

    retriever = build_retriever()

    procedure = retriever.get_procedure(
        "procedure-memory-retrieval"
    )

    assert procedure is not None

    assert (
        procedure.name
        == "Adaptive Memory Retrieval Workflow"
    )


def test_unknown_procedure_returns_none():

    retriever = build_retriever()

    procedure = retriever.get_procedure(
        "procedure-does-not-exist"
    )

    assert procedure is None


def test_get_next_states_from_project_workflow():

    retriever = build_retriever()

    procedure = retriever.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    first_state = next(
        iter(procedure.states.values())
    )

    next_states = retriever.get_next_states(
        procedure.procedure_id,
        first_state.state_id,
    )

    assert isinstance(
        next_states,
        list,
    )


def test_reachable_states_from_project_workflow():

    retriever = build_retriever()

    procedure = retriever.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    first_state = next(
        iter(procedure.states.values())
    )

    reachable = retriever.get_reachable_states(
        procedure.procedure_id,
        first_state.state_id,
    )

    assert isinstance(
        reachable,
        list,
    )


def test_reachable_states_follow_workflow():

    retriever = build_retriever()

    procedure = retriever.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    states = list(
        procedure.states.values()
    )

    assert len(states) == 6

    start_state = states[0]

    reachable = retriever.get_reachable_states(
        procedure.procedure_id,
        start_state.state_id,
    )

    assert len(reachable) > 0


def test_next_state_lookup_preserves_state_identity():

    retriever = build_retriever()

    procedure = retriever.get_procedure(
        "procedure-memory-retrieval"
    )

    assert procedure is not None

    first_state = next(
        iter(procedure.states.values())
    )

    next_states = retriever.get_next_states(
        procedure.procedure_id,
        first_state.state_id,
    )

    for state in next_states:

        assert state.state_id in (
            procedure.states
        )


def test_terminal_states_are_identifiable():

    retriever = build_retriever()

    terminal_states = []

    for procedure_id in [
        "procedure-project-evaluation",
        "procedure-memory-retrieval",
    ]:

        procedure = retriever.get_procedure(
            procedure_id
        )

        assert procedure is not None

        for state in procedure.states.values():

            if state.is_terminal:
                terminal_states.append(
                    state
                )

    assert len(
        terminal_states
    ) >= 2


def test_workflow_transitions_are_preserved():

    retriever = build_retriever()

    total_transitions = 0

    for procedure_id in [
        "procedure-project-evaluation",
        "procedure-memory-retrieval",
    ]:

        procedure = retriever.get_procedure(
            procedure_id
        )

        assert procedure is not None

        total_transitions += len(
            procedure.transitions
        )

        for transition in procedure.transitions:

            assert transition.action

            assert (
                transition.from_state
                in procedure.states
            )

            assert (
                transition.to_state
                in procedure.states
            )

    assert total_transitions == 10