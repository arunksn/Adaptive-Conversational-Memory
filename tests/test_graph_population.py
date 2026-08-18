import json
from pathlib import Path

from src.storage.graph_store import GraphStore


PROCEDURES_PATH = Path(
    "data/memories/procedures.json"
)


def load_procedure_data():
    with open(
        PROCEDURES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_store():
    data = load_procedure_data()

    store = GraphStore()

    from scripts.populate_graph_store import (
        build_procedure,
    )

    for procedure_data in data:

        procedure = build_procedure(
            procedure_data
        )

        store.add_procedure(
            procedure
        )

    return store


def test_graph_store_contains_all_procedures():

    store = build_store()

    assert store.count() == 2

    assert (
        "procedure-project-evaluation"
        in store.procedures
    )

    assert (
        "procedure-memory-retrieval"
        in store.procedures
    )


def test_project_evaluation_procedure_contains_expected_states():

    store = build_store()

    procedure = store.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    assert len(
        procedure.states
    ) == 6

    assert len(
        procedure.transitions
    ) == 5


def test_memory_retrieval_procedure_contains_expected_states():

    store = build_store()

    procedure = store.get_procedure(
        "procedure-memory-retrieval"
    )

    assert procedure is not None

    assert len(
        procedure.states
    ) == 6

    assert len(
        procedure.transitions
    ) == 5


def test_all_transition_endpoints_are_valid():

    store = build_store()

    for procedure in store.procedures.values():

        state_ids = set(
            procedure.states.keys()
        )

        for transition in procedure.transitions:

            assert (
                transition.from_state
                in state_ids
            )

            assert (
                transition.to_state
                in state_ids
            )


def test_next_states_can_be_retrieved():

    store = build_store()

    procedure = store.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    first_state = next(
        iter(procedure.states.values())
    )

    next_states = store.get_next_states(
        procedure.procedure_id,
        first_state.state_id,
    )

    assert isinstance(
        next_states,
        list,
    )


def test_transition_actions_can_be_retrieved_from_transitions():

    store = build_store()

    procedure = store.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    first_state = next(
        iter(procedure.states.values())
    )

    transitions = [
        transition
        for transition in procedure.transitions
        if transition.from_state
        == first_state.state_id
    ]

    assert isinstance(
        transitions,
        list,
    )

    for transition in transitions:

        assert transition.action

        assert isinstance(
            transition.action,
            str,
        )


def test_reachable_states_can_be_traversed():

    store = build_store()

    procedure = store.get_procedure(
        "procedure-project-evaluation"
    )

    assert procedure is not None

    first_state = next(
        iter(procedure.states.values())
    )

    reachable = store.get_reachable_states(
        procedure.procedure_id,
        first_state.state_id,
    )

    assert isinstance(
        reachable,
        list,
    )

    assert first_state.state_id not in reachable


def test_terminal_states_are_preserved():

    store = build_store()

    terminal_count = 0

    for procedure in store.procedures.values():

        for state in procedure.states.values():

            if state.is_terminal:

                terminal_count += 1

    assert terminal_count >= 2


def test_procedure_state_names_are_preserved():

    store = build_store()

    for procedure in store.procedures.values():

        for state in procedure.states.values():

            assert state.name

            assert isinstance(
                state.name,
                str,
            )


def test_transition_actions_are_preserved():

    store = build_store()

    transition_count = 0

    for procedure in store.procedures.values():

        for transition in procedure.transitions:

            assert transition.action

            assert isinstance(
                transition.action,
                str,
            )

            transition_count += 1

    assert transition_count == 10