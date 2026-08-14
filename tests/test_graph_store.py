import pytest

from src.models.procedure import (
    Procedure,
    ProcedureState,
    ProcedureTransition
)

from src.storage.graph_store import GraphStore


def create_procedure():

    procedure = Procedure(
        name="Deploy Python Application",
        description="Basic deployment workflow."
    )

    start = ProcedureState(
        name="Start"
    )

    install = ProcedureState(
        name="Install Dependencies"
    )

    test = ProcedureState(
        name="Run Tests"
    )

    deploy = ProcedureState(
        name="Deploy"
    )

    done = ProcedureState(
        name="Complete",
        is_terminal=True
    )

    procedure.add_state(start)
    procedure.add_state(install)
    procedure.add_state(test)
    procedure.add_state(deploy)
    procedure.add_state(done)

    procedure.add_transition(
        ProcedureTransition(
            from_state=start.state_id,
            to_state=install.state_id,
            action="Install dependencies"
        )
    )

    procedure.add_transition(
        ProcedureTransition(
            from_state=install.state_id,
            to_state=test.state_id,
            action="Run tests"
        )
    )

    procedure.add_transition(
        ProcedureTransition(
            from_state=test.state_id,
            to_state=deploy.state_id,
            action="Build and deploy"
        )
    )

    procedure.add_transition(
        ProcedureTransition(
            from_state=deploy.state_id,
            to_state=done.state_id,
            action="Finish deployment"
        )
    )

    return (
        procedure,
        start,
        install,
        test,
        deploy,
        done
    )


def test_add_procedure():

    store = GraphStore()

    procedure, *_ = create_procedure()

    store.add_procedure(
        procedure
    )

    assert store.count() == 1

    assert (
        store.get_procedure(
            procedure.procedure_id
        )
        == procedure
    )


def test_add_duplicate_procedure():

    store = GraphStore()

    procedure, *_ = create_procedure()

    store.add_procedure(
        procedure
    )

    with pytest.raises(ValueError):

        store.add_procedure(
            procedure
        )


def test_add_state():

    store = GraphStore()

    procedure = Procedure(
        name="Deployment"
    )

    store.add_procedure(
        procedure
    )

    state = ProcedureState(
        name="Start"
    )

    store.add_state(
        procedure.procedure_id,
        state
    )

    result = store.get_procedure(
        procedure.procedure_id
    )

    assert result is not None

    assert (
        result.states[state.state_id]
        == state
    )


def test_add_transition():

    store = GraphStore()

    procedure = Procedure(
        name="Deployment"
    )

    start = ProcedureState(
        name="Start"
    )

    deploy = ProcedureState(
        name="Deploy"
    )

    procedure.add_state(start)
    procedure.add_state(deploy)

    store.add_procedure(
        procedure
    )

    transition = ProcedureTransition(
        from_state=start.state_id,
        to_state=deploy.state_id,
        action="Deploy application"
    )

    store.add_transition(
        procedure.procedure_id,
        transition
    )

    assert len(
        procedure.transitions
    ) == 1


def test_get_next_states():

    store = GraphStore()

    (
        procedure,
        start,
        install,
        *_,
    ) = create_procedure()

    store.add_procedure(
        procedure
    )

    next_states = store.get_next_states(
        procedure.procedure_id,
        start.state_id
    )

    assert len(next_states) == 1

    assert next_states[0].state_id == (
        install.state_id
    )


def test_reachable_states():

    store = GraphStore()

    (
        procedure,
        start,
        install,
        test,
        deploy,
        done
    ) = create_procedure()

    store.add_procedure(
        procedure
    )

    reachable = store.get_reachable_states(
        procedure.procedure_id,
        start.state_id
    )

    reachable_ids = {
        state.state_id
        for state in reachable
    }

    assert reachable_ids == {
        start.state_id,
        install.state_id,
        test.state_id,
        deploy.state_id,
        done.state_id
    }


def test_invalid_procedure():

    store = GraphStore()

    with pytest.raises(ValueError):

        store.add_state(
            "invalid-procedure",
            ProcedureState(
                name="Start"
            )
        )


def test_invalid_state():

    store = GraphStore()

    procedure = Procedure(
        name="Deployment"
    )

    store.add_procedure(
        procedure
    )

    with pytest.raises(ValueError):

        store.get_next_states(
            procedure.procedure_id,
            "invalid-state"
        )


def test_update_procedure():

    store = GraphStore()

    procedure, *_ = create_procedure()

    store.add_procedure(
        procedure
    )

    procedure.version = 2
    procedure.description = (
        "Updated deployment workflow."
    )

    store.update_procedure(
        procedure
    )

    result = store.get_procedure(
        procedure.procedure_id
    )

    assert result.version == 2

    assert result.description == (
        "Updated deployment workflow."
    )


def test_delete_procedure():

    store = GraphStore()

    procedure, *_ = create_procedure()

    store.add_procedure(
        procedure
    )

    assert store.count() == 1

    store.delete_procedure(
        procedure.procedure_id
    )

    assert store.count() == 0

    assert (
        store.get_procedure(
            procedure.procedure_id
        )
        is None
    )