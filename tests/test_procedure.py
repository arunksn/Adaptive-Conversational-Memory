import pytest

from src.models.procedure import (
    Procedure,
    ProcedureState,
    ProcedureTransition
)


def test_create_procedure():

    procedure = Procedure(
        name="Deploy Python Application",
        description="Basic deployment workflow."
    )

    assert procedure.name == (
        "Deploy Python Application"
    )

    assert procedure.version == 1

    assert procedure.states == {}

    assert procedure.transitions == []


def test_add_state():

    procedure = Procedure(
        name="Deployment"
    )

    state = ProcedureState(
        name="Start"
    )

    procedure.add_state(state)

    assert state.state_id in procedure.states

    assert (
        procedure.states[state.state_id].name
        == "Start"
    )


def test_add_transition():

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

    transition = ProcedureTransition(
        from_state=start.state_id,
        to_state=deploy.state_id,
        action="Begin deployment"
    )

    procedure.add_transition(
        transition
    )

    assert len(
        procedure.transitions
    ) == 1


def test_get_next_states():

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

    procedure.add_transition(
        ProcedureTransition(
            from_state=start.state_id,
            to_state=deploy.state_id,
            action="Begin deployment"
        )
    )

    next_states = procedure.get_next_states(
        start.state_id
    )

    assert len(next_states) == 1

    assert next_states[0].name == "Deploy"


def test_invalid_transition_source():

    procedure = Procedure(
        name="Deployment"
    )

    state = ProcedureState(
        name="Deploy"
    )

    procedure.add_state(state)

    transition = ProcedureTransition(
        from_state="invalid",
        to_state=state.state_id,
        action="Deploy"
    )

    with pytest.raises(ValueError):

        procedure.add_transition(
            transition
        )


def test_invalid_transition_destination():

    procedure = Procedure(
        name="Deployment"
    )

    state = ProcedureState(
        name="Start"
    )

    procedure.add_state(state)

    transition = ProcedureTransition(
        from_state=state.state_id,
        to_state="invalid",
        action="Deploy"
    )

    with pytest.raises(ValueError):

        procedure.add_transition(
            transition
        )