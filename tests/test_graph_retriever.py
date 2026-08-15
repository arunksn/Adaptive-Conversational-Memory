from src.models.procedure import (
    Procedure,
    ProcedureState,
    ProcedureTransition
)
from src.retrieval.graph_retriever import GraphRetriever


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


def test_add_and_get_procedure():

    retriever = GraphRetriever()

    procedure, *_ = create_procedure()

    retriever.add_procedure(
        procedure
    )

    result = retriever.get_procedure(
        procedure.procedure_id
    )

    assert result == procedure


def test_get_state():

    retriever = GraphRetriever()

    (
        procedure,
        start,
        *_,
    ) = create_procedure()

    retriever.add_procedure(
        procedure
    )

    result = retriever.get_state(
        procedure.procedure_id,
        start.state_id
    )

    assert result.name == "Start"


def test_get_next_states():

    retriever = GraphRetriever()

    (
        procedure,
        start,
        install,
        *_,
    ) = create_procedure()

    retriever.add_procedure(
        procedure
    )

    next_states = retriever.get_next_states(
        procedure.procedure_id,
        start.state_id
    )

    assert len(next_states) == 1

    assert next_states[0].name == (
        "Install Dependencies"
    )


def test_get_next_actions():

    retriever = GraphRetriever()

    (
        procedure,
        start,
        install,
        *_,
    ) = create_procedure()

    retriever.add_procedure(
        procedure
    )

    actions = retriever.get_next_actions(
        procedure.procedure_id,
        start.state_id
    )

    assert len(actions) == 1

    assert actions[0]["action"] == (
        "Install dependencies"
    )

    assert actions[0]["next_state"].name == (
        "Install Dependencies"
    )


def test_reachable_states():

    retriever = GraphRetriever()

    (
        procedure,
        start,
        install,
        test,
        deploy,
        done
    ) = create_procedure()

    retriever.add_procedure(
        procedure
    )

    states = retriever.get_reachable_states(
        procedure.procedure_id,
        start.state_id
    )

    state_names = [
        state.name
        for state in states
    ]

    assert state_names == [
        "Start",
        "Install Dependencies",
        "Run Tests",
        "Deploy",
        "Complete"
    ]


def test_find_state():

    retriever = GraphRetriever()

    procedure, *_ = create_procedure()

    retriever.add_procedure(
        procedure
    )

    result = retriever.find_state(
        procedure.procedure_id,
        "run tests"
    )

    assert result is not None

    assert result.name == "Run Tests"


def test_find_missing_state():

    retriever = GraphRetriever()

    procedure, *_ = create_procedure()

    retriever.add_procedure(
        procedure
    )

    result = retriever.find_state(
        procedure.procedure_id,
        "Unknown State"
    )

    assert result is None


def test_terminal_states():

    retriever = GraphRetriever()

    (
        procedure,
        start,
        install,
        test,
        deploy,
        done
    ) = create_procedure()

    retriever.add_procedure(
        procedure
    )

    terminal_states = (
        retriever.get_terminal_states(
            procedure.procedure_id
        )
    )

    assert len(terminal_states) == 1

    assert terminal_states[0].name == (
        "Complete"
    )


def test_invalid_procedure():

    retriever = GraphRetriever()

    procedure, *_ = create_procedure()

    retriever.add_procedure(
        procedure
    )

    try:
        retriever.get_state(
            "invalid-procedure",
            "invalid-state"
        )
        assert False
    except ValueError:
        assert True


def test_invalid_state():

    retriever = GraphRetriever()

    procedure, *_ = create_procedure()

    retriever.add_procedure(
        procedure
    )

    try:
        retriever.get_state(
            procedure.procedure_id,
            "invalid-state"
        )
        assert False
    except ValueError:
        assert True