import json
from pathlib import Path

from src.models.procedure import (
    Procedure,
    ProcedureState,
    ProcedureTransition,
)
from src.storage.graph_store import GraphStore


PROCEDURES_PATH = Path(
    "data/memories/procedures.json"
)


def load_procedures(
    path: Path
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Procedures file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "procedures.json must contain a list"
        )

    return data


def build_procedure(
    data: dict
) -> Procedure:

    procedure = Procedure(
        name=data["name"],
        description=data.get(
            "description",
            ""
        ),
        procedure_id=data[
            "procedure_id"
        ],
        version=data.get(
            "version",
            1
        ),
    )

    # STATES

    for state_data in data.get(
        "states",
        []
    ):

        state = ProcedureState(
            name=state_data["name"],
            description=state_data.get(
                "description",
                ""
            ),
            state_id=state_data[
                "state_id"
            ],
            is_terminal=state_data.get(
                "is_terminal",
                False
            ),
        )

        procedure.add_state(
            state
        )

    # TRANSITIONS

    for transition_data in data.get(
        "transitions",
        []
    ):

        transition = ProcedureTransition(
            from_state=transition_data[
                "from_state"
            ],
            to_state=transition_data[
                "to_state"
            ],
            action=transition_data[
                "action"
            ],
            condition=transition_data.get(
                "condition"
            ),
            transition_id=transition_data[
                "transition_id"
            ],
        )

        procedure.add_transition(
            transition
        )

    return procedure


def populate_graph_store(
    procedures_path: Path = PROCEDURES_PATH,
) -> GraphStore:

    print("Loading procedures...")

    procedure_data = load_procedures(
        procedures_path
    )

    print(
        f"Loaded {len(procedure_data)} procedures."
    )

    store = GraphStore()

    print(
        "Populating knowledge graph..."
    )

    for index, data in enumerate(
        procedure_data,
        start=1
    ):

        procedure = build_procedure(
            data
        )

        store.add_procedure(
            procedure
        )

        print(
            f"[{index}/{len(procedure_data)}] "
            f"{procedure.procedure_id} - "
            f"{procedure.name}"
        )

        print(
            f"    States: "
            f"{len(procedure.states)}"
        )

        print(
            f"    Transitions: "
            f"{len(procedure.transitions)}"
        )

    return store


def validate_graph_store(
    store: GraphStore
):

    print()
    print(
        "Validating knowledge graph..."
    )

    assert store.count() > 0

    total_states = 0
    total_transitions = 0

    for procedure in store.procedures.values():

        total_states += len(
            procedure.states
        )

        total_transitions += len(
            procedure.transitions
        )

        state_ids = set(
            procedure.states.keys()
        )

        for transition in (
            procedure.transitions
        ):

            if transition.from_state not in state_ids:
                raise ValueError(
                    "Invalid transition source: "
                    f"{transition.transition_id}"
                )

            if transition.to_state not in state_ids:
                raise ValueError(
                    "Invalid transition destination: "
                    f"{transition.transition_id}"
                )

    print(
        f"Procedures: {store.count()}"
    )

    print(
        f"States: {total_states}"
    )

    print(
        f"Transitions: {total_transitions}"
    )

    print(
        "Knowledge graph validation passed."
    )


def main():

    store = populate_graph_store()

    validate_graph_store(
        store
    )

    print()
    print(
        "Knowledge graph populated successfully."
    )


if __name__ == "__main__":
    main()