from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class ProcedureState:

    name: str
    description: str = ""

    state_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    is_terminal: bool = False


@dataclass
class ProcedureTransition:

    from_state: str
    to_state: str

    action: str
    condition: Optional[str] = None

    transition_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )


@dataclass
class Procedure:

    name: str
    description: str = ""

    procedure_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    version: int = 1

    states: dict[str, ProcedureState] = field(
        default_factory=dict
    )

    transitions: list[ProcedureTransition] = field(
        default_factory=list
    )

    def add_state(
        self,
        state: ProcedureState
    ):
        self.states[state.state_id] = state

    def add_transition(
        self,
        transition: ProcedureTransition
    ):
        if transition.from_state not in self.states:
            raise ValueError(
                f"Unknown source state: "
                f"{transition.from_state}"
            )

        if transition.to_state not in self.states:
            raise ValueError(
                f"Unknown destination state: "
                f"{transition.to_state}"
            )

        self.transitions.append(
            transition
        )

    def get_next_states(
        self,
        state_id: str
    ) -> list[ProcedureState]:

        next_states = []

        for transition in self.transitions:

            if transition.from_state == state_id:

                next_state = self.states[
                    transition.to_state
                ]

                next_states.append(
                    next_state
                )

        return next_states