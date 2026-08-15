from src.models.procedure import (
    Procedure,
    ProcedureState
)
from src.storage.graph_store import GraphStore


class GraphRetriever:

    def __init__(
        self,
        graph_store: GraphStore | None = None
    ):
        """
        Create a procedural memory retriever.

        A GraphStore can be injected for testing or
        for sharing the same procedural memory store.
        """

        self.graph_store = (
            graph_store
            if graph_store is not None
            else GraphStore()
        )

    # PROCEDURE MANAGEMENT

    def add_procedure(
        self,
        procedure: Procedure
    ):
        """
        Add a procedure to procedural memory.
        """

        self.graph_store.add_procedure(
            procedure
        )

    def get_procedure(
        self,
        procedure_id: str
    ) -> Procedure | None:
        """
        Retrieve a procedure by ID.
        """

        return self.graph_store.get_procedure(
            procedure_id
        )

    # STATE RETRIEVAL

    def get_state(
        self,
        procedure_id: str,
        state_id: str
    ) -> ProcedureState:
        """
        Retrieve a specific state from a procedure.
        """

        procedure = self.graph_store.get_procedure(
            procedure_id
        )

        if procedure is None:
            raise ValueError(
                f"Procedure not found: {procedure_id}"
            )

        if state_id not in procedure.states:
            raise ValueError(
                f"State not found: {state_id}"
            )

        return procedure.states[state_id]

    # NEXT STATE

    def get_next_states(
        self,
        procedure_id: str,
        state_id: str
    ) -> list[ProcedureState]:
        """
        Retrieve the states directly reachable from
        the current state.
        """

        return self.graph_store.get_next_states(
            procedure_id,
            state_id
        )
    
    # NEXT ACTIONS

    def get_next_actions(
        self,
        procedure_id: str,
        state_id: str
    ) -> list[dict]:
        """
        Retrieve the actions that can be performed
        from the current state.

        Returns:
            [
                {
                    "action": "...",
                    "condition": "...",
                    "next_state": ProcedureState
                }
            ]
        """

        procedure = self.graph_store.get_procedure(
            procedure_id
        )

        if procedure is None:
            raise ValueError(
                f"Procedure not found: {procedure_id}"
            )

        if state_id not in procedure.states:
            raise ValueError(
                f"State not found: {state_id}"
            )

        actions = []

        for transition in procedure.transitions:

            if transition.from_state != state_id:
                continue

            next_state = procedure.states[
                transition.to_state
            ]

            actions.append(
                {
                    "action": transition.action,
                    "condition": transition.condition,
                    "next_state": next_state
                }
            )

        return actions

    # GRAPH TRAVERSAL

    def get_reachable_states(
        self,
        procedure_id: str,
        start_state_id: str
    ) -> list[ProcedureState]:
        """
        Retrieve all states reachable from a
        starting state.
        """

        return self.graph_store.get_reachable_states(
            procedure_id,
            start_state_id
        )

    # FIND STATE BY NAME

    def find_state(
        self,
        procedure_id: str,
        state_name: str
    ) -> ProcedureState | None:
        """
        Find a state using its human-readable name.

        Matching is case-insensitive.
        """

        procedure = self.graph_store.get_procedure(
            procedure_id
        )

        if procedure is None:
            raise ValueError(
                f"Procedure not found: {procedure_id}"
            )

        normalized_name = state_name.strip().lower()

        for state in procedure.states.values():

            if state.name.strip().lower() == normalized_name:
                return state

        return None

    # TERMINAL STATES

    def get_terminal_states(
        self,
        procedure_id: str
    ) -> list[ProcedureState]:
        """
        Return all terminal states in a procedure.
        """

        procedure = self.graph_store.get_procedure(
            procedure_id
        )

        if procedure is None:
            raise ValueError(
                f"Procedure not found: {procedure_id}"
            )

        return [
            state
            for state in procedure.states.values()
            if state.is_terminal
        ]