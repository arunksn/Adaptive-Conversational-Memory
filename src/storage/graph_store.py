from src.models.procedure import (
    Procedure,
    ProcedureState,
    ProcedureTransition
)


class GraphStore:

    def __init__(self):
        self.procedures: dict[str, Procedure] = {}

    # ADD PROCEDURE

    def add_procedure(
        self,
        procedure: Procedure
    ):
        """
        Store a complete procedure graph.
        """

        if procedure.procedure_id in self.procedures:
            raise ValueError(
                f"Procedure already exists: "
                f"{procedure.procedure_id}"
            )

        self.procedures[
            procedure.procedure_id
        ] = procedure

    # GET PROCEDURE

    def get_procedure(
        self,
        procedure_id: str
    ):
        """
        Retrieve a procedure by ID.
        """

        return self.procedures.get(
            procedure_id
        )

    # UPDATE PROCEDURE

    def update_procedure(
        self,
        procedure: Procedure
    ):
        """
        Replace an existing procedure.
        """

        if procedure.procedure_id not in self.procedures:
            raise ValueError(
                f"Procedure not found: "
                f"{procedure.procedure_id}"
            )

        self.procedures[
            procedure.procedure_id
        ] = procedure


    # DELETE PROCEDURE

    def delete_procedure(
        self,
        procedure_id: str
    ):
        """
        Delete a procedure.
        """

        if procedure_id not in self.procedures:
            raise ValueError(
                f"Procedure not found: "
                f"{procedure_id}"
            )

        del self.procedures[
            procedure_id
        ]

    # ADD STATE

    def add_state(
        self,
        procedure_id: str,
        state: ProcedureState
    ):
        """
        Add a state to an existing procedure.
        """

        procedure = self._require_procedure(
            procedure_id
        )

        if state.state_id in procedure.states:
            raise ValueError(
                f"State already exists: "
                f"{state.state_id}"
            )

        procedure.add_state(
            state
        )

    # ADD TRANSITION

    def add_transition(
        self,
        procedure_id: str,
        transition: ProcedureTransition
    ):
        """
        Add a transition to an existing procedure.
        """

        procedure = self._require_procedure(
            procedure_id
        )

        procedure.add_transition(
            transition
        )

    # NEXT STATES

    def get_next_states(
        self,
        procedure_id: str,
        state_id: str
    ) -> list[ProcedureState]:
        """
        Return states directly reachable from
        the specified state.
        """

        procedure = self._require_procedure(
            procedure_id
        )

        if state_id not in procedure.states:
            raise ValueError(
                f"State not found: {state_id}"
            )

        return procedure.get_next_states(
            state_id
        )

    # REACHABLE STATES

    def get_reachable_states(
        self,
        procedure_id: str,
        start_state_id: str
    ) -> list[ProcedureState]:
        """
        Traverse the procedure graph and return
        all states reachable from the starting state.
        """

        procedure = self._require_procedure(
            procedure_id
        )

        if start_state_id not in procedure.states:
            raise ValueError(
                f"State not found: "
                f"{start_state_id}"
            )

        visited: set[str] = set()
        queue: list[str] = [
            start_state_id
        ]

        reachable = []

        while queue:

            current_state_id = queue.pop(0)

            if current_state_id in visited:
                continue

            visited.add(
                current_state_id
            )

            current_state = procedure.states[
                current_state_id
            ]

            reachable.append(
                current_state
            )

            for next_state in procedure.get_next_states(
                current_state_id
            ):

                if next_state.state_id not in visited:
                    queue.append(
                        next_state.state_id
                    )

        return reachable

    # PROCEDURE COUNT

    def count(self) -> int:
        return len(
            self.procedures
        )

    # INTERNAL

    def _require_procedure(
        self,
        procedure_id: str
    ) -> Procedure:

        procedure = self.get_procedure(
            procedure_id
        )

        if procedure is None:
            raise ValueError(
                f"Procedure not found: "
                f"{procedure_id}"
            )

        return procedure