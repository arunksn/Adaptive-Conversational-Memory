from dataclasses import dataclass
from enum import Enum


class MemoryRoute(Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


@dataclass
class RoutingResult:
    routes: list[MemoryRoute]
    confidence: float
    reason: str

    @property
    def primary_route(self) -> MemoryRoute:
        """
        Return the highest-priority route.
        """

        return self.routes[0]


class MemoryRouter:


    TEMPORAL_KEYWORDS = {
        "yesterday",
        "today",
        "tomorrow",
        "ago",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "date",
        "time",
        "happened",
        "did",
    }

    # These indicate historical/current temporal state only
    # when combined with an appropriate query.
    #
    # "previous programming language" should remain semantic,
    # because it asks about a stored preference/fact rather
    # than an event.

    TEMPORAL_EVENT_KEYWORDS = {
        "worked",
        "work",
        "attended",
        "attend",
        "implemented",
        "did",
        "happened",
        "event",
        "activity",
    }


    PROCEDURAL_KEYWORDS = {
        "steps",
        "step",
        "procedure",
        "process",
        "workflow",
        "guide",
        "instructions",
        "deploy",
        "deploying",
        "deployed",
        "deployment",
        "install",
        "installing",
        "installed",
        "installation",
        "configure",
        "configuring",
        "configured",
        "configuration",
        "setup",
        "build",
        "building",
        "built",
        "execute",
        "executing",
        "execution",
        "perform",
        "performing",
        "run",
        "running",
    }


    SEMANTIC_KEYWORDS = {
        "prefer",
        "preference",
        "preferences",
        "like",
        "likes",
        "favorite",
        "favourite",
        "usually",
        "always",
        "fact",
        "facts",
        "information",
        "knowledge",
        "know",
        "believe",
        "interest",
        "interests",
        "learning",
        "learn",
        "technology",
        "technologies",
        "language",
        "database",
        "framework",
        "project",
        "development",
        "backend",
        "current",
        "latest",
        "previous",
        "previously",
        "old",
        "former",
        "long-term",
    }


    TEMPORAL_PHRASES = (
        "yesterday",
        "today",
        "tomorrow",
        "last month",
        "last week",
        "last year",
        "recently",
        "a few days ago",
        "earlier",
        "previous conversation",
        "previously",
    )


    PROCEDURAL_PHRASES = (
        "how do i",
        "how can i",
        "how to",
        "steps to",
        "steps for",
        "guide me",
        "instructions for",
        "what should i do",
        "when deploying",
        "when installing",
        "when configuring",
        "how do i deploy",
        "how do i configure",
        "how do i run",
        "steps i use",
    )


    def route(
        self,
        query: str,
    ) -> RoutingResult:
        """
        Analyze a query and determine which memory
        source or sources should be used.

        Routing is deterministic and based on explicit
        lexical and phrase-level signals.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        normalized_query = query.lower().strip()

        tokens = self._tokenize(
            query
        )

        scores = {
            MemoryRoute.SEMANTIC: 0,
            MemoryRoute.EPISODIC: 0,
            MemoryRoute.PROCEDURAL: 0,
        }


        semantic_matches = (
            tokens
            & self.SEMANTIC_KEYWORDS
        )

        scores[
            MemoryRoute.SEMANTIC
        ] += len(
            semantic_matches
        )


        procedural_matches = (
            tokens
            & self.PROCEDURAL_KEYWORDS
        )

        scores[
            MemoryRoute.PROCEDURAL
        ] += len(
            procedural_matches
        ) * 2

        # Strong procedural phrases.
        for phrase in self.PROCEDURAL_PHRASES:

            if phrase in normalized_query:

                scores[
                    MemoryRoute.PROCEDURAL
                ] += 4


        temporal_matches = (
            tokens
            & self.TEMPORAL_KEYWORDS
        )

        temporal_event_matches = (
            tokens
            & self.TEMPORAL_EVENT_KEYWORDS
        )

        # Temporal words by themselves are not always enough.
        #
        # Example:
        #
        # "What was my previous programming language?"
        #
        # contains "previous", but this is a semantic
        # historical preference, not an episodic event.

        explicit_event_signal = (
            bool(
                temporal_event_matches
            )
        )

        explicit_temporal_signal = (
            bool(
                temporal_matches
            )
        )

        for phrase in self.TEMPORAL_PHRASES:

            if phrase in normalized_query:

                # "previously" and "earlier" are treated
                # as temporal only when the query also
                # contains an event/action signal.
                if phrase in {
                    "previously",
                    "earlier",
                }:
                    if explicit_event_signal:
                        scores[
                            MemoryRoute.EPISODIC
                        ] += 3

                else:
                    scores[
                        MemoryRoute.EPISODIC
                    ] += 3

        # Explicit event language strongly indicates episodic
        # memory when accompanied by temporal context.

        if (
            explicit_temporal_signal
            and explicit_event_signal
        ):

            scores[
                MemoryRoute.EPISODIC
            ] += len(
                temporal_matches
            ) * 2

        # Direct event questions such as:
        #
        # "What did I work on yesterday?"
        #
        # should clearly prefer episodic memory.

        if (
            "what did i" in normalized_query
            and explicit_temporal_signal
        ):

            scores[
                MemoryRoute.EPISODIC
            ] += 4

        # SPECIAL CASES

        # Historical/current facts and preferences should remain
        # semantic even when words such as "previous", "current",
        # "latest", or "last" appear.
        #
        # Examples:
        #
        # "What was my previous programming language?"
        # "What is my latest preferred backend framework?"
        # "What database do I currently prefer?"

        historical_fact_phrases = (
            "previous programming language",
            "previous language",
            "old programming language",
            "former programming language",
            "latest preferred",
            "current preferred",
            "currently prefer",
            "currently using",
            "currently use",
            "current project",
            "current learning",
            "technical interests",
        )

        for phrase in historical_fact_phrases:

            if phrase in normalized_query:

                scores[
                    MemoryRoute.SEMANTIC
                ] += 5

                # Prevent accidental episodic routing.
                scores[
                    MemoryRoute.EPISODIC
                ] = 0


        max_score = max(
            scores.values()
        )

        if max_score == 0:

            return RoutingResult(
                routes=[
                    MemoryRoute.SEMANTIC
                ],
                confidence=0.5,
                reason=(
                    "No strong temporal, procedural, or "
                    "semantic signal was detected; using "
                    "semantic memory as the default route."
                ),
            )


        ranked_routes = sorted(
            scores.items(),
            key=lambda item: (
                item[1],
                self._route_priority(
                    item[0]
                ),
            ),
            reverse=True,
        )

        primary_route = ranked_routes[0][0]
        primary_score = ranked_routes[0][1]

        selected_routes = [
            primary_route
        ]


        if len(ranked_routes) > 1:

            second_route = ranked_routes[1][0]
            second_score = ranked_routes[1][1]

            if (
                second_score > 0
                and second_score >= primary_score * 0.50
            ):

                selected_routes.append(
                    second_route
                )

        confidence = min(
            1.0,
            primary_score / 6.0
        )

        reason = self._build_reason(
            temporal_matches=temporal_matches,
            procedural_matches=procedural_matches,
            semantic_matches=semantic_matches,
            selected_routes=selected_routes,
        )

        return RoutingResult(
            routes=selected_routes,
            confidence=confidence,
            reason=reason,
        )


    @staticmethod
    def _route_priority(
        route: MemoryRoute,
    ) -> int:
        """
        Deterministic tie-breaking priority.

        Procedural > Episodic > Semantic
        """

        priorities = {
            MemoryRoute.PROCEDURAL: 3,
            MemoryRoute.EPISODIC: 2,
            MemoryRoute.SEMANTIC: 1,
        }

        return priorities[
            route
        ]


    @staticmethod
    def _tokenize(
        query: str,
    ) -> set[str]:
        """
        Convert query text into normalized tokens.
        """

        cleaned = (
            query.lower()
            .replace("?", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace("!", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("(", " ")
            .replace(")", " ")
        )

        return set(
            cleaned.split()
        )


    @staticmethod
    def _build_reason(
        temporal_matches: set[str],
        procedural_matches: set[str],
        semantic_matches: set[str],
        selected_routes: list[MemoryRoute],
    ) -> str:
        """
        Build a human-readable explanation of the
        routing decision.
        """

        reasons = []

        if temporal_matches:

            reasons.append(
                "temporal signals: "
                + ", ".join(
                    sorted(
                        temporal_matches
                    )
                )
            )

        if procedural_matches:

            reasons.append(
                "procedural signals: "
                + ", ".join(
                    sorted(
                        procedural_matches
                    )
                )
            )

        if semantic_matches:

            reasons.append(
                "semantic signals: "
                + ", ".join(
                    sorted(
                        semantic_matches
                    )
                )
            )

        route_names = ", ".join(
            route.value
            for route in selected_routes
        )

        if reasons:

            return (
                f"Selected {route_names} route(s) "
                f"based on "
                + "; ".join(reasons)
                + "."
            )

        return (
            f"Selected {route_names} route(s)."
        )