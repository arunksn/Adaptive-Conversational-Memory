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

    # TEMPORAL SIGNALS

    TEMPORAL_KEYWORDS = {
        "yesterday",
        "today",
        "tomorrow",
        "last",
        "recent",
        "recently",
        "ago",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "date",
        "time",
        "earlier",
        "before",
        "after",
        "previous",
        "previously",
        "history",
        "happened",
        "told",
        "said",
        "did",
    }

    # PROCEDURAL SIGNALS

    # Avoid generic grammatical words such as "do".
    #
    # These keywords represent actual procedural concepts
    # or actions.

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

    # SEMANTIC SIGNALS

    # These are strong semantic-memory indicators.
    #
    # Generic question words such as:
    # what, who, where, why, how
    #
    # are intentionally excluded.

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
    }

    # ROUTING

    def route(
        self,
        query: str
    ) -> RoutingResult:
        """
        Analyze a query and determine which memory
        source or sources should be used.

        This is the deterministic routing baseline.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        normalized_query = query.lower()

        tokens = self._tokenize(
            query
        )

        scores = {
            MemoryRoute.SEMANTIC: 0,
            MemoryRoute.EPISODIC: 0,
            MemoryRoute.PROCEDURAL: 0,
        }


        temporal_matches = (
            tokens
            & self.TEMPORAL_KEYWORDS
        )

        scores[
            MemoryRoute.EPISODIC
        ] += len(
            temporal_matches
        ) * 2

        # PROCEDURAL SIGNALS

        procedural_matches = (
            tokens
            & self.PROCEDURAL_KEYWORDS
        )

        scores[
            MemoryRoute.PROCEDURAL
        ] += len(
            procedural_matches
        ) * 2


        semantic_matches = (
            tokens
            & self.SEMANTIC_KEYWORDS
        )

        scores[
            MemoryRoute.SEMANTIC
        ] += len(
            semantic_matches
        )


        temporal_phrases = [
            "last month",
            "last week",
            "last year",
            "yesterday",
            "recently",
            "a few days ago",
            "earlier",
            "previous conversation",
            "previously",
        ]

        for phrase in temporal_phrases:

            if phrase in normalized_query:

                scores[
                    MemoryRoute.EPISODIC
                ] += 3


        procedural_phrases = [
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
        ]

        for phrase in procedural_phrases:

            if phrase in normalized_query:

                scores[
                    MemoryRoute.PROCEDURAL
                ] += 3

        # NO STRONG SIGNAL

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
                    "No strong temporal or procedural "
                    "signal was detected; using semantic "
                    "memory as the default route."
                )
            )


        ranked_routes = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        primary_route = ranked_routes[0][0]
        primary_score = ranked_routes[0][1]

        selected_routes = [
            primary_route
        ]

        # SECONDARY MEMORY SOURCE

        if len(ranked_routes) > 1:

            second_route = ranked_routes[1][0]
            second_score = ranked_routes[1][1]

            # A secondary route is only included when
            # it has a meaningful signal and is at least
            # half as strong as the primary intent.
            #
            # This allows queries such as:
            #
            # "What did I previously do when deploying
            #  my project?"
            #
            # to use both episodic and procedural memory.
            #
            # At the same time, weak grammatical signals
            # cannot create unnecessary routes.

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
            temporal_matches,
            procedural_matches,
            semantic_matches,
            selected_routes
        )

        return RoutingResult(
            routes=selected_routes,
            confidence=confidence,
            reason=reason
        )

    # TOKENIZATION

    @staticmethod
    def _tokenize(
        query: str
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
        )

        return set(
            cleaned.split()
        )

    # ROUTING EXPLANATION

    @staticmethod
    def _build_reason(
        temporal_matches: set[str],
        procedural_matches: set[str],
        semantic_matches: set[str],
        selected_routes: list[MemoryRoute]
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