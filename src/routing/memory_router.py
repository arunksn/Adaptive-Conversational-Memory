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
        "last",
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
        "history",
        "happened",
        "told",
        "said",
        "did",
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
        "goal",
        "career",
        "current",
        "currently",
        "latest",
        "technology",
        "technologies",
        "language",
        "database",
        "framework",
        "learning",
        "learn",
        "project",
        "development",
        "backend",
    }

    

    CURRENT_SEMANTIC_PHRASES = {
        "currently",
        "current",
        "latest",
        "right now",
        "at the moment",
        "my current",
        "my latest",
        "currently using",
        "currently prefer",
        "currently learning",
        "current project",
        "current interests",
        "current technical interests",
        "current programming language",
        "current database",
        "latest preferred",
    }


    EPISODIC_PHRASES = {
        "yesterday",
        "last week",
        "last month",
        "last year",
        "a few days ago",
        "previous conversation",
        "previously",
        "earlier",
        "what did i",
        "what happened",
        "what was i doing",
        "what have i done",
    }



    PROCEDURAL_PHRASES = {
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
    }



    def route(
        self,
        query: str
    ) -> RoutingResult:
        """
        Analyze a query and determine which memory
        source or sources should be used.

        The router uses deterministic lexical signals.

        Important routing behavior:

        - Strong episodic queries are routed to episodic memory.
        - Strong procedural queries are routed to procedural memory.
        - Current/factual/preference queries favor semantic memory.
        - Ambiguous queries may use multiple memory sources.
        - Temporal words such as "recently" do not automatically
          force episodic retrieval when the query asks for a
          current semantic fact.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        normalized_query = (
            query.lower().strip()
        )

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

        procedural_matches = (
            tokens
            & self.PROCEDURAL_KEYWORDS
        )

        semantic_matches = (
            tokens
            & self.SEMANTIC_KEYWORDS
        )

        # Temporal signals receive stronger weight because
        # explicit temporal queries usually require episodic
        # memory.
        scores[
            MemoryRoute.EPISODIC
        ] += len(
            temporal_matches
        ) * 2

        # Procedural signals.
        scores[
            MemoryRoute.PROCEDURAL
        ] += len(
            procedural_matches
        ) * 2

        # Semantic signals.
        scores[
            MemoryRoute.SEMANTIC
        ] += len(
            semantic_matches
        )

     

        episodic_phrase_matches = []

        for phrase in self.EPISODIC_PHRASES:

            if phrase in normalized_query:

                episodic_phrase_matches.append(
                    phrase
                )

                scores[
                    MemoryRoute.EPISODIC
                ] += 3



        procedural_phrase_matches = []

        for phrase in self.PROCEDURAL_PHRASES:

            if phrase in normalized_query:

                procedural_phrase_matches.append(
                    phrase
                )

                scores[
                    MemoryRoute.PROCEDURAL
                ] += 3


        current_semantic_matches = []

        for phrase in self.CURRENT_SEMANTIC_PHRASES:

            if phrase in normalized_query:

                current_semantic_matches.append(
                    phrase
                )

                # Current facts/preferences are semantic
                # memories even when the query contains a
                # weak temporal word such as "recently".
                scores[
                    MemoryRoute.SEMANTIC
                ] += 3

       

        # Example:
        #
        # "What have I been learning recently?"
        #
        # "recently" alone should not force episodic
        # retrieval because the expected information may
        # represent a current semantic fact.
        #
        # If the query contains "recently" together with
        # semantic concepts such as learning, interests,
        # technology, project, preference, or development,
        # give semantic memory priority.

        recent_semantic_terms = {
            "learning",
            "learn",
            "interests",
            "interest",
            "technology",
            "technologies",
            "project",
            "development",
            "backend",
            "preference",
            "prefer",
            "currently",
            "current",
        }

        if (
            "recently" in tokens
            and tokens
            & recent_semantic_terms
        ):

            scores[
                MemoryRoute.SEMANTIC
            ] += 4

   

        explicit_current_signal = (
            bool(current_semantic_matches)
            or bool(
                tokens
                & {
                    "current",
                    "currently",
                    "latest",
                }
            )
        )

        if explicit_current_signal:

            # Current semantic information should normally
            # be retrieved from semantic memory.
            scores[
                MemoryRoute.SEMANTIC
            ] += 2

            # Weak temporal signals should not dominate a
            # current-information query.
            if (
                not episodic_phrase_matches
                and not (
                    "yesterday" in tokens
                    or "previous" in tokens
                    or "previously" in tokens
                )
            ):

                scores[
                    MemoryRoute.EPISODIC
                ] = min(
                    scores[
                        MemoryRoute.EPISODIC
                    ],
                    scores[
                        MemoryRoute.SEMANTIC
                    ] - 1
                )


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


        if len(ranked_routes) > 1:

            second_route = ranked_routes[1][0]
            second_score = ranked_routes[1][1]

            # Include a secondary source when its signal is
            # meaningful and reasonably close to the primary
            # signal.
            #
            # This supports mixed queries such as:
            #
            # "What did I previously do when deploying
            #  my project?"
            #
            # which may benefit from both episodic and
            # procedural memory.

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
            episodic_phrase_matches=episodic_phrase_matches,
            procedural_phrase_matches=procedural_phrase_matches,
            current_semantic_matches=current_semantic_matches,
        )

        return RoutingResult(
            routes=selected_routes,
            confidence=confidence,
            reason=reason
        )



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
            .replace("(", " ")
            .replace(")", " ")
            .replace("-", " ")
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
        episodic_phrase_matches: list[str] | None = None,
        procedural_phrase_matches: list[str] | None = None,
        current_semantic_matches: list[str] | None = None,
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

        if episodic_phrase_matches:

            reasons.append(
                "episodic phrases: "
                + ", ".join(
                    sorted(
                        episodic_phrase_matches
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

        if procedural_phrase_matches:

            reasons.append(
                "procedural phrases: "
                + ", ".join(
                    sorted(
                        procedural_phrase_matches
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

        if current_semantic_matches:

            reasons.append(
                "current semantic signals: "
                + ", ".join(
                    sorted(
                        current_semantic_matches
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
                + "; ".join(
                    reasons
                )
                + "."
            )

        return (
            f"Selected {route_names} route(s)."
        )