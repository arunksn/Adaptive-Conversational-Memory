from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset
)


class AdaptiveMemoryBenchmarkDataset:
    """
    Deterministic benchmark dataset for evaluating
    conversational memory retrieval.

    The main benchmark contains exactly 20 memory-ID
    retrieval cases.

    Procedural graph evaluation is maintained separately
    because graph retrieval uses procedure/state IDs
    rather than memory IDs.
    """

    @staticmethod
    def build() -> EvaluationDataset:
        """
        Build the main 20-case memory retrieval benchmark.
        """

        return EvaluationDataset([

            

            EvaluationCase(
                case_id="semantic-preference-001",
                query=(
                    "What programming language "
                    "do I prefer?"
                ),
                relevant_memory_ids=[
                    "memory-python-preference"
                ]
            ),

            EvaluationCase(
                case_id="semantic-goal-001",
                query=(
                    "What is my long-term "
                    "career goal?"
                ),
                relevant_memory_ids=[
                    "memory-career-goal"
                ]
            ),

            EvaluationCase(
                case_id="semantic-tool-001",
                query=(
                    "What database do I usually "
                    "work with?"
                ),
                relevant_memory_ids=[
                    "memory-postgresql"
                ]
            ),

          

            EvaluationCase(
                case_id="episodic-event-001",
                query=(
                    "What did I work on yesterday?"
                ),
                relevant_memory_ids=[
                    "memory-yesterday-project"
                ]
            ),

            EvaluationCase(
                case_id="episodic-event-002",
                query=(
                    "Which conference did I attend "
                    "last month?"
                ),
                relevant_memory_ids=[
                    "memory-conference"
                ]
            ),

            EvaluationCase(
                case_id="episodic-event-003",
                query=(
                    "What project did I work on "
                    "during my internship?"
                ),
                relevant_memory_ids=[
                    "memory-internship-project"
                ]
            ),

        

            EvaluationCase(
                case_id="procedural-001",
                query=(
                    "How do I deploy my application?"
                ),
                relevant_memory_ids=[
                    "memory-deployment-procedure"
                ]
            ),

            EvaluationCase(
                case_id="procedural-002",
                query=(
                    "What are the steps I use "
                    "to run the project?"
                ),
                relevant_memory_ids=[
                    "memory-project-workflow"
                ]
            ),

            EvaluationCase(
                case_id="procedural-003",
                query=(
                    "How do I evaluate the "
                    "retrieval system?"
                ),
                relevant_memory_ids=[
                    "memory-environment-setup"
                ]
            ),

           

            EvaluationCase(
                case_id="temporal-001",
                query=(
                    "What programming language "
                    "am I currently using?"
                ),
                relevant_memory_ids=[
                    "memory-current-language"
                ]
            ),

            EvaluationCase(
                case_id="temporal-002",
                query=(
                    "What was my previous "
                    "programming language?"
                ),
                relevant_memory_ids=[
                    "memory-old-language"
                ]
            ),


            EvaluationCase(
                case_id="conflict-preference-001",
                query=(
                    "Which database do I currently "
                    "prefer?"
                ),
                relevant_memory_ids=[
                    "memory-current-database"
                ]
            ),

            EvaluationCase(
                case_id="conflict-preference-002",
                query=(
                    "Which framework do I currently "
                    "prefer for building backend APIs "
                    "with Go?"
                ),
                relevant_memory_ids=[
                    "memory-current-framework"
                ]
            ),

         

            EvaluationCase(
                case_id="consolidation-001",
                query=(
                    "What type of development "
                    "am I interested in?"
                ),
                relevant_memory_ids=[
                    "memory-backend-development"
                ]
            ),

            EvaluationCase(
                case_id="consolidation-002",
                query=(
                    "What technology have I repeatedly "
                    "used in my projects?"
                ),
                relevant_memory_ids=[
                    "memory-python-repeated"
                ]
            ),

           

            EvaluationCase(
                case_id="noise-001",
                query=(
                    "What programming language "
                    "do I prefer?"
                ),
                relevant_memory_ids=[
                    "memory-python-preference"
                ]
            ),

            EvaluationCase(
                case_id="noise-002",
                query=(
                    "How do I deploy my application?"
                ),
                relevant_memory_ids=[
                    "memory-deployment-procedure"
                ]
            ),

           

            EvaluationCase(
                case_id="mixed-001",
                query=(
                    "What have I been learning "
                    "recently?"
                ),
                relevant_memory_ids=[
                    "memory-current-learning"
                ]
            ),

            EvaluationCase(
                case_id="mixed-002",
                query=(
                    "What is my current project about?"
                ),
                relevant_memory_ids=[
                    "memory-current-project"
                ]
            ),

            EvaluationCase(
                case_id="mixed-003",
                query=(
                    "What are my current technical "
                    "interests?"
                ),
                relevant_memory_ids=[
                    "memory-technical-interests"
                ]
            )
        ])

    @classmethod
    def retrieval_cases(cls):
        """
        Return the 20 main memory retrieval cases.
        """

        return cls.build().retrieval_cases()

    @classmethod
    def memory_cases(cls):
        """
        Return the main memory retrieval cases.
        """

        return cls.build().memory_retrieval_cases()


    @staticmethod
    def build_procedural_cases() -> EvaluationDataset:
        """
        Build the separate procedural graph benchmark.

        These cases are intentionally NOT included in build()
        because the main benchmark evaluates memory IDs.

        Procedural graph evaluation instead evaluates:
            - procedure_id
            - current state
            - expected next state
        """

        return EvaluationDataset([

            EvaluationCase(
                case_id="procedural-graph-001",
                query=(
                    "What should I do after "
                    "starting the project evaluation?"
                ),
                relevant_procedure_ids=[
                    "procedure-project-evaluation"
                ],
                relevant_state_ids=[
                    "state-implementation"
                ],
                metadata={
                    "evaluation_type": "procedural_graph",
                    "current_state_id": (
                        "state-project-start"
                    )
                }
            ),

            EvaluationCase(
                case_id="procedural-graph-002",
                query=(
                    "What should I do after "
                    "implementing the experiment?"
                ),
                relevant_procedure_ids=[
                    "procedure-project-evaluation"
                ],
                relevant_state_ids=[
                    "state-tests"
                ],
                metadata={
                    "evaluation_type": "procedural_graph",
                    "current_state_id": (
                        "state-implementation"
                    )
                }
            ),

            EvaluationCase(
                case_id="procedural-graph-003",
                query=(
                    "What happens after the "
                    "experiment tests pass?"
                ),
                relevant_procedure_ids=[
                    "procedure-project-evaluation"
                ],
                relevant_state_ids=[
                    "state-full-tests"
                ],
                metadata={
                    "evaluation_type": "procedural_graph",
                    "current_state_id": (
                        "state-experiment-tests"
                    )
                }
            )
        ])

    @classmethod
    def procedural_cases(cls):
        """
        Return the separate procedural graph dataset.
        """

        return cls.build_procedural_cases()

    @classmethod
    def procedural_retrieval_cases(cls):
        """
        Return procedural graph cases.
        """

        return (
            cls.build_procedural_cases()
            .procedural_retrieval_cases()
        )