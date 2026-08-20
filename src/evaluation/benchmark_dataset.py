from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset
)


class AdaptiveMemoryBenchmarkDataset:
    """
    Deterministic benchmark dataset for evaluating
    adaptive conversational memory retrieval.

    The main benchmark evaluates memory-ID retrieval.

    Procedural graph retrieval is represented separately
    through procedural ground truth while preserving the
    original procedural case IDs for compatibility with
    the existing test suite.
    """

    @staticmethod
    def build() -> EvaluationDataset:
        """
        Build the main memory-retrieval benchmark.

        Exactly 20 cases are returned.

        Procedural graph cases are represented using
        procedural ground truth and are handled separately
        by procedural_cases().
        """

        return EvaluationDataset([

            # SEMANTIC MEMORY
            # resolved case 1 issue: memory retrieval should return the most relevant memory based on semantic similarity, not just the most recent one. 

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

        
            # PROCEDURAL MEMORY
            #
            # These are retained as memory-ID benchmark cases
            # for compatibility with the existing benchmark.
            #
            # The actual procedure-graph behavior is evaluated
            # separately through procedural_cases().
         

            EvaluationCase(
                case_id="procedural-001",
                query=(
                    "How do I deploy my application?"
                ),
                relevant_memory_ids=[
                    "memory-deployment-procedure"
                ],
                metadata={
                    "evaluation_type": "procedural_memory"
                }
            ),

            EvaluationCase(
                case_id="procedural-002",
                query=(
                    "What are the steps I use "
                    "to run the project?"
                ),
                relevant_memory_ids=[
                    "memory-project-workflow"
                ],
                metadata={
                    "evaluation_type": "procedural_memory"
                }
            ),

            EvaluationCase(
                case_id="procedural-003",
                query=(
                    "How do I evaluate the retrieval system?"
                ),
                relevant_memory_ids=[
                    "memory-environment-setup"
                ],
                metadata={
                    "evaluation_type": "procedural_memory"
                }
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
                    "What is my latest preferred "
                    "backend framework?"
                ),
                relevant_memory_ids=[
                    "memory-current-framework"
                ]
            ),


            EvaluationCase(
                case_id="consolidation-001",
                query=(
                    "What type of development "
                    "do I frequently work on?"
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
                    "What technology is part of "
                    "my current project?"
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
            ),
        ])


    @classmethod
    def retrieval_cases(
        cls
    ) -> list[EvaluationCase]:
        """
        Return the main memory-retrieval benchmark cases.

        All cases contain memory-ID ground truth.
        """

        return cls.build().memory_retrieval_cases()


    @staticmethod
    def procedural_cases() -> EvaluationDataset:
        """
        Build the separate procedural graph benchmark.

        Procedural graph retrieval returns ProcedureState
        objects rather than Memory objects.

        Therefore these cases use:

            procedure_id
            state_id
            relevant_state_ids

        as their ground truth.

        The original procedural case IDs are preserved so
        existing tests and project references remain valid.
        """

        return EvaluationDataset([

            EvaluationCase(
                case_id="procedural-001",
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
                    "evaluation_type": "procedural_graph"
                }
            ),

            EvaluationCase(
                case_id="procedural-002",
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
                    "evaluation_type": "procedural_graph"
                }
            ),

            EvaluationCase(
                case_id="procedural-003",
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
                    "evaluation_type": "procedural_graph"
                }
            ),
        ])

   

    @classmethod
    def procedural_retrieval_cases(
        cls
    ) -> list[EvaluationCase]:
        """
        Return procedural graph evaluation cases.
        """

        return (
            cls.procedural_cases()
            .procedural_retrieval_cases()
        )