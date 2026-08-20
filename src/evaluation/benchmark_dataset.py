from src.evaluation.evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset
)


class AdaptiveMemoryBenchmarkDataset:
    """
    Deterministic benchmark dataset for evaluating
    conversational memory retrieval.

    The dataset covers:

    - semantic memories
    - episodic memories
    - procedural memories
    - temporal information
    - preference changes
    - repeated information
    - conflicting information
    - irrelevant memories
    """

    @staticmethod
    def build() -> EvaluationDataset:
        """
        Build the complete memory retrieval benchmark dataset.

        These cases evaluate memory-ID retrieval and are
        intentionally kept separate from procedural graph
        evaluation.
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
                    "How do I configure the "
                    "development environment?"
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
            )
        ])


    @classmethod
    def procedural_cases(
        cls
    ) -> EvaluationDataset:
        """
        Build deterministic procedural-retrieval cases.

        Procedural retrieval is evaluated separately from
        ordinary memory retrieval because procedural results
        come from the procedure/state graph rather than
        vector memories.
        """

        return EvaluationDataset([


            EvaluationCase(
                case_id="procedural-graph-001",
                query=(
                    "What should I do after starting "
                    "the project evaluation?"
                ),
                relevant_procedure_ids=[
                    "procedure-project-evaluation"
                ],
                relevant_state_ids=[
                    "state-implementation"
                ],
                procedure_id=(
                    "procedure-project-evaluation"
                ),
                state_id=(
                    "state-project-start"
                ),
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
                procedure_id=(
                    "procedure-project-evaluation"
                ),
                state_id=(
                    "state-implementation"
                ),
            ),

         

            EvaluationCase(
                case_id="procedural-graph-003",
                query=(
                    "What happens after the experiment "
                    "tests pass?"
                ),
                relevant_procedure_ids=[
                    "procedure-project-evaluation"
                ],
                relevant_state_ids=[
                    "state-full-tests"
                ],
                procedure_id=(
                    "procedure-project-evaluation"
                ),
                state_id=(
                    "state-experiment-tests"
                ),
            ),
        ])

  

    @classmethod
    def retrieval_cases(
        cls
    ) -> list[EvaluationCase]:
        """
        Return the existing memory retrieval benchmark cases.

        Procedural graph cases are intentionally excluded.
        """

        return (
            cls.build()
            .retrieval_cases()
        )