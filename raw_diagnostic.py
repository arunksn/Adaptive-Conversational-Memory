from scripts.run_benchmark import build_adaptive

adaptive = build_adaptive()

queries = [
    "What type of development do I frequently work on?",
    "What technology is part of my current project?",
]

for query in queries:
    print("\n" + "=" * 100)
    print("QUERY:", query)
    print("=" * 100)

    results = (
        adaptive
        .hybrid_retriever
        .vector_retriever
        .search(
            query=query,
            top_k=21
        )
    )

    for rank, result in enumerate(results, 1):
        memory = result.get("memory")

        print(
            f"{rank:2}. "
            f"{result.get('memory_id')} "
            f"| score={float(result.get('score', 0.0)):.4f} "
            f"| content={getattr(memory, 'content', '')}"
        )
