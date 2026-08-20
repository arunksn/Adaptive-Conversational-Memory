from scripts.run_benchmark import build_baseline, build_adaptive
from src.evaluation.benchmark_dataset import AdaptiveMemoryBenchmarkDataset

dataset = AdaptiveMemoryBenchmarkDataset.build()

baseline = build_baseline()
adaptive = build_adaptive()

print("=" * 100)
print("REMAINING DIFFERENCES")
print("=" * 100)

for case in dataset.retrieval_cases():

    baseline_result = baseline.retrieve(
        query=case.query,
        top_k=5
    )

    adaptive_result = adaptive.retrieve(
        query=case.query,
        top_k=5
    )

    if isinstance(baseline_result, tuple):
        baseline_results = baseline_result[1]
    else:
        baseline_results = baseline_result

    routing, adaptive_results = adaptive_result

    baseline_ids = [
        str(getattr(r, "memory_id", ""))
        for r in baseline_results
    ]

    adaptive_ids = [
        str(getattr(r, "memory_id", ""))
        for r in adaptive_results
    ]

    expected = set(
        str(x)
        for x in case.relevant_memory_ids
    )

    baseline_hit = bool(
        expected.intersection(baseline_ids)
    )

    adaptive_hit = bool(
        expected.intersection(adaptive_ids)
    )

    if not adaptive_hit:

        print()
        print("-" * 100)
        print("CASE:", case.case_id)
        print("QUERY:", case.query)
        print("EXPECTED:", case.relevant_memory_ids)
        print("BASELINE:", baseline_ids)
        print("ADAPTIVE:", adaptive_ids)
        print(
            "ROUTE:",
            [r.value for r in routing.routes]
        )
        print(
            "PRIMARY:",
            routing.primary_route.value
        )

print()
print("=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)

