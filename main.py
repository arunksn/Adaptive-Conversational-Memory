from src.pipeline.memory_pipeline import MemoryPipeline


def main():

    pipeline = MemoryPipeline()

    conversations = [
        "I prefer Python for machine learning projects.",
        "Yesterday I attended an AI workshop.",
        "First install Python, then install PyTorch.",
        "Okay",
        "My goal is to build a conversational AI system.",
    ]

    for text in conversations:

        memory = pipeline.process(text)

        print("=" * 60)
        print(f"Content     : {memory.content}")
        print(f"Memory ID   : {memory.memory_id}")
        print(f"Type        : {memory.memory_type.value}")
        print(f"Importance  : {memory.importance}")
        print(f"Confidence  : {memory.confidence}")
        print(f"Status      : {memory.status.value}")
        print(f"Timestamp   : {memory.timestamp}")
        print()


if __name__ == "__main__":
    main()