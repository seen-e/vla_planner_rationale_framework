"""Generic CLI for independent A/B/C stage queues."""
from planner_rationale import batch
from scheduling.episode_pipeline import EpisodePipeline
from scheduling.streaming_api import StreamingClient


def main():
    batch.Pipeline = EpisodePipeline
    batch.OpenAICompatibleClient = StreamingClient
    batch.main()


if __name__ == '__main__':
    main()
