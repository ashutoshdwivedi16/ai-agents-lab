"""CLI for querying metrics.

Usage:
    python -m shared.metrics summary
    python -m shared.metrics summary --agent simple-chatbot
    python -m shared.metrics records --limit 20
"""

import argparse

from shared.config import load_app_config
from shared.metrics import init_metrics, get_backend


def main():
    parser = argparse.ArgumentParser(
        prog="python -m shared.metrics",
        description="Query LLM call metrics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- summary subcommand ---
    summary_parser = subparsers.add_parser("summary", help="Show aggregated metrics")
    summary_parser.add_argument("--session", default=None, help="Filter by session ID")
    summary_parser.add_argument("--agent", default=None, help="Filter by agent name")

    # --- records subcommand ---
    records_parser = subparsers.add_parser("records", help="Show raw metric records")
    records_parser.add_argument("--session", default=None, help="Filter by session ID")
    records_parser.add_argument("--agent", default=None, help="Filter by agent name")
    records_parser.add_argument(
        "--limit", type=int, default=20, help="Max records to show (default: 20)"
    )

    args = parser.parse_args()

    # Initialize metrics from config
    app_config = load_app_config()
    init_metrics(app_config.metrics)
    backend = get_backend()

    if args.command == "summary":
        summary = backend.summary(session_id=args.session, agent_name=args.agent)
        if summary.total_calls == 0:
            print("No metrics recorded yet.")
            return

        print("=== LLM Metrics Summary ===")
        print(f"  Total calls:       {summary.total_calls}")
        print(f"  Input tokens:      {summary.total_input_tokens:,}")
        print(f"  Output tokens:     {summary.total_output_tokens:,}")
        print(f"  Total cost:        ${summary.total_cost:.6f}")
        print(f"  Avg latency:       {summary.avg_latency_ms:.1f}ms")

        if summary.by_provider:
            print("\n  By provider:")
            for prov, cnt in sorted(summary.by_provider.items()):
                print(f"    {prov}: {cnt} calls")

        if summary.by_model:
            print("\n  By model:")
            for model, cnt in sorted(summary.by_model.items()):
                print(f"    {model}: {cnt} calls")

        if summary.by_agent:
            print("\n  By agent:")
            for agent, cnt in sorted(summary.by_agent.items()):
                print(f"    {agent}: {cnt} calls")

    elif args.command == "records":
        recs = backend.records(
            session_id=args.session, agent_name=args.agent, limit=args.limit
        )
        if not recs:
            print("No records found.")
            return

        print(
            f"{'Timestamp':<22} {'Provider':<10} {'Model':<28} "
            f"{'Tokens':>10} {'Cost':>10} {'Latency':>10}"
        )
        print("-" * 92)
        for r in recs:
            tokens = f"{r.input_tokens}+{r.output_tokens}"
            print(
                f"{r.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<22} "
                f"{r.provider:<10} "
                f"{r.model:<28} "
                f"{tokens:>10} "
                f"${r.cost:<9.6f} "
                f"{r.latency_ms:>8.1f}ms"
            )


if __name__ == "__main__":
    main()
