from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoalpha.config import ResearchConfig
from autoalpha.data.current_panel import inspect_current_panel
from autoalpha.service.factor_research import run_factor_research


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoAlpha institutional research platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprint = subparsers.add_parser(
        "fingerprint", help="print the immutable fingerprint for a research configuration"
    )
    fingerprint.add_argument(
        "--config", type=Path, default=Path("config/research.toml"), help="research TOML file"
    )
    fingerprint.add_argument(
        "--checksum",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="include a named data checksum; repeat as needed",
    )
    inspect_data = subparsers.add_parser(
        "inspect-data", help="report price-research and institutional PIT data readiness"
    )
    inspect_data.add_argument("path", type=Path, help="partitioned parquet panel directory")
    factor_research = subparsers.add_parser(
        "factor-research", help="run the bounded single-researcher factor discovery MVP"
    )
    factor_research.add_argument("research_direction", help="research direction for the Researcher")
    factor_research.add_argument("--candidate-count", type=int, default=4)
    factor_research.add_argument("--rounds", type=int, default=3)
    factor_research.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="A 股日线 parquet 面板目录；未提供时读取 AUTOALPHA_DATA_PATH 或服务设置",
    )
    factor_research.add_argument("--output-dir", type=Path, default=None)
    return parser


def _checksums(values: list[str]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for value in values:
        name, separator, checksum = value.partition("=")
        if not separator or not name or not checksum:
            raise ValueError(f"Invalid checksum declaration: {value!r}; expected NAME=SHA256")
        checksums[name] = checksum
    return checksums


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fingerprint":
        config = ResearchConfig.from_toml(args.config)
        payload = {
            "name": config.name,
            "generation": config.generation,
            "fingerprint": config.fingerprint(data_checksums=_checksums(args.checksum)),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "inspect-data":
        report = inspect_current_panel(args.path)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "factor-research":
        summary = run_factor_research(
            args.research_direction,
            candidate_count=args.candidate_count,
            rounds=args.rounds,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
