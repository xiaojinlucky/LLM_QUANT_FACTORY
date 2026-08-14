import pytest

from autoalpha.cli import _parser


def test_cli_exposes_bounded_factor_research_command() -> None:
    args = _parser().parse_args(
        [
            "factor-research",
            "短期反转",
            "--candidates-per-round",
            "4",
            "--data-path",
            "D:/data/ashare",
        ]
    )

    assert args.command == "factor-research"
    assert args.research_direction == "短期反转"
    assert args.candidates_per_round == 4
    assert args.rounds == 3
    assert str(args.data_path).replace("\\", "/") == "D:/data/ashare"


def test_cli_does_not_expose_candidate_count_alias() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["factor-research", "短期反转", "--candidate-count", "4"]
        )
