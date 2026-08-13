from autoalpha.cli import _parser


def test_cli_exposes_bounded_factor_research_command() -> None:
    args = _parser().parse_args(["factor-research", "短期反转", "--candidate-count", "4"])

    assert args.command == "factor-research"
    assert args.research_direction == "短期反转"
    assert args.candidate_count == 4
    assert args.rounds == 3
