from mobile_convert.cli import build_parser


def test_cli_smoke():
    parser = build_parser()
    args = parser.parse_args(["quantize-int8"])
    assert args.cmd == "quantize-int8"
