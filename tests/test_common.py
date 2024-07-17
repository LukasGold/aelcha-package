from aelcha.common import parse_cycles_to_plot


def test_parse_cycles_to_plot():
    assert parse_cycles_to_plot("1,2,3,4") == [1, 2, 3, 4]
    assert parse_cycles_to_plot("1-3,5") == [1, 2, 3, 5]
    assert parse_cycles_to_plot("1-3;5") == [1, 2, 3, 5]
    assert parse_cycles_to_plot("1-3,5-7,9") == [1, 2, 3, 5, 6, 7, 9]
    assert parse_cycles_to_plot("1-3;5-7;9") == [1, 2, 3, 5, 6, 7, 9]
    assert parse_cycles_to_plot("0, 1-3,5-7;9-11") == [0, 1, 2, 3, 5, 6, 7, 9, 10, 11]
