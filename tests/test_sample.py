import pytest


def test_hello_world() -> None:
    """Simple sanity check to verify pytest works."""
    greeting = "Hello, world!"
    assert greeting == "Hello, world!"

@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (-1, 5, 4),
    (0, 0, 0),
])
def test_addition(a: int, b: int, expected: int) -> None:
    """Example of a parameterized test."""
    assert a + b == expected