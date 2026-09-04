"""Fixture for the benchmark: a small module with a known shape."""

GREETING = "hello"
MAGIC_NUMBER = 42


def add(left: int, right: int) -> int:
    """Return the sum of two integers."""
    return left + right


def multiply(left: int, right: int) -> int:
    """Return the product of two integers."""
    return left * right


def greet(name: str) -> str:
    return f"{GREETING}, {name}!"


class Counter:
    """Counts how many times tick() was called."""

    def __init__(self) -> None:
        self.value = 0

    def tick(self) -> int:
        self.value += 1
        return self.value


if __name__ == "__main__":
    print(add(2, 2))
    print(multiply(6, 7))
    print(greet("world"))
