"""Tiny local fallback implementing the subset pandas.to_markdown requires."""

__version__ = "0.9.0"


def tabulate(tabular_data, headers=(), tablefmt="pipe", showindex=False, **_kwargs):
    # pandas calls tabulate with ``headers="keys"``. The old fallback used
    # ``list(headers)`` unconditionally, turning that sentinel into the four
    # columns k/e/y/s and corrupting generated Markdown tables.
    if headers == "keys":
        if hasattr(tabular_data, "keys"):
            header_row = list(tabular_data.keys())
            rows = [list(row) for row in tabular_data.values]
        else:
            rows = [list(row) for row in tabular_data]
            header_row = list(range(len(rows[0]))) if rows else []
    else:
        rows = [list(row) for row in tabular_data]
        header_row = list(headers)
    if showindex:
        header_row = [""] + header_row
        rows = [[idx] + row for idx, row in enumerate(rows)]

    def render(row):
        return "| " + " | ".join(str(value) for value in row) + " |"

    lines = [render(header_row), render(["---"] * len(header_row))]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)
