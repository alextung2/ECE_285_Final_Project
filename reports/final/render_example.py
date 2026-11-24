"""Draw the illustrative two-node budget comparison in the report."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), layout="constrained")
    positions = [(0, .50), (1, .80), (1, .20), (2.1, .80)]
    for axis, title, child, chosen in zip(
        axes, ["Marginal product", "Conditional product"], [".2475", ".3575"], [2, 3]
    ):
        for parent, node in [(0, 1), (0, 2), (1, 3)]:
            active = node in (1, chosen)
            x, y = zip(positions[parent], positions[node])
            axis.add_line(Line2D(x, y, color="#216478" if active else "#999999",
                                 linewidth=2, linestyle="-" if active else "--"))
        for index, ((x, y), label) in enumerate(zip(positions, ["root", ".55", ".30", child])):
            active = index in (0, 1, chosen)
            axis.text(x, y, label, ha="center", va="center", fontsize=11,
                      bbox=dict(boxstyle="round,pad=.28", facecolor="white",
                                edgecolor="#216478" if active else "#999999"))
        axis.set(xlim=(-.38, 2.56), ylim=(0, 1.02), title=title)
        axis.axis("off")
    fig.savefig(Path(__file__).parent / "tables" / "two_slots.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
