from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np


class Node:
    def __init__(self, name, parent: Node | None, stage: int, values: dict[str, Any]):
        self.name = name
        self.stage = stage

        self.parent = parent
        self.children = []

        self.values = values

    def add_child(self, child: Node):
        self.children.append(child)

    def __repr__(self):
        # return f"Node({self.name}, {self.parent.name if self.parent else None}, {self.stage}, {self.values})"
        return f"Node('{self.name}')"

    def __getitem__(self, item: int):
        if not self.children:
            raise IndexError(f"Node has no children (tried to access {item}")
        if item > len(self.children):
            raise IndexError(f"Node has {len(self.children)} children, tried to access {item}")

        return self.children[item]

    def __hash__(self):
        return self.name.__hash__()

    def detail_str(self):
        return f"Node({self.name}, {self.parent.name if self.parent else None}, {self.stage}, {self.values})"


class RandomStrategy(Enum):
    RANDOM_WALK = auto()
    UNIFORM = auto()


class RandVar:
    def __init__(self, name, strategy, args: dict):
        self.name = name
        self.strategy = strategy
        self.args = args


def random_walk_tree_builder(
    variables: list[str],
    start_val: list[float],
    walk_std: list[float],
    walk_max: list[float],
    stages: int,
    branch_factor: int,
    seed=None,
    truncate_places: int | None = None,
):
    rng = np.random.default_rng(seed)

    root = Node("root", None, 0, {})

    for i, var in enumerate(variables):
        root.values[var] = start_val[i]

    all_nodes = [root]
    current_nodes = [root]

    for stage in range(1, stages):
        next_nodes = []

        for node in current_nodes:
            for branch in range(branch_factor):
                child = Node(f"{node.name}_{branch}", node, stage, {})

                for var_idx, var in enumerate(variables):
                    walk_amt = rng.normal(0, walk_std[var_idx])
                    walk_amt = np.clip(walk_amt, -walk_max[var_idx], walk_max[var_idx])
                    if truncate_places is not None:
                        walk_amt = np.round(walk_amt, truncate_places)
                    new_val = node.values[var] + walk_amt

                    child.values[var] = new_val

                # print(f"Adding child ({child.name}) to parent={node.name}")
                node.add_child(child)
                # print(f"Now has children: {[c.name for c in node.children]}")
                next_nodes.append(child)
                all_nodes.append(child)

        current_nodes = next_nodes

    # print(f"Root node: {root} has children {root.children}")
    return root, all_nodes


if __name__ == "__main__":
    root, all_nodes = random_walk_tree_builder(
        ["x", "y"], [100, 50], [10, 5], [30, 20], 5, 3, seed=42, truncate_places=0
    )

    for node in all_nodes:
        print(node)

    print("trying to access children:")
    print(root[0][1])
