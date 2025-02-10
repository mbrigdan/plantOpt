from collections import defaultdict

import cvxpy as cp
import numpy as np

from plant_opt.scenario_tree.tree import Node, random_walk_tree_builder


def build_non_recourse_var(
    var_dict: dict, domain_const_list: list, root: Node, length=None, name=None
):
    nodes_to_process = [root]

    while nodes_to_process:
        node = nodes_to_process.pop(0)
        var_name = f"{node.name}_{name}" if name else None
        new_var = cp.Variable(length, name=var_name) if length else cp.Variable(name=var_name)

        var_dict[node] = new_var
        domain_const_list.append(new_var >= 0)

        for child in node.children:
            if child.children:
                # Non-recourse variables are not added at terminal nodes, so only add if the node has children
                nodes_to_process.append(child)


def build_recourse_var(var_dict: dict, domain_const_list: list, root: Node, length=None, name=None):
    # Don't add recourse variables at the root node, only process the children
    nodes_to_process = root.children.copy()

    while nodes_to_process:
        node = nodes_to_process.pop(0)
        var_name = f"{node.name}_{name}" if name else None
        new_var = (
            cp.Variable(length, name=var_name)
            if (length and length != 1)
            else cp.Variable(name=var_name)
        )

        var_dict[node] = new_var
        domain_const_list.append(new_var >= 0)

        for child in node.children:
            nodes_to_process.append(child)


def calculate_node_objective(m, node):
    node_val = 0
    for output in m.outputs:
        node_val += m.prod_full_price[node][output] * node.values[f"prod_price_{output}"]
    node_val -= m.light_crude_import[node.parent] * node.values["crude_light_price"]
    node_val -= m.heavy_crude_import[node.parent] * node.values["crude_heavy_price"]

    return node_val


def calculate_node_objective_recursive(m, node):
    if node.parent is None:
        # Don't calculate objective value at the root node, as the values at its children will use the import costs from it already
        return 0

    node_val = calculate_node_objective(m, node)

    return node_val + calculate_node_objective_recursive(m, node.parent)


class Plant:
    def __init__(
        self,
        crude_distil_cap,  # Capacity of the distillation unit (crude oil input)
        products: int,  # Number of products
        crude_ratios: list[list[float]],
        refine_caps: list[float],
        product_ratios: np.array,  # indexed by outputs then intermediates
        stages: int,
        scenario_tree_root: Node,
        scenario_tree_all_nodes: list[Node],
        allowed_output_change: float,
        cvar: (
            tuple[float, float] | None
        ) = None,  # beta, weight for CVaR objective term (applied to each node value)
        chance: (
            tuple[float, float] | None
        ) = None,  # cutoff value, portion for chance constraint, only portion of outcomes are allowed to be below the cutoff
    ):
        self.stages = list(range(0, stages))
        self.nodes = scenario_tree_all_nodes

        self.products = list(range(0, products))
        self.outputs = list(range(0, products))

        # Define controllable variables
        # Imports - everything goes to distillation

        self.variable_domain_constraints = []

        self.light_crude_import = {}
        self.heavy_crude_import = {}

        build_non_recourse_var(
            self.light_crude_import,
            self.variable_domain_constraints,
            scenario_tree_root,
            None,
            "light_crude_import",
        )
        build_non_recourse_var(
            self.heavy_crude_import,
            self.variable_domain_constraints,
            scenario_tree_root,
            None,
            "heavy_crude_import",
        )

        # Quantities of intermediates
        self.intermediates = {}
        build_non_recourse_var(
            self.intermediates,
            self.variable_domain_constraints,
            scenario_tree_root,
            products,
            "intermediates",
        )

        # What gets sent to the refining units (intermediate products are the inputs)
        self.intermediate_to_unit = {}
        build_recourse_var(
            self.intermediate_to_unit,
            self.variable_domain_constraints,
            scenario_tree_root,
            (products, products),
            "intermediate_to_unit",
        )

        # Outputs
        self.product_output = {}

        build_recourse_var(
            self.product_output,
            self.variable_domain_constraints,
            scenario_tree_root,
            products,
            "products",
        )

        # Production of intermediates
        self.distil_const = {}
        for node in self.nodes:
            if not node.children:
                # Skip terminal nodes (non-recourse var)
                continue

            self.distil_const[node] = self.intermediates[node] == cp.hstack(
                [
                    self.light_crude_import[node] * crude_ratios[prod][0]
                    + self.heavy_crude_import[node] * crude_ratios[prod][1]
                    for prod in self.products
                ]
            )

        # "distil" capacity constraints
        self.distil_cap_const = {}
        for node in self.nodes:
            if not node.children:
                # Skip terminal nodes (non-recourse var)
                continue
            self.distil_cap_const[node] = (
                self.light_crude_import[node] + self.heavy_crude_import[node] <= crude_distil_cap
            )

        # Production of outputs
        self.product_out_const = {}

        for node in self.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue
            # Matrix multiplication makes this much easier than the pyomo version!
            self.product_out_const[node] = self.product_output[node] == cp.diag(
                self.intermediate_to_unit[node] @ product_ratios.T
            )

        self.refine_cap_const = {}
        for node in self.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            # Matrix multiplication with a vector of ones = summing each row of the intermediate_to_unit matrix
            self.refine_cap_const[node] = (
                self.intermediate_to_unit[node] @ np.ones(len(self.outputs)) <= refine_caps
            )

        # Intermediates must exist to be sent to refining units
        # 100% of intermediates must be sent to refining units - can be changed to allow for non-use
        # by making the constraint <= instead of ==
        self.intermediates_rule = {}
        for node in self.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            # Multiplying a column vector of ones by the intermediate_to_unit matrix sums each column
            self.intermediates_rule[node] = (
                self.intermediate_to_unit[node].T @ np.ones(len(self.outputs))
                == self.intermediates[node.parent]
            )
            # print(self.intermediates_rule[node])

        # Break down outputs so we can calculate payment only up to demand
        self.prod_full_price = {}
        build_recourse_var(
            self.prod_full_price,
            self.variable_domain_constraints,
            scenario_tree_root,
            products,
            "prod_full_price",
        )

        self.prod_excess = {}
        build_recourse_var(
            self.prod_excess,
            self.variable_domain_constraints,
            scenario_tree_root,
            products,
            "prod_excess",
        )

        self.breakdown_const = {}
        for node in self.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            self.breakdown_const[node] = (
                self.product_output[node] == self.prod_full_price[node] + self.prod_excess[node]
            )

        self.demand_const = {}
        for node in self.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            self.demand_const[node] = self.prod_full_price[node] <= cp.hstack(
                [node.values[f"demand_{output}"] for output in self.outputs]
            )
            # print(self.demand_const[node])

        self.interstage_const_upper = {}
        self.interstage_const_lower = {}

        for node in self.nodes:
            if not node.parent or not node.parent.parent:
                # Skip root node (recourse var) and first level
                continue

            self.interstage_const_upper[node] = (
                self.product_output[node]
                <= self.product_output[node.parent] + allowed_output_change
            )
            self.interstage_const_lower[node] = (
                self.product_output[node]
                >= self.product_output[node.parent] - allowed_output_change
            )

        node_value_list = []
        unscaled_node_value_list = []

        stage_node_count = defaultdict(int)
        for node in scenario_tree_all_nodes:
            stage_node_count[node.stage] += 1

        for node in self.nodes:
            if not node.parent:
                # Skip root node
                continue

            # Important: need to multiply the value of nodes by 1/(num nodes in stage) to get the correct value
            # (implicit assumption of equal probability of each node in the stage)
            node_val = 0
            for output in self.outputs:
                node_val += self.prod_full_price[node][output] * node.values[f"prod_price_{output}"]
            node_val -= self.light_crude_import[node.parent] * node.values["crude_light_price"]
            node_val -= self.heavy_crude_import[node.parent] * node.values["crude_heavy_price"]

            node_value_list.append(node_val / stage_node_count[node.stage])
            unscaled_node_value_list.append(node_val)

        terminal_nodes = [node for node in scenario_tree_all_nodes if node.stage == stages - 1]
        self.terminal_values = {}
        for node in terminal_nodes:
            self.terminal_values[node] = calculate_node_objective_recursive(self, node)

        if cvar:
            cvar_term = cp.cvar(-cp.hstack(self.terminal_values.values()), cvar[0]) * cvar[1]
        else:
            cvar_term = 0

        if chance:
            # Node values - offset
            n = len(self.terminal_values.values())
            self.b = cp.Variable(n, boolean=True)
            self.offset_mask = cp.Variable(n, nonneg=True)

            values = cp.hstack(self.terminal_values.values())
            self.chance_consts = [
                values >= self.offset_mask,
                self.offset_mask >= chance[0] * self.b,
                sum(self.b) >= n * chance[1],
            ]
        else:
            self.chance_consts = None

        self.cvar_term = cvar_term
        self.obj = cp.Maximize(sum(node_value_list) - cvar_term)

    def get_problem(self):
        constraints = [
            *self.variable_domain_constraints,
            # *[const for const_list in self.distil_const.values() for const in const_list],
            *self.distil_const.values(),
            *self.distil_cap_const.values(),
            *self.product_out_const.values(),
            *self.refine_cap_const.values(),
            *self.intermediates_rule.values(),
            *self.breakdown_const.values(),
            *self.demand_const.values(),
            *self.interstage_const_upper.values(),
            *self.interstage_const_lower.values(),
        ]

        if self.chance_consts is not None:
            constraints.extend(self.chance_consts)

        return cp.Problem(self.obj, constraints)


def main():
    # Generate a simple scenario tree
    stages = 4

    root, all_nodes = random_walk_tree_builder(
        [
            "crude_light_price",
            "crude_heavy_price",
            "prod_price_0",
            "prod_price_1",
            "prod_price_2",
            "demand_0",
            "demand_1",
            "demand_2",
        ],
        [30, 20, 50, 40, 30, 400, 300, 200],
        [1, 1, 1, 1, 1, 30, 30, 30],
        [0, 0, 0, 0, 0, 60, 60, 60],
        stages=stages,
        branch_factor=2,
        seed=42,
        truncate_places=0,
    )

    p = Plant(
        crude_distil_cap=1000,
        products=3,
        crude_ratios=np.array([[3, 1], [1, 2], [0, 1]]),
        refine_caps=[1000, 1000, 1000],
        product_ratios=np.array(
            [
                [2, 1, 0.0],  # Amount of light product from l/m/h intermediates
                [0.2, 1.0, 0.2],  # Amount of medium product from l/m/h intermediates
                [0.0, 0.8, 2.0],  # Amount of heavy product from l/m/h intermediates
            ]
        ),
        stages=stages,
        scenario_tree_root=root,
        scenario_tree_all_nodes=all_nodes,
        allowed_output_change=20,
    )

    problem = p.get_problem()
    # print(problem)
    solver = cp.CLARABEL
    result = problem.solve(solver=solver, verbose=True)
    # result = problem.solve(canon_backend=cp.SCIPY_CANON_BACKEND, solver=solver, verbose=True)
    print("status:", problem.status)
    print("optimal value", problem.value)
    print("Light Import: ", p.light_crude_import[root].value)
    print("Heavy Import: ", p.heavy_crude_import[root].value)

    # for variable in problem.variables():
    #     print(f"{variable.name()}: {variable.value}")


def main2():
    stages = 6
    variables = [
        "crude_light_price",
        "crude_heavy_price",
        "prod_price_0",
        "prod_price_1",
        "prod_price_2",
        "demand_0",
        "demand_1",
        "demand_2",
    ]
    root, all_nodes = random_walk_tree_builder(
        variables,
        [30, 20, 50, 20, 10, 400, 300, 200],
        [1, 1, 1, 1, 1, 30, 30, 30],
        [0, 0, 0, 0, 0, 60, 60, 60],
        stages=stages,
        branch_factor=5,
        seed=42,
        truncate_places=0,
    )
    print(f"Generated {len(all_nodes)} nodes")

    crude_ratios = [
        [0.7, 0.3],
        [0.5, 0.5],
        [0, 1],
    ]  # light from [light,heavy]; medium from [light,heavy]; heavy from [light,heavy]
    product_ratios = [
        [0.7, 0.3, 0.0],  # Amount of light product from l/m/h intermediates
        [0.2, 0.5, 0.2],  # Amount of medium product from l/m/h intermediates
        [0.0, 0.3, 0.7],  # Amount of heavy product from l/m/h intermediates
    ]

    p = Plant(
        crude_distil_cap=1000,
        products=3,
        crude_ratios=crude_ratios,
        refine_caps=[1000, 1000, 1000],
        product_ratios=np.array(product_ratios),
        stages=stages,
        scenario_tree_root=root,
        scenario_tree_all_nodes=all_nodes,
        allowed_output_change=20,
        chance=(9000, 0.998),
    )

    problem = p.get_problem()
    # print(problem)
    # solver = cp.CLARABEL
    result = problem.solve(verbose=True)


if __name__ == "__main__":
    # main()
    main2()
