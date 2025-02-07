from collections import defaultdict

import pyomo.kernel as pmo
from pyomo.kernel.util import pprint

from plant_opt.scenario_tree.tree import Node, random_walk_tree_builder


def build_non_recourse_var(var_dict: pmo.variable_dict, root, domain):
    nodes_to_process = [root]

    while nodes_to_process:
        node = nodes_to_process.pop(0)
        new_var = pmo.variable(domain=domain)

        var_dict[node] = new_var

        for child in node.children:
            if child.children:
                # Non-recourse variables are not added at terminal nodes, so only add if the node has children
                nodes_to_process.append(child)


def build_non_recourse_var_list(var_dict: pmo.variable_dict, root, domain, length):
    nodes_to_process = [root]

    while nodes_to_process:
        node = nodes_to_process.pop(0)
        new_var = pmo.variable_list()

        for _ in range(length):
            new_var.append(pmo.variable(domain=domain))

        var_dict[node] = new_var

        for child in node.children:
            if child.children:
                # Non-recourse variables are not added at terminal nodes, so only add if the node has children
                nodes_to_process.append(child)


def build_recourse_var(var_dict: pmo.variable_dict, root, domain, initializer=None):
    # Don't add recourse variables at the root node, only process the children
    nodes_to_process = root.children.copy()

    while nodes_to_process:
        node = nodes_to_process.pop(0)
        new_var = pmo.variable(domain=domain) if initializer is None else initializer()

        var_dict[node] = new_var

        for child in node.children:
            nodes_to_process.append(child)


def build_intermediate_to_unit_var(var_dict: pmo.variable_dict, root, domain, products: int):
    # This is a recourse variable, so the function is similar to the above,
    # except the intermediate_to_unit is actually an n x n matrix, where n is the number of products
    # so we construct a list of lists of variables

    # Don't add recourse variables at the root node, only process the children
    nodes_to_process = root.children.copy()

    while nodes_to_process:
        node = nodes_to_process.pop(0)
        new_var = pmo.variable_list()

        for _ in range(products):
            inner_list = pmo.variable_list()
            new_var.append(inner_list)
            for _ in range(products):
                inner_list.append(pmo.variable(domain=domain))

        var_dict[node] = new_var

        for child in node.children:
            nodes_to_process.append(child)


class Plant:
    def __init__(
        self,
        crude_distil_cap,  # Capacity of the distillation unit (crude oil input)
        products: int,  # Number of products
        crude_ratios: list[list[float]],
        refine_caps: list[float],
        product_ratios: list[list[float]],  # indexed by outputs then intermediates
        stages: int,
        scenario_tree_root: Node,
        scenario_tree_all_nodes: list[Node],
        allowed_output_change: float,
    ):
        self.model = m = pmo.block()

        m.stages = range(0, stages)
        m.nodes = scenario_tree_all_nodes

        m.products = range(0, products)
        m.outputs = range(0, products)

        # Define controllable variables
        # Imports - everything goes to distillation

        self.node_var_map = defaultdict(dict)

        m.light_crude_import = pmo.variable_dict()
        m.heavy_crude_import = pmo.variable_dict()

        build_non_recourse_var(
            m.light_crude_import,
            scenario_tree_root,
            pmo.NonNegativeReals,
        )
        build_non_recourse_var(
            m.heavy_crude_import,
            scenario_tree_root,
            pmo.NonNegativeReals,
        )

        # Quantities of intermediates
        m.intermediates = pmo.variable_dict()
        build_non_recourse_var_list(
            m.intermediates, scenario_tree_root, pmo.NonNegativeReals, products
        )

        # What gets sent to the refining units (intermediate products are the inputs)
        m.intermediate_to_unit = pmo.variable_dict()
        build_intermediate_to_unit_var(
            m.intermediate_to_unit,
            scenario_tree_root,
            pmo.NonNegativeReals,
            products,
        )

        # Outputs
        m.product_output = pmo.variable_dict()

        def make_prod_var_list():
            l = pmo.variable_list()
            for _ in range(products):
                l.append(pmo.variable(domain=pmo.NonNegativeReals))
            return l

        build_recourse_var(
            m.product_output,
            scenario_tree_root,
            pmo.NonNegativeReals,
            initializer=make_prod_var_list,
        )

        # Production of intermediates
        m.distil_const = pmo.constraint_dict()
        for node in m.nodes:
            if not node.children:
                # Skip terminal nodes (non-recourse var)
                continue
            m.distil_const[node] = pmo.constraint_list()
            for prod in m.products:
                m.distil_const[node].append(
                    pmo.constraint(
                        m.intermediates[node][prod]
                        == m.light_crude_import[node] * crude_ratios[prod][0]
                        + m.heavy_crude_import[node] * crude_ratios[prod][1]
                    )
                )

        # "distil" capacity constraints
        m.distil_cap_const = pmo.constraint_dict()
        for node in m.nodes:
            if not node.children:
                # Skip terminal nodes (non-recourse var)
                continue
            m.distil_cap_const[node] = pmo.constraint(
                m.light_crude_import[node] + m.heavy_crude_import[node] <= crude_distil_cap
            )

        # Production of outputs
        m.product_out_const = pmo.constraint_dict()
        for node in m.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue
            m.product_out_const[node] = pmo.constraint_list()
            for output in m.outputs:
                m.product_out_const[node].append(
                    pmo.constraint(
                        m.product_output[node][output]
                        == sum(
                            m.intermediate_to_unit[node][prod][output]
                            * product_ratios[output][prod]
                            for prod in m.products
                        )
                    )
                )

        m.refine_cap_const = pmo.constraint_dict()
        for node in m.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            m.refine_cap_const[node] = pmo.constraint_list()
            for output in m.outputs:
                m.refine_cap_const[node].append(
                    pmo.constraint(
                        sum(m.intermediate_to_unit[node][prod][output] for prod in m.products)
                        <= refine_caps[output]
                    )
                )

        # Intermediates must exist to be sent to refining units
        # 100% of intermediates must be sent to refining units - can be changed to allow for non-use
        # by making the constraint <= instead of ==
        m.intermediates_rule = pmo.constraint_dict()
        for node in m.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            m.intermediates_rule[node] = pmo.constraint_list()
            for prod in m.products:
                m.intermediates_rule[node].append(
                    pmo.constraint(
                        m.intermediates[node.parent][prod]
                        == sum(m.intermediate_to_unit[node][prod][output] for output in m.outputs)
                    )
                )

        # Break down outputs so we can calculate payment only up to demand
        m.prod_full_price = pmo.variable_dict()
        build_recourse_var(
            m.prod_full_price,
            scenario_tree_root,
            pmo.NonNegativeReals,
            initializer=make_prod_var_list,
        )

        m.prod_excess = pmo.variable_dict()
        build_recourse_var(
            m.prod_excess,
            scenario_tree_root,
            pmo.NonNegativeReals,
            initializer=make_prod_var_list,
        )

        m.breakdown_const = pmo.constraint_dict()
        for node in m.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            m.breakdown_const[node] = pmo.constraint_list()
            for output in m.outputs:
                m.breakdown_const[node].append(
                    pmo.constraint(
                        m.product_output[node][output]
                        == m.prod_full_price[node][output] + m.prod_excess[node][output]
                    )
                )

        m.demand_const = pmo.constraint_dict()
        for node in m.nodes:
            if not node.parent:
                # Skip root node (recourse var)
                continue

            m.demand_const[node] = pmo.constraint_list()
            for output in m.outputs:
                m.demand_const[node].append(
                    pmo.constraint(
                        m.prod_full_price[node][output] <= node.values[f"demand_{output}"]
                    )
                )

        m.interstage_const = pmo.constraint_dict()

        for node in m.nodes:
            if not node.parent or not node.parent.parent:
                # Skip root node (recourse var) and first level
                continue

            m.interstage_const[node] = pmo.constraint_list()
            for output in m.outputs:
                if node.parent is None:
                    continue
                m.interstage_const[node].append(
                    pmo.constraint(
                        m.product_output[node][output]
                        <= m.product_output[node.parent][output] + allowed_output_change
                    )
                )
                m.interstage_const[node].append(
                    pmo.constraint(
                        m.product_output[node][output]
                        >= m.product_output[node.parent][output] - allowed_output_change
                    )
                )

        node_value_list = []

        stage_node_count = defaultdict(int)
        for node in scenario_tree_all_nodes:
            stage_node_count[node.stage] += 1

        for node in m.nodes:
            if not node.parent:
                # Skip root node
                continue

            # Important: need to multiply the value of nodes by 1/(num nodes in stage) to get the correct value
            # (implicit assumption of equal probability of each node in the stage)
            node_val = 0
            for output in m.outputs:
                node_val += m.prod_full_price[node][output] * node.values[f"prod_price_{output}"]
            node_val -= m.light_crude_import[node.parent] * node.values["crude_light_price"]
            node_val -= m.heavy_crude_import[node.parent] * node.values["crude_heavy_price"]

            node_value_list.append(node_val / stage_node_count[node.stage])

        m.obj = pmo.objective(sum(node_value_list), sense=pmo.maximize)


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
        crude_ratios=[[3, 1, 0], [1, 2, 1], [0, 1, 2]],
        refine_caps=[1000, 1000, 1000],
        product_ratios=[
            [2, 1, 0.0],  # Amount of light product from l/m/h intermediates
            [0.2, 1.0, 0.2],  # Amount of medium product from l/m/h intermediates
            [0.0, 0.8, 2.0],  # Amount of heavy product from l/m/h intermediates
        ],
        stages=stages,
        scenario_tree_root=root,
        scenario_tree_all_nodes=all_nodes,
        allowed_output_change=20,
    )

    opt = pmo.SolverFactory("glpk")
    results = opt.solve(p.model)

    print(results)

    pprint(p.model)


if __name__ == "__main__":
    main()
