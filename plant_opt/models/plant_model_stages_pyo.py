import pyomo.environ as pyo


class Plant:
    def __init__(
        self,
        crude_distil_cap,   # Capacity of the distillation unit (crude oil input)
        products: int,      # Number of products
        crude_ratios: list[list[float]],
        refine_caps: list[float],
        product_ratios: list[list[float]],  # indexed by outputs then intermediates
        crude_light_price: list[float],
        crude_heavy_price: list[float],
        stages: int,
        prod_prices: list[list[float]],
        prod_demands: list[list[float]],
        allowed_output_change: float,
    ):
        self.model = m = pyo.ConcreteModel()

        m.stages = pyo.RangeSet(0, stages - 1)
        m.non_initial_stages = pyo.RangeSet(1, stages - 1)

        m.products = pyo.RangeSet(0, products - 1)
        m.outputs = pyo.RangeSet(0, products - 1)

        # Define controllable variables
        # Imports - everything goes to distillation
        m.light_crude_import = pyo.Var(m.stages, domain=pyo.NonNegativeReals)
        m.heavy_crude_import = pyo.Var( m.stages, domain=pyo.NonNegativeReals)

        # Quantities of intermediates
        m.intermediates = pyo.Var(m.products, m.stages, domain=pyo.NonNegativeReals)

        # What gets sent to the refining units (intermediate products are the inputs)
        # note indexed twice because any intermediate can be sent to any unit
        m.intermediate_to_unit = pyo.Var(m.products, m.outputs, m.stages, domain=pyo.NonNegativeReals)

        # Outputs
        m.product_output = pyo.Var(m.outputs, m.stages, domain=pyo.NonNegativeReals)

        # Distillation constraints (production of intermediates)
        def distil_rule(m, prod, stage):
            return m.intermediates[prod, stage] == m.light_crude_import[stage] * crude_ratios[prod][0] + m.heavy_crude_import[stage] * crude_ratios[prod][1]

        m.distil_const = pyo.Constraint(m.products, m.stages, rule=distil_rule)

        # Production of outputs
        def output_rule(m, output, stage):
            return m.product_output[output, stage] == sum(m.intermediate_to_unit[prod, output, stage] * product_ratios[output][prod] for prod in m.products)

        m.product_out_const = pyo.Constraint(m.outputs, m.stages, rule=output_rule)

        # Inputs to the refining units constraints
        # The max capacity of each refining unit
        def refine_cap_rule(m, output, stage):
            return sum(m.intermediate_to_unit[prod, output, stage] for prod in m.products) <= refine_caps[output]

        m.refine_cap_const = pyo.Constraint(m.outputs, m.stages, rule=refine_cap_rule)

        # Need to send intermediates to send to the refining units
        def intermediate_to_unit_rule(m, prod, stage):
            return m.intermediates[prod, stage] == sum(m.intermediate_to_unit[prod, output, stage] for output in m.outputs)

        m.intermediates_rule = pyo.Constraint(m.products, m.stages, rule=intermediate_to_unit_rule)

        # Capacity constraints
        def distil_cap_rule(m, stage):
            return m.light_crude_import[stage] + m.heavy_crude_import[stage] <= crude_distil_cap
        m.distil_cap_const = pyo.Constraint(m.stages, rule=distil_cap_rule)

        # Break down outputs so we can calculate payment only up to demand
        m.prod_full_price = pyo.Var(m.outputs, m.stages, domain=pyo.NonNegativeReals)
        m.prod_excess = pyo.Var(m.outputs, m.stages, domain=pyo.NonNegativeReals)

        def breakdown_rule(m, output, stage):
            return m.product_output[output, stage] == m.prod_full_price[output, stage] + m.prod_excess[output, stage]

        def demand_rule(m, output, stage):
            return m.prod_full_price[output, stage] <= prod_demands[output][stage]

        m.breakdown_const = pyo.Constraint(m.outputs, m.stages, rule=breakdown_rule)
        m.demand_const = pyo.Constraint(m.outputs, m.stages, rule=demand_rule)

        # Interstage constraints to make the stages actually have meaning
        # don't let product outputs change between different stages by more than a set amount
        def interstage_rule_less(m, output, stage):
            if stage == 0:
                return pyo.Constraint.Skip
            return m.product_output[output, stage] <= m.product_output[output, stage - 1] + allowed_output_change
        def interstage_rule_greater(m, output, stage):
            if stage == 0:
                return pyo.Constraint.Skip
            return m.product_output[output, stage] >= m.product_output[output, stage - 1] - allowed_output_change

        m.interstage_const_less = pyo.Constraint(m.outputs, m.stages, rule=interstage_rule_less)
        m.interstage_const_greater = pyo.Constraint(m.outputs, m.stages, rule=interstage_rule_greater)

        def objective_rule(m):
            value = 0
            for output in m.outputs:
                for stage in m.stages:
                    value += m.prod_full_price[output, stage] * prod_prices[output][stage]

            for stage in m.stages:
                value -= m.light_crude_import[stage] * crude_light_price[stage]
                value -= m.heavy_crude_import[stage] * crude_heavy_price[stage]

            return value

        m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


def main():
    p = Plant(
        crude_distil_cap=1000,
        products=3,
        crude_ratios=[[2, 0], [1, 1], [0, 2]],
        refine_caps=[1000, 1000, 1000],
        product_ratios=[
            [2, 1, 0.0],   # Amount of light product from l/m/h intermediates
            [0.2, 1.0, 0.2], # Amount of medium product from l/m/h intermediates
            [0.0, 0.8, 2.0], # Amount of heavy product from l/m/h intermediates
        ],
        crude_light_price=[30, 20, 50],
        crude_heavy_price=[20, 50, 25],
        stages=3,
        prod_prices=[
            [40, 50, 30],  # Prices of light product in each stage
            [20, 60, 10],  # Prices of medium product in each stage
            [15, 30, 5],  # Prices of heavy product in each stage
        ],
        prod_demands=[
            [100, 200, 100],
            [200, 300, 200],
            [500, 500, 500],
        ],
        allowed_output_change=50,
    )


    opt = pyo.SolverFactory("glpk")
    results = opt.solve(p.model)

    print(results)

    p.model.pprint()


if __name__ == "__main__":
    main()