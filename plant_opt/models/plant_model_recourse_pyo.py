import pyomo.environ as pyo


class Plant:
    def __init__(
        self,
        crude_distil_cap,   # Capacity of the distillation unit (crude oil input)
        crude_light_ratios, # 1 unit light crude turns into (x, y, z) units of light, medium, heavy intermediate
        crude_heavy_ratios, # 1 unit heavy crude turns into (x, y, z) units of light, medium, heavy intermediate
        refine_light_cap,   # maximum amount of inputs that can be sent to the light product refining unit
        refine_medium_cap,  # maximum amount of inputs that can be sent to the medium product refining unit
        refine_heavy_cap,   # maximum amount of inputs that can be sent to the heavy product refining unit
        prod_light_ratios,  # (x, y) -> light unit turns 1 unit of light intermediate into x units of light product, 1 unit of medium intermediate into y units of light product
        prod_medium_ratios, # (x, y) -> med unit turns 1 unit of light intermediate into x units of medium product, 1 unit of medium intermediate into y units of medium product
        prod_heavy_ratios,  # (x, y) -> heavy unit turns 1 unit of medium intermediate into x units of heavy product, 1 unit of heavy intermediate into y units of heavy product
        crude_light_price,
        crude_heavy_price,
        scenario_count,
        prod_light_price,
        prod_light_demand,
        prod_medium_price,
        prod_medium_demand,
        prod_heavy_price,
        prod_heavy_demand,
    ):
        self.model = m = pyo.ConcreteModel()

        # Define controllable variables

        # Stage 1: No recourse
        # I.e. we choose the imports without seeing the prices and demands for the products
        # Imports - everything goes to distillation
        m.light_crude_import = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_crude_import = pyo.Var( domain=pyo.NonNegativeReals)

        # Stage 2: Recourse - we see the prices and demands for the products and now make the production decisions
        # based on the imports we made in stage 1
        # So we index these variables (and associated constraints) by the scenarios
        m.scenarios = pyo.RangeSet(0, scenario_count - 1)

        # What gets sent to the refining units (intermediate products are the inputs)
        m.light_to_light_unit = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.med_to_light_unit = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)

        m.light_to_med_unit = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.med_to_med_unit = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)

        m.med_to_heavy_unit = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.heavy_to_heavy_unit = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)

        # Outputs
        m.light_prod_output = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.med_prod_output = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.heavy_prod_output = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)

        # Define intermediate variables
        m.light_intermediate = pyo.Var(domain=pyo.NonNegativeReals)
        m.med_intermediate = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_intermediate = pyo.Var(domain=pyo.NonNegativeReals)

        # Distillation constraints - these are stage 1, not indexed by scenario
        light_inter_expr = m.light_intermediate == m.light_crude_import * crude_light_ratios[0] + m.heavy_crude_import * crude_heavy_ratios[0]
        med_inter_expr = m.med_intermediate == m.light_crude_import * crude_light_ratios[1] + m.heavy_crude_import * crude_heavy_ratios[1]
        heavy_inter_expr = m.heavy_intermediate == m.light_crude_import * crude_light_ratios[2] + m.heavy_crude_import * crude_heavy_ratios[2]

        m.prod_light_const = pyo.Constraint(expr=light_inter_expr)
        m.prod_med_const = pyo.Constraint(expr=med_inter_expr)
        m.prod_heavy_const = pyo.Constraint(expr=heavy_inter_expr)

        # Production of outputs - these are stage 2, indexed by scenario
        def light_out_rule(m, i):
            return m.light_prod_output[i] == m.light_to_light_unit[i] * prod_light_ratios[0] + m.med_to_light_unit[i] * prod_light_ratios[1]
        def med_out_rule(m, i):
            return m.med_prod_output[i] == m.light_to_med_unit[i] * prod_medium_ratios[0] + m.med_to_med_unit[i] * prod_medium_ratios[1]
        def heavy_out_rule(m, i):
            return m.heavy_prod_output[i] == m.med_to_heavy_unit[i] * prod_heavy_ratios[0] + m.heavy_to_heavy_unit[i] * prod_heavy_ratios[1]

        m.prod_light_out_const = pyo.Constraint(m.scenarios, rule=light_out_rule)
        m.prod_med_out_const = pyo.Constraint(m.scenarios, rule=med_out_rule)
        m.prod_heavy_out_const = pyo.Constraint(m.scenarios, rule=heavy_out_rule)

        # Inputs to the refining units constraints
        def light_refine_rule(m, i):
            return m.light_to_light_unit[i] + m.light_to_med_unit[i] == m.light_intermediate
        def med_refine_rule(m, i):
            return m.med_to_light_unit[i] + m.med_to_med_unit[i] + m.med_to_heavy_unit[i] == m.med_intermediate
        def heavy_refine_rule(m, i):
            return m.heavy_to_heavy_unit[i] == m.heavy_intermediate

        m.light_refine_const = pyo.Constraint(m.scenarios, rule=light_refine_rule)
        m.med_refine_const = pyo.Constraint(m.scenarios, rule=med_refine_rule)
        m.heavy_refine_const = pyo.Constraint(m.scenarios, rule=heavy_refine_rule)

        # Capacity constraints
        distil_cap_expr = m.light_crude_import + m.heavy_crude_import <= crude_distil_cap
        m.distil_cap_const = pyo.Constraint(expr=distil_cap_expr)

        def refine_cap_l_rule(m, i):
            return m.light_to_light_unit[i] + m.med_to_light_unit[i] <= refine_light_cap
        def refine_cap_m_rule(m, i):
            return m.light_to_med_unit[i] + m.med_to_med_unit[i] <= refine_medium_cap
        def refine_cap_h_rule(m, i):
            return m.med_to_heavy_unit[i] + m.heavy_to_heavy_unit[i] <= refine_heavy_cap

        m.refine_cap_l_const = pyo.Constraint(m.scenarios, rule=refine_cap_l_rule)
        m.refine_cap_m_const = pyo.Constraint(m.scenarios, rule=refine_cap_m_rule)
        m.refine_cap_h_const = pyo.Constraint(m.scenarios, rule=refine_cap_h_rule)

        # Break down outputs so we can calculate payment only up to demand
        m.light_prod_full_price = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.light_prod_excess = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.med_prod_full_price = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.med_prod_excess = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.heavy_prod_full_price = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)
        m.heavy_prod_excess = pyo.Var(m.scenarios, domain=pyo.NonNegativeReals)

        def make_breakdown_rule(output, full_price, excess):
            def rule(_m, i):
                return output[i] == full_price[i] + excess[i]
            return rule

        def make_demand_rule(full_price, demand):
            def rule(_m, i):
                return full_price[i] <= demand[i]
            return rule

        output_breakdowns = [
            ("light", m.light_prod_output, m.light_prod_full_price, m.light_prod_excess, prod_light_demand),
            ("med", m.med_prod_output, m.med_prod_full_price, m.med_prod_excess, prod_medium_demand),
            ("heavy", m.heavy_prod_output, m.heavy_prod_full_price, m.heavy_prod_excess, prod_heavy_demand)
        ]

        m.output_breakdown_constraints = []

        for name, output, full_price, excess, demand in output_breakdowns:
            setattr(m, f"demand_breakdown_{name}_const", pyo.Constraint(m.scenarios, rule=make_breakdown_rule(output, full_price, excess)))
            setattr(m, f"demand_constraint_{name}_const", pyo.Constraint(m.scenarios, rule=make_demand_rule(full_price, demand)))

        def objective_rule(m):
            value = sum(m.light_prod_full_price[i] * prod_light_price[i]
                     + m.med_prod_full_price[i] * prod_medium_price[i]
                     + m.heavy_prod_full_price[i] * prod_heavy_price[i]
                     - m.light_crude_import * crude_light_price
                     - m.heavy_crude_import * crude_heavy_price for i in m.scenarios
                     )
            return value

        m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


def main():
    # Scenarios to test in optimization:
    # 0 - light demand low (below capacity) + others normal
    # 1 - med demand low (below capacity) + others normal
    # 2 - heavy demand zero (below capacity) + others normal
    # 3 - all demands normal (above capacity)
    # 4 - light and med demand low (below capacity) + heavy demand normal (above capacity)
    scenarios = 5

    c = 2000

    prod_light_price = [50] * 5
    prod_light_demand = [500, c, c, c, 500]
    prod_medium_price = [30] * 5
    prod_medium_demand = [c, 500, c, c, 500]
    prod_heavy_price = [10] * 5
    prod_heavy_demand = [c, c, 0, c, c]

    p = Plant(
        crude_distil_cap=c,
        crude_light_ratios=(4, 2, 1),
        crude_heavy_ratios=(1, 2, 4),
        refine_light_cap=c,
        refine_medium_cap=c,
        refine_heavy_cap=c,
        prod_light_ratios=(2, 1),
        prod_medium_ratios=(1, 2),
        prod_heavy_ratios=(1, 2),
        crude_light_price=30,
        crude_heavy_price=10,
        scenario_count=scenarios,
        prod_light_price=prod_light_price,
        prod_light_demand=prod_light_demand,
        prod_medium_price=prod_medium_price,
        prod_medium_demand=prod_medium_demand,
        prod_heavy_price=prod_heavy_price,
        prod_heavy_demand=prod_heavy_demand,
    )

    opt = pyo.SolverFactory("glpk")
    p.model.pprint()
    results = opt.solve(p.model)

    print(results)

    for v in p.model.component_objects(pyo.Var, active=True):
        print("Variable", v)
        for index in v:
            print(" ", index, v[index].value)

    p.model.pprint()

    print("Objective", pyo.value(p.model.obj))
    print("Light Crude Import: ", pyo.value(p.model.light_crude_import))
    print("Heavy Crude Import: ", pyo.value(p.model.heavy_crude_import))

if __name__ == "__main__":
    main()