import pyomo.environ as pyo


class Plant:
    def __init__(
        self,
        crude_distil_cap,  # Capacity of the distillation unit (crude oil input)
        crude_light_ratios,  # 1 unit light crude turns into (x, y, z) units of light, medium, heavy intermediate
        crude_heavy_ratios,  # 1 unit heavy crude turns into (x, y, z) units of light, medium, heavy intermediate
        refine_light_cap,  # maximum amount of inputs that can be sent to the light product refining unit
        refine_medium_cap, # maximum amount of inputs that can be sent to the medium product refining unit
        refine_heavy_cap,  # maximum amount of inputs that can be sent to the heavy product refining unit
        prod_light_ratios, # (x, y) -> light unit turns 1 unit of light intermediate into x units of light product, 1 unit of medium intermediate into y units of light product
        prod_medium_ratios, # (x, y) -> med unit turns 1 unit of light intermediate into x units of medium product, 1 unit of medium intermediate into y units of medium product
        prod_heavy_ratios,  # (x, y) -> heavy unit turns 1 unit of medium intermediate into x units of heavy product, 1 unit of heavy intermediate into y units of heavy product
        crude_light_price,
        crude_heavy_price,
        prod_light_price,
        prod_light_demand,
        prod_medium_price,
        prod_medium_demand,
        prod_heavy_price,
        prod_heavy_demand,
    ):
        self.model = m = pyo.ConcreteModel()

        # Define controllable variables
        # Imports - everything goes to distillation
        m.light_crude_import = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_crude_import = pyo.Var( domain=pyo.NonNegativeReals)
        # What gets sent to the refining units (intermediate products are the inputs)
        m.light_to_light_unit = pyo.Var(domain=pyo.NonNegativeReals)
        m.med_to_light_unit = pyo.Var(domain=pyo.NonNegativeReals)

        m.light_to_med_unit = pyo.Var(domain=pyo.NonNegativeReals)
        m.med_to_med_unit = pyo.Var(domain=pyo.NonNegativeReals)

        m.med_to_heavy_unit = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_to_heavy_unit = pyo.Var(domain=pyo.NonNegativeReals)

        # Outputs
        m.light_prod_output = pyo.Var(domain=pyo.NonNegativeReals)
        m.med_prod_output = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_prod_output = pyo.Var(domain=pyo.NonNegativeReals)

        # Define intermediate variables
        m.light_intermediate = pyo.Var( domain=pyo.NonNegativeReals)
        m.med_intermediate = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_intermediate = pyo.Var(domain=pyo.NonNegativeReals)

        # Production constraints
        light_inter_expr = m.light_intermediate == m.light_crude_import * crude_light_ratios[0] + m.heavy_crude_import * crude_heavy_ratios[0]
        med_inter_expr = m.med_intermediate == m.light_crude_import * crude_light_ratios[1] + m.heavy_crude_import * crude_heavy_ratios[1]
        heavy_inter_expr = m.heavy_intermediate == m.light_crude_import * crude_light_ratios[2] + m.heavy_crude_import * crude_heavy_ratios[2]

        m.prod_light_const = pyo.Constraint(expr=light_inter_expr)
        m.prod_med_const = pyo.Constraint(expr=med_inter_expr)
        m.prod_heavy_const = pyo.Constraint(expr=heavy_inter_expr)

        light_out_expr = m.light_prod_output == m.light_to_light_unit * prod_light_ratios[0] + m.med_to_light_unit * prod_light_ratios[1]
        med_out_expr = m.med_prod_output == m.light_to_med_unit * prod_medium_ratios[0] + m.med_to_med_unit * prod_medium_ratios[1]
        heavy_out_expr = m.heavy_prod_output == m.med_to_heavy_unit * prod_heavy_ratios[0] + m.heavy_to_heavy_unit * prod_heavy_ratios[1]

        m.prod_light_out_const = pyo.Constraint(expr=light_out_expr)
        m.prod_med_out_const = pyo.Constraint(expr=med_out_expr)
        m.prod_heavy_out_const = pyo.Constraint(expr=heavy_out_expr)

        # Inputs to the refining units constraints
        light_refine_expr = m.light_to_light_unit + m.light_to_med_unit == m.light_intermediate
        med_refine_expr = m.med_to_light_unit + m.med_to_med_unit + m.med_to_heavy_unit == m.med_intermediate
        heavy_refine_expr = m.heavy_to_heavy_unit == m.heavy_intermediate

        m.light_refine_const = pyo.Constraint(expr=light_refine_expr)
        m.med_refine_const = pyo.Constraint(expr=med_refine_expr)
        m.heavy_refine_const = pyo.Constraint(expr=heavy_refine_expr)

        # Capacity constraints
        distil_cap_expr = m.light_crude_import + m.heavy_crude_import <= crude_distil_cap
        m.distil_cap_const = pyo.Constraint(expr=distil_cap_expr)

        refine_cap_l_expr = m.light_to_light_unit + m.med_to_light_unit <= refine_light_cap
        refine_cap_m_expr = m.light_to_med_unit + m.med_to_med_unit  <= refine_medium_cap
        refine_cap_h_expr = m.med_to_heavy_unit + m.heavy_to_heavy_unit <= refine_heavy_cap

        m.refine_cap_l_const = pyo.Constraint(expr=refine_cap_l_expr)
        m.refine_cap_m_const = pyo.Constraint(expr=refine_cap_m_expr)
        m.refine_cap_h_const = pyo.Constraint(expr=refine_cap_h_expr)

        # Break down outputs so we can calculate payment only up to demand
        m.light_prod_full_price = pyo.Var(domain=pyo.NonNegativeReals)
        m.light_prod_excess = pyo.Var(domain=pyo.NonNegativeReals)
        m.med_prod_full_price = pyo.Var(domain=pyo.NonNegativeReals)
        m.med_prod_excess = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_prod_full_price = pyo.Var(domain=pyo.NonNegativeReals)
        m.heavy_prod_excess = pyo.Var(domain=pyo.NonNegativeReals)

        def make_breakdown_expr(output, full_price, excess, _demand):
            return output == full_price + excess

        def make_demand_expr(_output, full_price, _excess, demand):
            return full_price <= demand

        output_breakdowns = [
            (m.light_prod_output, m.light_prod_full_price, m.light_prod_excess, prod_light_demand),
            (m.med_prod_output, m.med_prod_full_price, m.med_prod_excess, prod_medium_demand),
            (m.heavy_prod_output, m.heavy_prod_full_price, m.heavy_prod_excess, prod_heavy_demand)
        ]

        m.output_breakdown_constraints = pyo.ConstraintList()

        for output, full_price, excess, demand in output_breakdowns:
            m.output_breakdown_constraints.add(make_breakdown_expr(output, full_price, excess, demand))
            m.output_breakdown_constraints.add(make_demand_expr(output, full_price, excess, demand))

        def objective_rule(m):
            value = (m.light_prod_full_price * prod_light_price
                     + m.med_prod_full_price * prod_medium_price
                     + m.heavy_prod_full_price * prod_heavy_price
                     - m.light_crude_import * crude_light_price
                     - m.heavy_crude_import * crude_heavy_price
                     )
            return value

        m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


if __name__ == "__main__":
    p = Plant(
        crude_distil_cap=1000,
        crude_light_ratios=(4, 2, 1),
        crude_heavy_ratios=(0, 2, 3),
        refine_light_cap=1000,
        refine_medium_cap=1500,
        refine_heavy_cap=5000,
        prod_light_ratios=(2, 1),
        prod_medium_ratios=(1, 2),
        prod_heavy_ratios=(1, 2),
        crude_light_price=20,
        crude_heavy_price=10,
        prod_light_price=60,
        prod_light_demand=10000,
        prod_medium_price=10,
        prod_medium_demand=10000,
        prod_heavy_price=10,
        prod_heavy_demand=10000,
    )


    opt = pyo.SolverFactory("glpk")
    print(p.model)
    results = opt.solve(p.model)

    print(results)

    for v in p.model.component_objects(pyo.Var, active=True):
        print("Variable", v)
        for index in v:
            print(" ", index, v[index].value)