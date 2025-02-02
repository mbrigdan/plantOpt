import pyomo.environ as pyo
import pandas as pd

from plant_con.models.plant_model_single_pyo import Plant as PlantSingle
from plant_con.models.plant_model_recourse_pyo import Plant as PlantRecourse


def scenario_obj(
    m,
    i: int,
    prod_light_price: list[int],
    prod_medium_price: list[int],
    prod_heavy_price: list[int],
    crude_light_price: int,
    crude_heavy_price: int,
) -> float:
    """
    Determine the objective function for a given scenario of the recourse model
    - the real objective includes all the scenarios combined, so use this for comparison
    """
    v = pyo.value

    return (
        v(m.light_prod_full_price[i]) * prod_light_price[i]
        + v(m.med_prod_full_price[i]) * prod_medium_price[i]
        + v(m.heavy_prod_full_price[i]) * prod_heavy_price[i]
        - v(m.light_crude_import) * crude_light_price
        - v(m.heavy_crude_import) * crude_heavy_price
    )


def main():
    # Scenarios to test in optimization:
    # 0 - light demand is zero, medium and heavy demand high (2000 - higher than achievable)
    # 1 - light demand is 50, medium demand is 50, heavy demand is high
    # 2 - all demands high

    # constants
    crude_light_ratios = (3, 0.3, 0)
    crude_heavy_ratios = (0, 1, 4)
    light_product_ratios = (2, 1)
    medium_product_input_ratios = (1, 1)
    heavy_product_input_ratios = (1, 2)

    crude_light_price = 30
    crude_heavy_price = 10

    # Define values
    # capacity
    c = 2000
    crude_cap = 100

    scenarios = 3
    prod_light_price = [50] * 3
    prod_light_demand = [0, 50, c]
    prod_medium_price = [10] * 3
    prod_medium_demand = [c, 50, c]
    prod_heavy_price = [10] * 3
    prod_heavy_demand = [c, c, c]

    opt = pyo.SolverFactory("glpk")

    # First, we will solve the non-recourse model individually for each scenario.
    non_recourse_df = pd.DataFrame(
        columns=[
            "Scenario",
            "Objective Value",
            "Light Crude Import",
            "Heavy Crude Import",
            "Light Product Output",
            "Medium Product Output",
            "Heavy Product Output",
            "Light Inter to Light Unit",
            "Med Inter to Light Unit",
            "Light Inter to Med Unit",
            "Med Inter to Med Unit",
            "Med Inter to Heavy Unit",
            "Heavy Inter to Heavy Unit",
        ]
    )

    for ii in range(scenarios):
        p = PlantSingle(
            crude_distil_cap=crude_cap,
            crude_light_ratios=crude_light_ratios,
            crude_heavy_ratios=crude_heavy_ratios,
            refine_light_cap=c,
            refine_medium_cap=c,
            refine_heavy_cap=c,
            prod_light_ratios=light_product_ratios,
            prod_medium_ratios=medium_product_input_ratios,
            prod_heavy_ratios=heavy_product_input_ratios,
            crude_light_price=crude_light_price,
            crude_heavy_price=crude_heavy_price,
            prod_light_price=prod_light_price[ii],
            prod_light_demand=prod_light_demand[ii],
            prod_medium_price=prod_medium_price[ii],
            prod_medium_demand=prod_medium_demand[ii],
            prod_heavy_price=prod_heavy_price[ii],
            prod_heavy_demand=prod_heavy_demand[ii],
        )

        results = opt.solve(p.model)

        with open(f"results_single_{ii}.txt", "w") as f:
            p.model.pprint(ostream=f)

        non_recourse_df.loc[ii] = (
            ii,
            pyo.value(p.model.obj),
            pyo.value(p.model.light_crude_import),
            pyo.value(p.model.heavy_crude_import),
            pyo.value(p.model.light_prod_output),
            pyo.value(p.model.med_prod_output),
            pyo.value(p.model.heavy_prod_output),
            pyo.value(p.model.light_to_light_unit),
            pyo.value(p.model.med_to_light_unit),
            pyo.value(p.model.light_to_med_unit),
            pyo.value(p.model.med_to_med_unit),
            pyo.value(p.model.med_to_heavy_unit),
            pyo.value(p.model.heavy_to_heavy_unit),
        )

    print(non_recourse_df.to_markdown())

    # Now, solve the recourse model
    p = PlantRecourse(
        crude_distil_cap=crude_cap,
        crude_light_ratios=crude_light_ratios,
        crude_heavy_ratios=crude_heavy_ratios,
        refine_light_cap=c,
        refine_medium_cap=c,
        refine_heavy_cap=c,
        prod_light_ratios=light_product_ratios,
        prod_medium_ratios=medium_product_input_ratios,
        prod_heavy_ratios=heavy_product_input_ratios,
        crude_light_price=crude_light_price,
        crude_heavy_price=crude_heavy_price,
        scenario_count=scenarios,
        prod_light_price=prod_light_price,
        prod_light_demand=prod_light_demand,
        prod_medium_price=prod_medium_price,
        prod_medium_demand=prod_medium_demand,
        prod_heavy_price=prod_heavy_price,
        prod_heavy_demand=prod_heavy_demand,
    )

    print(f"Recourse Model")
    results = opt.solve(p.model)

    with open(f"results_single_{ii}.txt", "w") as f:
        p.model.pprint(ostream=f)

    recourse_df = pd.DataFrame(
        columns=[
            "Scenario",
            "Objective Value",
            "Light Crude Import",
            "Heavy Crude Import",
            "Light Product Output",
            "Medium Product Output",
            "Heavy Product Output",
            "Light Inter to Light Unit",
            "Med Inter to Light Unit",
            "Light Inter to Med Unit",
            "Med Inter to Med Unit",
            "Med Inter to Heavy Unit",
            "Heavy Inter to Heavy Unit",
        ]
    )

    recourse_df["Scenario"] = range(scenarios)
    recourse_df["Objective Value"] = [
        scenario_obj(
            p.model,
            ii,
            prod_light_price,
            prod_medium_price,
            prod_heavy_price,
            crude_light_price,
            crude_heavy_price,
        )
        for ii in range(scenarios)
    ]
    recourse_df["Light Crude Import"] = [
        pyo.value(p.model.light_crude_import) for ii in range(scenarios)
    ]
    recourse_df["Heavy Crude Import"] = [
        pyo.value(p.model.heavy_crude_import) for ii in range(scenarios)
    ]
    recourse_df["Light Product Output"] = [
        pyo.value(p.model.light_prod_output[ii]) for ii in range(scenarios)
    ]
    recourse_df["Medium Product Output"] = [
        pyo.value(p.model.med_prod_output[ii]) for ii in range(scenarios)
    ]
    recourse_df["Heavy Product Output"] = [
        pyo.value(p.model.heavy_prod_output[ii]) for ii in range(scenarios)
    ]
    recourse_df["Light Inter to Light Unit"] = [
        pyo.value(p.model.light_to_light_unit[ii]) for ii in range(scenarios)
    ]
    recourse_df["Med Inter to Light Unit"] = [
        pyo.value(p.model.med_to_light_unit[ii]) for ii in range(scenarios)
    ]
    recourse_df["Light Inter to Med Unit"] = [
        pyo.value(p.model.light_to_med_unit[ii]) for ii in range(scenarios)
    ]
    recourse_df["Med Inter to Med Unit"] = [
        pyo.value(p.model.med_to_med_unit[ii]) for ii in range(scenarios)
    ]
    recourse_df["Med Inter to Heavy Unit"] = [
        pyo.value(p.model.med_to_heavy_unit[ii]) for ii in range(scenarios)
    ]
    recourse_df["Heavy Inter to Heavy Unit"] = [
        pyo.value(p.model.heavy_to_heavy_unit[ii]) for ii in range(scenarios)
    ]

    print(recourse_df.to_markdown())


if __name__ == "__main__":
    main()
