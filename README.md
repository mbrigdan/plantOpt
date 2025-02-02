# plantOpt

Comparison of different linear programming techniques for optimizing a simplified (deliberately non-realistic) model of an oil refinery.
Using pyomo (could be expanded in the future (e.g. cvxpy?))

Model definitions are in [plant_opt/models/](plant_opt/models/)

| File                                                                        | Description                                              |
|-----------------------------------------------------------------------------|----------------------------------------------------------|
| [plant_model_single_pyo.py](plant_opt/models/plant_model_single_pyo.py)    | Simplest, basic linear optimization of a single scenario |
| [plant_model_recourse_pyo.py](plant_opt/models/plant_model_recourse_pyo.py) | Model with incomplete initial information and resource   |

Recommendation: start with [compare_recourse_nb.ipynb](plant_opt/compare_recourse_nb.ipynb)
which compares the simple model to the recourse model across a few scenarios. 