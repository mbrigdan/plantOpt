# plantOpt

Comparison of different linear programming techniques for optimizing a simplified (deliberately non-realistic) model of an oil refinery.
Using pyomo (could be expanded in the future (e.g. cvxpy?))

Model definitions are in [plant_opt/models/](plant_opt/models/)

| File                                                                                                              | Description                                                                    |
|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [plant_model_single_pyo.py](plant_opt/models/plant_model_single_pyo.py)                                           | Simplest, basic linear optimization of a single scenario                       |
| [plant_model_recourse_pyo.py](plant_opt/models/plant_model_recourse_pyo.py)                                       | Model with incomplete initial information and resource                         |
| [plant_model_stages_pyo.py](plant_opt/models/plant_model_stages_pyo.py)                                           | Model with multiple stages of decision making                                  |
| [plant_models_stages_recourse_stochastic_pyo.py](plant_opt/models/plant_models_stages_recourse_stochastic_pyo.py) | Multistage stochastic model, using pyomo kernel interface, scenario tree input |
| [plant_models_stages_recourse_stochastic_cvx.py](plant_opt/models/plant_models_stages_recourse_stochastic_cvx.py) | Above model re-implemented in CVXPY to facilitate convex constraints           |

Recommendation: start with [compare_recourse_nb.ipynb](plant_opt/compare_recourse_nb.ipynb)
which compares the simple model to the recourse model across a few scenarios. 

Then, [compare_stochastic_nb.ipynb](plant_opt/compare_stochastic_nb.ipynb) examines a stochastic model with multiple stages, and investigates the effect of "truncating" the model by using average values for scenarios further into the tree; examining the value of the additional stages and the distribution of outcomes.

[stochastic_modified_nb.ipynb](plant_opt/stochastic_modified_nb.ipynb) then builds on the above, using a re-implementation of the model within CVXPY to use convex constraints; primarily what is examined is the application of chance constraints to the model.