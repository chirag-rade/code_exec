import Agents
import json
from test_multiple import conversation


notebook = "Data/python_basics_&_scripting__write_code_in_python__22__24_03_2024_08_57_45_10.ipynb"
#eval = Agents.run(notebook_path=notebook,  happy_cases_n=3, edge_cases_n=3, recursion_limit=50) # max for cases == 20
eval = Agents.run_multiple_turns(notebook_path=notebook,  happy_cases_n=1, edge_cases_n=1, recursion_limit=50, n_workers=4, n_workers_inner_thread=4)

print("------------------------EVALUATION------------------------")
print(eval)
with open('final_evaluation.json', 'w') as f:
    json.dump(eval, f, indent=4)


#TODO
#fix turn
