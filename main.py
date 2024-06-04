import Agents
import json

notebook = "Data/latest/3-june/data_analysis__write___modify___fix_SQL_code__8__18_03_2024_08_55_14_10.ipynb"
eval = Agents.run(notebook_path=notebook,  happy_cases_n=1, edge_cases_n=1, recursion_limit=50) # max for cases == 4
print("------------------------EVALUATION------------------------")
print(eval)
with open('final_evaluation.json', 'w') as f:
    json.dump(eval, f, indent=4)

