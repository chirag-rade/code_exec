import Agents
import json

notebook = "Data/latest/app_development_practices_with_Python__2__25-03-2024_09-10-23_1.ipynb"
eval = Agents.run(notebook_path=notebook,  happy_cases_n=1, edge_cases_n=1, recursion_limit=30) # max for cases == 4
print("------------------------EVALUATION------------------------")
print(eval)
with open('final_evaluation.json', 'w') as f:
    json.dump(eval, f)

