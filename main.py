import Agents
import json

notebook = "Data/test_write.ipynb"
eval = Agents.run(notebook_path=notebook,  happy_cases_n=3, edge_cases_n=3, recursion_limit=50) # max for cases == 20
print("------------------------EVALUATION------------------------")
print(eval)
with open('final_evaluation.json', 'w') as f:
    json.dump(eval, f, indent=4)


