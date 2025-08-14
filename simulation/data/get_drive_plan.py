import pandas as pd

url = "https://www.tricon-terminal.de/info-desk/fahrplan"
dfs = pd.read_html(url)

print(len(dfs))

dfs[0].to_csv('driving_plan_versand.csv', index=False)
dfs[2].to_csv('driving_plan_empfang.csv', index=False)