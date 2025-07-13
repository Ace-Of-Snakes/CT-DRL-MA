import pandas as pd

url = "https://www.tricon-terminal.de/info-desk/fahrplan"
dfs = pd.read_html(url)

print(len(dfs))

dfs[0].to_csv('driving_plan.csv', index=False)