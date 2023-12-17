import json
import pandas as pd
import openai
import random
from sklearn.model_selection import train_test_split

# read openai api
with open("api_keys.json", "r") as config_file:
    config = json.load(config_file)
    api_key = config.get("api_key", "")
openai.api_key = api_key

# Replace 'your_file.jsonl' with the path to your JSON Lines file
file_path = '../data/clean_up.jsonl'

# Read the .jsonl file line by line and parse each line as JSON
data = []
with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))
df = pd.DataFrame(data)


# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

unique_formulars = train_df["formula"].unique()

formular_data = {}

for formula in unique_formulars:
    tmp = train_df[train_df["formula"] == formula]
    random.seed(42)  
    selected_items = random.sample(tmp["natural"].tolist(), 5)
    examples = ', '.join(selected_items)

    formular_data[formula] = examples

assert formular_data and len(formular_data) == len(unique_formulars), print("Data processing failed")

with open('natural_formula_5.json', 'w') as json_file:
    json.dump(formular_data, json_file, indent=4)


file_path = 'natural_formula_5.json'
with open(file_path, 'r') as file:
    data = json.load(file)

contexts = []
for formula, examples in data.items():
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a natural language to linear temporal language translator. You are tasked with guiding a robot to navigate four rooms with colours, red, blue, green and yellow or any all colors synonymous to these. Given a linear temporal instruction and examples to go with it, generate a context describing that instruction in general terms."
            },
            {
                "role": "user",
                "content": f"formula: '{formula}', examples: '{examples}'"
            }
        ],
        temperature=1,
        max_tokens=512, # cahnge to 256 when necessary
        top_p=1,
        frequency_penalty=0.4,
        presence_penalty=0
    )
    context = response.choices[0].message.content 
    contexts.append({'formula': formula, 'context': context})

df = pd.DataFrame(contexts)
df.to_csv("contexts_512.csv", index=False) #change file name when 256