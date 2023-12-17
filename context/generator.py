"""
Module to generate context
"""

import json
import pandas as pd
import openai
import random
from sklearn.model_selection import train_test_split

def read_api_key(config_file_path):
    """Reads and returns the OpenAI API key from a config file."""
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)
        return config.get("api_key", "")

def load_data(file_path):
    """Loads data from a JSONL file into a pandas DataFrame."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def create_formula_examples(dataframe, unique_formulas, sample_size):
    """Creates a dictionary of formulas with example phrases."""
    formula_data = {}
    for formula in unique_formulas:
        formula_df = dataframe[dataframe["formula"] == formula]
        random.seed(42)
        examples = random.sample(formula_df["natural"].tolist(), sample_size)
        formula_data[formula] = ', '.join(examples)
    return formula_data

def get_contexts(data, model="gpt-4", max_tokens=512):
    """Generates context descriptions using OpenAI's chat completions API."""
    contexts = []
    for formula, examples in data.items():
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": ("You are a natural language to linear temporal language translator. "
                                "You are tasked with guiding a robot to navigate four rooms with colours, "
                                "red, blue, green, and yellow or any all colors synonymous to these. Given a "
                                "linear temporal instruction and examples to go with it, generate a context describing "
                                "that instruction in general terms.")
                },
                {
                    "role": "user",
                    "content": f"formula: '{formula}', examples: '{examples}'"
                }
            ],
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.4,
            presence_penalty=0
        )
        context = response.choices[0].message.content
        contexts.append({'formula': formula, 'context': context})
    return contexts

def main():
    """Main function to process data and generate contexts."""
    api_key = read_api_key("api_keys.json")
    openai.api_key = api_key

    file_path = '../data/clean_up.jsonl'
    df = load_data(file_path)

    train_df, _ = train_test_split(df, test_size=0.3, random_state=42)
    unique_formulas = train_df["formula"].unique()

    formula_data = create_formula_examples(train_df, unique_formulas, 5)
    assert formula_data, "Data processing failed"

    with open('natural_formula_5.json', 'w') as json_file:
        json.dump(formula_data, json_file, indent=4)

    contexts = get_contexts(formula_data)
    context_df = pd.DataFrame(contexts)
    context_df.to_csv("contexts_512.csv", index=False)

if __name__ == "__main__":
    main()
