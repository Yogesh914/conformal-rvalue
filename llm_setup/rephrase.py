import json
import os

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = os.environ.get("OPENAI_REPHRASE_MODEL", "gpt-4o-2024-08-06")


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running rephrase.py.")
    return OpenAI(api_key=api_key)


def rephrase_input_with_options(input_text, options, client, num_rephrases=20, retries=10):
    """
    Generate rephrased versions of the input text using the OpenAI chat completion model.
    Includes answer options for context and uses structured JSON output format.
    """
    options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert exam question rephraser. Your task is to:
                        1. Read the question and its answer options carefully.
                        2. Generate different rephrased versions of the question while maintaining relevance to the answer options.
                        3. Return the rephrased versions as a JSON object with a key 'rephrased_questions' containing a list of strings.
                        4. Return exactly the requested number of rephrased questions."""
                    },
                    {
                        "role": "user",
                        "content": f"""Rephrase this question while considering the following answer options:
                        
                        Question:
                        {input_text}
                        
                        Answer options:
                        {options_text}

                        Number of rephrased questions to return:
                        {num_rephrases}
                        """
                    }
                ],
                response_format={"type": "json_object"}
            )
            rephrased_json = json.loads(response.choices[0].message.content)
            rephrased_questions = rephrased_json.get("rephrased_questions", [])
            
            if isinstance(rephrased_questions, list) and len(rephrased_questions) == num_rephrases:
                return rephrased_questions
            raise ValueError(f"Expected {num_rephrases} rephrased questions in response JSON")
        
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return [None] * num_rephrases


def add_rephrased_inputs_to_split(dataset_split, client, num_rephrases=20):
    """
    Add a 'rephrased_inputs' column to a dataset split with progress tracking.
    """
    def rephrase_row(row):
        options = {"A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"]}
        row["rephrased_inputs"] = rephrase_input_with_options(
            row["input"],
            options,
            client=client,
            num_rephrases=num_rephrases,
        )
        return row

    return dataset_split.map(
        rephrase_row, 
        desc="Rephrasing rows", 
        batch_size=1
    )

def process_datasets(task_list, num_rephrases=20):
    """
    Load each dataset and its splits, and add the 'rephrased_inputs' feature with progress tracking.
    """
    client = get_openai_client()
    datasets = {}
    for task in tqdm(task_list, desc="Processing tasks", position=0):
        print(f"\nProcessing task: {task}")
        dataset = load_dataset("lukaemon/mmlu", task)
        
        for split in dataset.keys():
            print(f"  Processing split: {split}")
            dataset[split] = add_rephrased_inputs_to_split(
                dataset[split],
                client=client,
                num_rephrases=num_rephrases,
            )
        
        datasets[task] = dataset
    
    return datasets

# Task list
task_list = [
    "college_computer_science", "formal_logic", "high_school_computer_science",
    "computer_security", "machine_learning",
    "clinical_knowledge", "high_school_biology", "anatomy", "college_chemistry",
    "college_medicine", "professional_medicine",
    "business_ethics", "professional_accounting", "public_relations",
    "management", "marketing",
]


def main():
    processed_datasets = process_datasets(task_list, num_rephrases=20)
    for task, dataset in tqdm(processed_datasets.items(), desc="Saving datasets", position=0):
        dataset.save_to_disk(f"./data/{task}_with_rephrased_inputs")


if __name__ == "__main__":
    main()
    
    
    
    
    
# GPQA

# from datasets import load_dataset
# from tqdm import tqdm
# from openai import OpenAI
# import json
# import random

# # Initialize OpenAI API client
# client = get_openai_client()

# def rephrase_input_with_options(input_text, options, retries=10):
#     options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
#     for attempt in range(retries):
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-2024-08-06",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": """You are an expert exam question rephraser. Your task is to:
#                         1. Read the question and its answer options carefully.
#                         2. Generate 20 different rephrased versions of the question while maintaining relevance to the answer options.
#                         3. Return all 20 rephrased versions as a JSON object with a key 'rephrased_questions' containing a list of strings."""
#                     },
#                     {
#                         "role": "user",
#                         "content": f"""Rephrase this question while considering the following answer options:
                        
#                         Question:
#                         {input_text}
                        
#                         Answer options:
#                         {options_text}
#                         """
#                     }
#                 ],
#                 response_format={"type": "json_object"}
#             )
#             rephrased_json = json.loads(response.choices[0].message.content)
#             rephrased_questions = rephrased_json.get("rephrased_questions", [])
            
#             if isinstance(rephrased_questions, list) and len(rephrased_questions) == 20:
#                 return rephrased_questions
#             raise ValueError("Expected 20 rephrased questions in response JSON")
        
#         except Exception as e:
#             print(f"Error on attempt {attempt + 1}: {e}")
#             if attempt == retries - 1:
#                 return [None] * 20

# def rephrase_gpqa_row(row):
#     # Prepare the answer choices
#     choices = [
#         ("Correct Answer", row["Correct Answer"]),
#         ("Incorrect Answer 1", row["Incorrect Answer 1"]),
#         ("Incorrect Answer 2", row["Incorrect Answer 2"]),
#         ("Incorrect Answer 3", row["Incorrect Answer 3"])
#     ]
#     random.shuffle(choices)  # shuffle for this inference
    
#     labels = ['A', 'B', 'C', 'D']
#     options = dict(zip(labels, [choice[1] for choice in choices]))

#     # Track new correct label index if needed:
#     # correct_label = labels[[label for label, (k, _) in zip(labels, choices) if k == "Correct Answer"][0]]

#     row['rephrased_inputs'] = rephrase_input_with_options(row['Question'], options)
#     row['shuffled_options'] = options  # optional: store the options used
#     return row


# def process_gpqa_dataset():
#     # Load the dataset
#     data = load_dataset("Idavidrein/gpqa", 'gpqa_main', split='train')

#     data = data.remove_columns([
#         col for col in data.column_names
#         if col not in ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Subdomain']
#     ])

#     # Add rephrased inputs with progress bar
#     rephrased_dataset = data.map(rephrase_gpqa_row, desc="Rephrasing GPQA", batch_size=1)

#     return rephrased_dataset

# # Run it
# processed_gpqa = process_gpqa_dataset()

# # (Optional) Save to disk
# processed_gpqa.save_to_disk("./data/gpqa_main_with_rephrased_inputs")
