import prompt_questions as p
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from collections import defaultdict
import pickle
import os


def get_prompt(task_data, task, question_num=0, prompt_q=None):
    '''
    task_data:
    Question num specifies which question will be used as prompt.
    If prompt_q is provided, it is used as 1-shot prompt question. This
    corresponds to GPT-4 based question prompts that we created. Else, we
    select question corresponding to question_num from the MMLU itself to
    generate the prompt. We select prompt from test set in this case,
    since train set is very small sometime and may not have 10 samples.
    We use 10 different prompts and take avergae over them to estimate
    performance on a subject. The function returns the 1-shot question prompt.
    '''

    if prompt_q is None:
        prompt_set = 'test'
        if question_num > len(task_data['test']['input']) - 1:
            print('prompt question id exceeds the length of test set')
            print('selecting last question of the test set')
            question_num = len(task_data['test']['input']) - 1
        prompt_add = f'This is a question from {task.replace("_", " ")}.\n'
        prompt_add += f"{task_data[prompt_set]['input'][question_num]}\n"
        for letter in ['A', 'B', 'C', 'D']:
            prompt_add += '    ' + letter + '. ' + task_data[prompt_set][letter][question_num] + '\n'
        prompt_add += f"The correct answer is option: {task_data[prompt_set]['target'][question_num]}\n"
    else:
        prompt_add = f'This is a question from {task.replace("_", " ")}.'
        prompt_add += prompt_q
        prompt_add += '\n'
    prompt_add += f"You are the world's best expert in {task.replace('_', ' ')}. "
    prompt_add += '''Reason step-by-step and answer the following question. '''
    return prompt_add


def get_question_dict(task_data, prompt_add, prompt_q_id=None):
    questions = []
    answers = []
    splits = ['train', 'validation', 'test']
    splits = ['train']
    
    for split in splits:
        if split == 'train':
            start = 1
        else:
            start = 0
        for i in range(start, len(task_data[split]['input'])):
            if split == 'test' and prompt_q_id is not None:
                if i == prompt_q_id:
                    continue
                    
            # Create a list to store all versions of this question (original + rephrased)
            question_versions = []
            
            # Process original question
            question_dict = {}
            prompt_q = prompt_add + task_data[split]['input'][i] + '\n'
            for letter in ['A', 'B', 'C', 'D']:
                prompt_q += '(' + letter + ') ' + task_data[split][letter][i] + ' '
            prompt_q += "\nThe correct answer is option: "
            for letter in ['A', 'B', 'C', 'D']:
                question_dict[letter] = prompt_q + letter
            question_versions.append(question_dict)
            
            # Process rephrased questions
            rephrased_inputs = task_data[split]['rephrased_inputs'][i]
            for rephrased_input in rephrased_inputs:
                if rephrased_input is not None:  # Skip None values
                    question_dict = {}
                    prompt_q = prompt_add + rephrased_input + '\n'
                    for letter in ['A', 'B', 'C', 'D']:
                        prompt_q += '(' + letter + ') ' + task_data[split][letter][i] + ' '
                    prompt_q += "\nThe correct answer is option: "
                    for letter in ['A', 'B', 'C', 'D']:
                        question_dict[letter] = prompt_q + letter
                    question_versions.append(question_dict)
            
            questions.append(question_versions)
            answers.append(task_data[split]['target'][i])
            
    return questions, answers


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    '''
    Takes model, tokenizer and input_texts corresponding to each of the choices
    to do a forward pass through the model.
    Returns both raw logits and log-softmax scores as tuples
    '''
    all_outputs = []
    all_input_ids = []
    for text in input_texts:
        input_ids = tokenizer(text, padding=True, return_tensors="pt").input_ids.to("cuda")
        outputs = model(input_ids)
        logits = outputs.logits.detach().cpu()
        all_outputs.append(logits)
        all_input_ids.append(input_ids.detach().cpu())
        del outputs, input_ids
        torch.cuda.empty_cache()

    all_outputs = torch.concat(all_outputs, 0)[:, -2:-1, :]  # Take logit corresponding to option token
    all_input_ids = torch.concat(all_input_ids, 0)[:, -1:]  # Include token id for options
    raw_logits = all_outputs.clone()  # Store raw logits before softmax
    probs = torch.log_softmax(all_outputs.float(), dim=-1).detach().cpu()
    torch.cuda.empty_cache()

    gen_probs = torch.gather(probs, 2, all_input_ids[:, :, None]).squeeze(-1)
    gen_logits = torch.gather(raw_logits, 2, all_input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs, input_logits in zip(all_input_ids[:, 0], gen_probs[:, 0], gen_logits[:, 0]):
        batch.append((tokenizer.decode(input_sentence), input_probs.item(), input_logits.item()))
    return batch


def softmax(logits):
    '''
    converts log-softmax scores to probablities.
    '''
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits)
    probabilities = exp_logits / sum_exp_logits
    return probabilities


def extract_answer(batch):
    '''
    Converts the batch of option, log-softmax score, and logit tuples to option, probability, and logit tuples
    '''
    probabilities = softmax(np.array([answer[1] for answer in batch]))
    logits = np.array([answer[2] for answer in batch])

    output_with_probabilities_and_logits = [(batch[i][0], probabilities[i], logits[i]) for i in range(len(batch))]
    return output_with_probabilities_and_logits


def average_question_predictions(prediction_list):
    '''
    Calculates the average of the probability for question-option pairs by avergaing the
    probability across prompts.
    '''
    num_seeds = len(prediction_list)  # Number of random seeds (or runs)
    average_list = []  # List to store the average predictions for each question

    # Iterate through each question
    for question_idx in range(len(prediction_list[0])):
        # Initialize a dictionary to store the sums of probabilities for each option
        option_sums = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        # Iterate through each random seed
        for seed_idx in range(num_seeds):
            # Iterate through each option and its probability for the current question and seed
            for option, value in prediction_list[seed_idx][question_idx]:
                # Add the probability to the corresponding option sum
                option_sums[option] += value

        # Calculate the average probability for each option and store them as tuples
        option_averages = [(key, value / num_seeds) for key, value in option_sums.items()]
        # Add the average probabilities for the current question to the list
        average_list.append(option_averages)

    return average_list


def accuracy(predicted_probs, correct_answers):
    total_count = len(correct_answers)
    assert len(correct_answers) == len(predicted_probs)
    correct_count = 0

    for i in range(total_count):
        # Average the predictions across all versions of the question
        averaged_probs = defaultdict(float)
        num_versions = len(predicted_probs[i])
        for version_pred in predicted_probs[i]:
            for option, prob in version_pred:
                averaged_probs[option] += prob / num_versions
        
        # Find the answer with the maximum average probability
        max_prob_answer = max(averaged_probs.items(), key=lambda x: x[1])[0].strip()
        if correct_answers[i] == max_prob_answer:
            correct_count += 1.0

    return correct_count / total_count


def get_max_size_prompt_len(task_data, task, n=10, max_allowed_prompt_len=700):
    '''
    get the size of maximum length prompt out of all n prompts considered.
    '''
    max_len = 0
    i = 0
    prompt_question_ids = []
    while len(prompt_question_ids) < n:
        prompt_add = get_prompt(task_data, task=task, question_num=i)
        prompt_len = len(prompt_add)

        if prompt_len > max_allowed_prompt_len:
            i += 1
            continue
        else:
            prompt_question_ids.append(i)
            i += 1

        if prompt_len > max_len:
            max_len = prompt_len
    return max_len, prompt_question_ids


def get_predictions_over_n_runs(model, tokenizer, task_data, prompt_q_list, task):
    predictions_list = []
    logits_list = []
    acc_list = []

    for j, prompt_q in enumerate(tqdm(prompt_q_list, desc=f"Processing prompts for {task}")):
        prompt_add = get_prompt(task_data, task=task, prompt_q=prompt_q)
        questions, solution_answers = get_question_dict(task_data, prompt_add=prompt_add)
        predictions = []
        logits = []
        targets = []
        
        for question_versions, answer in tqdm(zip(questions, solution_answers), 
                                            desc=f"Processing questions for prompt", 
                                            total=len(questions)):
            version_predictions = []
            version_logits = []
            for question in tqdm(question_versions, 
                               desc=f"Processing versions for q{len(predictions)}", 
                               leave=False):
                batch = to_tokens_and_logprobs(model, tokenizer, [v for v in question.values()])
                torch.cuda.empty_cache()
                results = extract_answer(batch)
                # Split probabilities and logits
                version_predictions.append([(r[0], r[1]) for r in results])
                version_logits.append([(r[0], r[2]) for r in results])
            predictions.append(version_predictions)
            logits.append(version_logits)
            targets.append(answer)
            
        acc = round(accuracy(predictions, targets), 3)
        print(f'Accuracy on {task} for iteration {j} is {acc:.2f} ')
        acc_list.append(acc)
        predictions_list.append(predictions)
        logits_list.append(logits)
    return predictions_list, logits_list, solution_answers, acc_list


def get_prediction_list(model, tokenizer, subject_name, prompt_list):
    # Load the dataset with rephrased inputs
    task_data = load_from_disk(f"../data/{subject_name}_with_rephrased_inputs")
    prediction_lists, logits_list, solution_answers, avg_acc = get_predictions_over_n_runs(model, tokenizer, task_data,
                                                                   prompt_list, subject_name)
    return prediction_lists, logits_list, solution_answers, avg_acc

    
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluations on a specific split of tasks')
    parser.add_argument('--split_number', type=int, required=True, help='Split number to process')
    parser.add_argument('--total_splits', type=int, required=True, help='Total number of splits')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    part_identifier = f"part{args.split_number}"
    
    # List of all tasks
    task_list = ['college_computer_science', 'formal_logic', 'high_school_computer_science',
                 'computer_security', 'machine_learning',
                 'clinical_knowledge', 'high_school_biology', 'anatomy', 'college_chemistry',
                 'college_medicine', 'professional_medicine',
                 'business_ethics', 'professional_accounting', 'public_relations',
                 'management', 'marketing']
    
    task_list = ["biology", "physics", "chemistry"]
    # Calculate split indices
    tasks_per_split = len(task_list) // args.total_splits
    start_idx = args.split_number * tasks_per_split
    end_idx = start_idx + tasks_per_split if args.split_number < args.total_splits - 1 else len(task_list)
    
    # Get tasks for this split
    split_tasks = task_list[start_idx:end_idx]
    
    # Import GPT-4 based question prompts
    prompt_list = [
        [p.prompt_q_list_college_cs[0]],
        [p.prompt_q_list_formal_logic[0]], 
        [p.prompt_q_list_high_school_cs[0]],
        [p.prompt_q_list_computer_security[0]], 
        [p.prompt_q_list_machine_learning[0]],
        [p.prompt_q_list_clinical_knowledge[0]], 
        [p.prompt_q_list_high_school_bio[0]], 
        [p.prompt_q_list_anatomy[0]],
        [p.promtp_q_list_college_chemistry[0]], 
        [p.prompt_q_list_college_medicine[0]],
        [p.prompt_q_list_professional_medicine[0]],
        [p.prompt_q_list_business_ethics[0]], 
        [p.prompt_q_list_professional_accounting[0]], 
        [p.prompt_q_list_pr[0]],
        [p.prompt_q_list_management[0]], 
        [p.prompt_q_list_marketing[0]]
    ]
    prompt_list = [[p.prompt_q_list_high_school_bio[0]], [p.prompt_q_list_professional_medicine[0]], [p.promtp_q_list_college_chemistry[0]], ]
    split_prompts = prompt_list[start_idx:end_idx]
    
    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    # if args.split_number == 0:
    #     model.half().cuda()
    # else:
    #     model.cuda()
        
    # Process tasks in this split
    acc_dicts_mmlu = {}
    for task, prompt in tqdm(zip(split_tasks, split_prompts), desc=f"Processing split {args.split_number} tasks"):
        prediction_lists, logits_lists, solution_answers, acc_list = get_prediction_list(model, tokenizer, task, prompt)
        avg_acc = np.mean(np.array(acc_list))
        print(f'Average accuracy on {task} is {avg_acc:.3f}')
        acc_dicts_mmlu[task] = acc_list
        
        # Save results
        # with open(os.path.join(args.output_dir, f"accuracy_gpt_prompts_10_{part_identifier}.pkl"), "wb") as f:
        #     pickle.dump(acc_dicts_mmlu, f)
        
        # Save probabilities
        scores = np.array([[[[a[1] for a in version_pred] for version_pred in pred_versions] 
                           for pred_versions in predictions] 
                          for predictions in prediction_lists])
        
        # Save logits
        logits = np.array([[[[a[1] for a in version_logit] for version_logit in logit_versions] 
                           for logit_versions in logits_batch] 
                          for logits_batch in logits_lists])
        
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        targets = np.array(list(map(lambda x: answer_map[x], solution_answers)))
        
        np.save(os.path.join(args.output_dir, f'{task}_escors.npy'), scores)
        np.save(os.path.join(args.output_dir, f'{task}_logits.npy'), logits)
        np.save(os.path.join(args.output_dir, f'{task}_targets.npy'), targets)

if __name__ == "__main__":
    main()