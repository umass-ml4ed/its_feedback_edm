import pandas as pd
from jinja2 import Environment, FileSystemLoader
from OpenAIInterface import OpenAIInterface as oAI
import hydra
from omegaconf import DictConfig, OmegaConf
from nltk.translate.bleu_score import sentence_bleu
from ExperimentLogger import ExperimentLogger as el
import os

# TODO Move to utils
def calculate_bleu(reference, candidate):
    reference = reference.lower().split()
    candidate = candidate.lower().split()
    return sentence_bleu([reference], candidate)


@hydra.main(version_base=None, config_path="conf", config_name="oAI")
def main(cfg: DictConfig):
    print("Config Dump:\n" + OmegaConf.to_yaml(cfg))
    wandb_hyrda_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    el.init_wandb(cfg.logger, wandb_hyrda_cfg)
    questions_filepath = cfg.openAI.question_path
    responses_filepath = cfg.openAI.response_path
    jinja_template_path = cfg.openAI.template   
    questions_df = pd.read_csv(questions_filepath)
    responses_df = pd.read_csv(responses_filepath)
    responses_df = responses_df[responses_df['item_part'] == 'Expression-Dep']

    # Gather an example for each error type from the training data
    train_responses_df = pd.read_csv('sanity_checks/train_responses_debug.csv')
    train_responses_df = train_responses_df[train_responses_df['item_part'] == 'Expression-Dep']
    error_classes = train_responses_df['error_classes'].unique()
    error_response_map = {}
    for error_class in error_classes:
        error_response_map[error_class] = train_responses_df[train_responses_df['error_classes'] == error_class].iloc[0]
    # print(error_response_map)

    # Create a Jinja2 environment and specify the template file location
    env = Environment(loader=FileSystemLoader('./prompts'))

    # Load the template from the file
    template = env.get_template(jinja_template_path)

    # For testing new things
    # responses_df = responses_df.head(20)

    prompts = []
    for _, row in responses_df.iterrows():
        # Render the template with the data, tokenize, and add to the dataset
        question = questions_df.loc[questions_df['item_id'] == row['item_id']].iloc[0]
        data = {
            'intro': question['intro'],
            'instructions': question['instructions'],
            'answer': row['answer'],
            'response': row['response'],
            'error': row['error_classes'],
        }
        if 'variable' in cfg.openAI.template:
            error_class = row['error_classes']
            ic_example_response = error_response_map[error_class]
            ic_example_question = questions_df.loc[questions_df['item_id'] == ic_example_response['item_id']].iloc[0]
            # Maybe there's a cleaner, more scalable way to build this dict
            data['intro_ex_0'] = ic_example_question['intro']
            data['instructions_ex_0'] = ic_example_question['instructions']
            data['answer_ex_0'] = ic_example_response['answer']
            data['response_ex_0'] = ic_example_response['response']
            data['feedback_ex_0'] = ic_example_response['feedback']
            data['error_ex_0'] = ic_example_response['error_classes']

        prompt = template.render(data)
        # Write all to a txt file
        prompts.append(prompt)

    prompt_responses = oAI.getCompletionForAllPrompts(cfg.openAI, prompts, batch_size=20, use_parallel=True)
    completions = [response.message.content for response in prompt_responses] if 'instruct' not in cfg.openAI.model else [response.text for response in prompt_responses]
    completions_df = pd.DataFrame({'completion':completions, 'prompt':prompts})
    
    #Temporary adds for human eval setup, adds question text and completions to responses_df
    # question_texts = [intro+"\n"+instructions for intro,instructions in zip(questions_df['intro'], questions_df['instructions'])]
    # responses_df['generated_fbs'] = completions
    # # extract question text for each row based on item_id (1-indexed where list is 0-indexed)
    # responses_df['question_texts'] = responses_df['item_id'].map(lambda x: question_texts[x-1])
    
    # responses_df.to_csv(f'splits/human_eval_downsample_v0_aug.csv', index=False)
    # responses_df.to_json(f'splits/human_eval_downsample_v0_aug.json', orient='records', indent=4)


    template_file = os.path.basename(cfg.openAI.template).split('.')[0]
    out_folder = f"./lm_outputs/open_ai/{cfg.openAI.model}_{template_file}/"
    print(f'Saving completions to {out_folder}...')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        # TODO think how should we do this instead?
        print(f"WARNING: Folder {out_folder} already exists! Potentially overwriting files...")
    # Save the BLEU scores to a file
    postfix = el.get_run_name()
    completions_df.to_json(f"{out_folder}{postfix}_completions.json", orient='records', indent=4)
    
    completions = completions_df['completion'].tolist()
    item_ids = responses_df['item_id'].tolist()
    reference_text = responses_df['feedback'].tolist()
    bleu_scores = [calculate_bleu(ref, gen) for ref, gen in zip(reference_text, completions)]
    print(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores)}")
    print(f"Max BLEU score: {max(bleu_scores)}")
    print(f"Min BLEU score: {min(bleu_scores)}")
    el.log({'avg_bleu_score': sum(bleu_scores)/len(bleu_scores), 'max_bleu_score': max(bleu_scores), 'min_bleu_score': min(bleu_scores)})
    
    print(f"Saving BLEU scores to {out_folder}{postfix}_bleu_scores.json...")
    bleu_df = pd.DataFrame({'completion': completions, 'bleu_score': bleu_scores, 'reference': reference_text, 'error_classes': responses_df['error_classes'].tolist(), 'answer':responses_df['answer'].tolist(), 'stu_response':responses_df['response'].tolist(), 'item_id': item_ids})
    bleu_df.to_csv(f'{out_folder}{postfix}_bleu_scores.csv', index=False)
    bleu_df.to_json(f'{out_folder}{postfix}_bleu_scores.json', orient='records', indent=4)
    if cfg.do_save:
        print(f"Writing to wandb")
        el.save_df_to_json(bleu_df, fileName=f'{postfix}_bleu_scores.json')
    el.finish_run()
    
if __name__ == "__main__":
    main()