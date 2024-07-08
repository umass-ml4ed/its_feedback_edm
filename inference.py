import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, PeftModel, get_peft_model
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from ExperimentLogger import ExperimentLogger as el
import os
from nltk.translate.bleu_score import sentence_bleu

# TODO refactor me
class MathiaCompletionDataset(Dataset):
    # BUG: This max length could be a bug now with the new template
    def __init__(self, questions_df, responses_df, jinja_template_path, tokenizer, icl_pool_df_path, max_length=300):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        # load the dataset
        self.questions_df = questions_df
        self.responses_df = responses_df[responses_df['item_part'] == 'Expression-Dep']
        
        # Create a Jinja2 environment and specify the template file location
        env = Environment(loader=FileSystemLoader('./prompts'))

        # Load the template from the file
        template = env.get_template(jinja_template_path)

        # HACK change me to something else when you have a second to think
        if 'variable' in jinja_template_path:
            # Gather an example for each error type from the training data
            train_responses_df = pd.read_csv(icl_pool_df_path)
            train_responses_df = train_responses_df[train_responses_df['item_part'] == 'Expression-Dep']
            error_classes = train_responses_df['error_classes'].unique()
            error_response_map = {}
            for error_class in error_classes:
                error_response_map[error_class] = train_responses_df[train_responses_df['error_classes'] == error_class].iloc[0]
            # print(error_response_map)


        for _, row in self.responses_df.iterrows():
            # Render the template with the data, tokenize, and add to the dataset
            question = self.questions_df.loc[self.questions_df['item_id'] == row['item_id']].iloc[0]
            data = {
                'intro': question['intro'],
                'instructions': question['instructions'],
                'answer': row['answer'],
                'response': row['response'],
                'error': row['error_classes'],
            }
            if 'variable' in jinja_template_path:
                error_class = row['error_classes']
                if error_class not in error_response_map:
                    print(f"Error class {error_class} not in error_response_map. Defaulting to first response in train set")
                    ic_example_response = train_responses_df.iloc[0]
                else:
                    ic_example_response = error_response_map[error_class]
                ic_example_question = questions_df.loc[questions_df['item_id'] == ic_example_response['item_id']].iloc[0]
                # Maybe there's a cleaner, more scalable way to build this dict
                data['intro_ex_0'] = ic_example_question['intro']
                data['instructions_ex_0'] = ic_example_question['instructions']
                data['answer_ex_0'] = ic_example_response['answer']
                data['response_ex_0'] = ic_example_response['response']
                data['feedback_ex_0'] = ic_example_response['feedback']
                data['error_ex_0'] = ic_example_response['error_classes']
            text = template.render(data)
            encodings_dict = tokenizer(f'<s>[INST]{text}[/INST]', truncation=True, max_length=max_length, padding="max_length", add_special_tokens=False)

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# TODO Move to utils
def calculate_bleu(reference, candidate):
    reference = reference.lower().split()
    candidate = candidate.lower().split()
    return sentence_bleu([reference], candidate)

@hydra.main(version_base=None, config_path="conf", config_name="generate")
def generate(cfg: DictConfig):
    print("Config Dump:\n" + OmegaConf.to_yaml(cfg))
    wandb_hyrda_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    el.init_wandb(cfg.logger, wandb_hyrda_cfg)
    model_name = cfg.gen.model_name
    device = f"cuda:{cfg.gen.gpu_num}" if cfg.gen.gpu_num is not None else "cuda"
    print(f"Using device: {device}")

    RUN_NAME = el.get_run_name()
    print(f"Run name: {RUN_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    questions_df = pd.read_csv(cfg.gen.questions_path)
    val_responses_df = pd.read_csv(cfg.gen.val_responses_path)

    # TODO using Dataset class seems odd to me, maybe change it later
    BATCH_SIZE = cfg.gen.batch_size
    
    # TODO update this to the common Dataset and Dataloader defined in train.py
    val_dataset = MathiaCompletionDataset(questions_df, val_responses_df, cfg.template, tokenizer, cfg.gen.train_responses_path)
    completions_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if cfg.gen.lora_weights_path is not None:
        model = PeftModel.from_pretrained(model, cfg.gen.lora_weights_path).merge_and_unload()
        
    #Compare completions between Mistral and Lora
    model = model.to(device)
    completions = []
    sample_cfg = cfg.gen.sample
    for idx, (input_ids, attn_masks) in enumerate(tqdm(completions_dataloader)):
        input_length = input_ids[0].size()[0]
        max_length = input_length + sample_cfg.max_length
        input_ids = input_ids.to(device)
        outputs = model.generate(
                                input_ids,
                                attention_mask=attn_masks.to(device),
                                max_length=max_length,
                                num_return_sequences=1,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=True,
                                top_p=sample_cfg.top_p,
                                top_k=sample_cfg.top_k,
                                temperature=sample_cfg.temperature
                            )
        completion_batch = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        completions.extend(completion_batch)
        # Save to a running json/csv file
        outputs = pd.DataFrame({'completion': completions})

        postfix = cfg.gen.save_override if cfg.gen.save_override is not None else RUN_NAME
        template_file = os.path.basename(cfg.template).split('.')[0]
        out_folder = f'./lm_outputs/{cfg.gen.model_name}_{template_file}/'
        print(f'Saving completions to {out_folder}...')
        # Create an output folder if it doesn't exist
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        elif idx == 0:
            # Maybe this should be stronger like a breaking exception.
            print(f"WARNING: Folder {out_folder} already exists! Potentially overwriting files...")
        outputs.to_csv(f'{out_folder}{postfix}_completions.csv', index=False)
        outputs.to_json(f'{out_folder}{postfix}_completions.json', orient='records', indent=4)
    # Compute BLEU scores for the completions
    # TODO from here can be moved into a utils file
    item_ids = val_responses_df['item_id'].tolist()
    reference_text = val_responses_df['feedback'].tolist()
    # Post-process mistral to terminate completion at first "\n###"
    outputs['cleaned_completions'] = outputs['completion'].apply(lambda x: x.split("\n###")[0])
    completions = outputs['cleaned_completions'].tolist()
    bleu_scores = [calculate_bleu(ref, gen) for ref, gen in zip(reference_text, completions)]
    print(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores)}")
    print(f"Max BLEU score: {max(bleu_scores)}")
    print(f"Min BLEU score: {min(bleu_scores)}")
    el.log({'avg_bleu_score': sum(bleu_scores)/len(bleu_scores), 'max_bleu_score': max(bleu_scores), 'min_bleu_score': min(bleu_scores)})
    
    # Save the BLEU scores to a file
    print(f"Saving BLEU scores to {out_folder}{postfix}_bleu_scores.json...")
    bleu_df = pd.DataFrame({'completion': completions, 'bleu_score': bleu_scores, 'reference': reference_text, 'error_classes': val_responses_df['error_classes'].tolist(), 'answer':val_responses_df['answer'].tolist(), 'stu_response':val_responses_df['response'].tolist(), 'item_id': item_ids})
    bleu_df.to_csv(f'{out_folder}{postfix}_bleu_scores.csv', index=False)
    bleu_df.to_json(f'{out_folder}{postfix}_bleu_scores.json', orient='records', indent=4)
    if cfg.do_save:
        print(f"Writing to wandb")
        el.save_df_to_json(bleu_df, fileName=f'{postfix}_bleu_scores.json')
    el.finish_run()

if __name__ == "__main__":
    generate()
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, PeftModel, get_peft_model
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from ExperimentLogger import ExperimentLogger as el
import os
from nltk.translate.bleu_score import sentence_bleu

# TODO refactor me
class MathiaCompletionDataset(Dataset):
    # BUG: This max length could be a bug now with the new template
    def __init__(self, questions_df, responses_df, jinja_template_path, tokenizer, icl_pool_df_path, max_length=300):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        # load the dataset
        self.questions_df = questions_df
        self.responses_df = responses_df[responses_df['item_part'] == 'Expression-Dep']
        
        # Create a Jinja2 environment and specify the template file location
        env = Environment(loader=FileSystemLoader('./prompts'))

        # Load the template from the file
        template = env.get_template(jinja_template_path)

        # HACK change me to something else when you have a second to think
        if 'variable' in jinja_template_path:
            # Gather an example for each error type from the training data
            train_responses_df = pd.read_csv(icl_pool_df_path)
            train_responses_df = train_responses_df[train_responses_df['item_part'] == 'Expression-Dep']
            error_classes = train_responses_df['error_classes'].unique()
            error_response_map = {}
            for error_class in error_classes:
                error_response_map[error_class] = train_responses_df[train_responses_df['error_classes'] == error_class].iloc[0]
            # print(error_response_map)


        for _, row in self.responses_df.iterrows():
            # Render the template with the data, tokenize, and add to the dataset
            question = self.questions_df.loc[self.questions_df['item_id'] == row['item_id']].iloc[0]
            data = {
                'intro': question['intro'],
                'instructions': question['instructions'],
                'answer': row['answer'],
                'response': row['response'],
                'error': row['error_classes'],
            }
            if 'variable' in jinja_template_path:
                error_class = row['error_classes']
                if error_class not in error_response_map:
                    print(f"Error class {error_class} not in error_response_map. Defaulting to first response in train set")
                    ic_example_response = train_responses_df.iloc[0]
                else:
                    ic_example_response = error_response_map[error_class]
                ic_example_question = questions_df.loc[questions_df['item_id'] == ic_example_response['item_id']].iloc[0]
                # Maybe there's a cleaner, more scalable way to build this dict
                data['intro_ex_0'] = ic_example_question['intro']
                data['instructions_ex_0'] = ic_example_question['instructions']
                data['answer_ex_0'] = ic_example_response['answer']
                data['response_ex_0'] = ic_example_response['response']
                data['feedback_ex_0'] = ic_example_response['feedback']
                data['error_ex_0'] = ic_example_response['error_classes']
            text = template.render(data)
            encodings_dict = tokenizer(f'<s>[INST]{text}[/INST]', truncation=True, max_length=max_length, padding="max_length", add_special_tokens=False)

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# TODO Move to utils
def calculate_bleu(reference, candidate):
    reference = reference.lower().split()
    candidate = candidate.lower().split()
    return sentence_bleu([reference], candidate)

@hydra.main(version_base=None, config_path="conf", config_name="generate")
def generate(cfg: DictConfig):
    print("Config Dump:\n" + OmegaConf.to_yaml(cfg))
    wandb_hyrda_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    el.init_wandb(cfg.logger, wandb_hyrda_cfg)
    model_name = cfg.gen.model_name
    device = f"cuda:{cfg.gen.gpu_num}" if cfg.gen.gpu_num is not None else "cuda"
    print(f"Using device: {device}")

    RUN_NAME = el.get_run_name()
    print(f"Run name: {RUN_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    questions_df = pd.read_csv(cfg.gen.questions_path)
    val_responses_df = pd.read_csv(cfg.gen.val_responses_path)

    # TODO using Dataset class seems odd to me, maybe change it later
    BATCH_SIZE = cfg.gen.batch_size
    
    # TODO update this to the common Dataset and Dataloader defined in train.py
    val_dataset = MathiaCompletionDataset(questions_df, val_responses_df, cfg.template, tokenizer, cfg.gen.train_responses_path)
    completions_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if cfg.gen.lora_weights_path is not None:
        model = PeftModel.from_pretrained(model, cfg.gen.lora_weights_path).merge_and_unload()
        
    #Compare completions between Mistral and Lora
    model = model.to(device)
    completions = []
    sample_cfg = cfg.gen.sample
    for idx, (input_ids, attn_masks) in enumerate(tqdm(completions_dataloader)):
        input_length = input_ids[0].size()[0]
        max_length = input_length + sample_cfg.max_length
        input_ids = input_ids.to(device)
        outputs = model.generate(
                                input_ids,
                                attention_mask=attn_masks.to(device),
                                max_length=max_length,
                                num_return_sequences=1,
                                eos_token_id=tokenizer.eos_token_id,
                                do_sample=True,
                                top_p=sample_cfg.top_p,
                                top_k=sample_cfg.top_k,
                                temperature=sample_cfg.temperature
                            )
        completion_batch = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        completions.extend(completion_batch)
        # Save to a running json/csv file
        outputs = pd.DataFrame({'completion': completions})

        postfix = cfg.gen.save_override if cfg.gen.save_override is not None else RUN_NAME
        template_file = os.path.basename(cfg.template).split('.')[0]
        out_folder = f'./lm_outputs/{cfg.gen.model_name}_{template_file}/'
        print(f'Saving completions to {out_folder}...')
        # Create an output folder if it doesn't exist
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        elif idx == 0:
            # Maybe this should be stronger like a breaking exception.
            print(f"WARNING: Folder {out_folder} already exists! Potentially overwriting files...")
        outputs.to_csv(f'{out_folder}{postfix}_completions.csv', index=False)
        outputs.to_json(f'{out_folder}{postfix}_completions.json', orient='records', indent=4)
    # Compute BLEU scores for the completions
    # TODO from here can be moved into a utils file
    item_ids = val_responses_df['item_id'].tolist()
    reference_text = val_responses_df['feedback'].tolist()
    # Post-process mistral to terminate completion at first "\n###"
    outputs['cleaned_completions'] = outputs['completion'].apply(lambda x: x.split("\n###")[0])
    completions = outputs['cleaned_completions'].tolist()
    bleu_scores = [calculate_bleu(ref, gen) for ref, gen in zip(reference_text, completions)]
    print(f"Average BLEU score: {sum(bleu_scores)/len(bleu_scores)}")
    print(f"Max BLEU score: {max(bleu_scores)}")
    print(f"Min BLEU score: {min(bleu_scores)}")
    el.log({'avg_bleu_score': sum(bleu_scores)/len(bleu_scores), 'max_bleu_score': max(bleu_scores), 'min_bleu_score': min(bleu_scores)})
    
    # Save the BLEU scores to a file
    print(f"Saving BLEU scores to {out_folder}{postfix}_bleu_scores.json...")
    bleu_df = pd.DataFrame({'completion': completions, 'bleu_score': bleu_scores, 'reference': reference_text, 'error_classes': val_responses_df['error_classes'].tolist(), 'answer':val_responses_df['answer'].tolist(), 'stu_response':val_responses_df['response'].tolist(), 'item_id': item_ids})
    bleu_df.to_csv(f'{out_folder}{postfix}_bleu_scores.csv', index=False)
    bleu_df.to_json(f'{out_folder}{postfix}_bleu_scores.json', orient='records', indent=4)
    if cfg.do_save:
        print(f"Writing to wandb")
        el.save_df_to_json(bleu_df, fileName=f'{postfix}_bleu_scores.json')
    el.finish_run()

if __name__ == "__main__":
    generate()
