import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, PeftModel, get_peft_model
import math
from ExperimentLogger import ExperimentLogger as el
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

# TODO is this necessary? Can there be one class for both that lives somewhere else?
class MathiaResponseDataset(Dataset):
    def __init__(self, questions_df, responses_df, jinja_template_path):
        self.prompts = []
        self.completions = []
        # load the dataset
        self.questions_df = questions_df
        self.responses_df = responses_df[responses_df['item_part'] == 'Expression-Dep']
        
        # Create a Jinja2 environment and specify the template file location
        env = Environment(loader=FileSystemLoader('./prompts'))

        # Load the template from the file
        template = env.get_template(jinja_template_path)

        # TODO to make this faster we need to use batch tokenization
        for _, row in self.responses_df.iterrows():
            # Render the template with the data, tokenize, and add to the dataset
            question = self.questions_df.loc[self.questions_df['item_id'] == row['item_id']].iloc[0]
            data = {
                'intro': question['intro'],
                'instructions': question['instructions'],
                'answer': row['answer'],
                'response': row['response'],
                'feedback': row['feedback'],
                'error': row['error_classes'],
            }
            prompt = template.render(data)
            formatted_prompt = f"[INST]{prompt}[/INST]"
            self.prompts.append(formatted_prompt)
            self.completions.append(row['feedback'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (self.prompts[idx], self.completions[idx])
    
class MathiaResponseDataCollator:
    def __init__(self, tokenizer, inference=False) -> None:
        self.tokenizer = tokenizer;
        self.inference = inference

    def __call__(self, batch):
        all_prompts = [sample[0] for sample in batch]
        # TODO maxlength?
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        if self.inference:
            return {
                "input_ids": prompts_tokenized.input_ids,
                "attention_mask": prompts_tokenized.attention_mask,
            }
        else: # training
            batch_inputs = [sample[0] + sample[1] for sample in batch] # 0 is prompt, 1 is completion
            # TODO double check the eos and sos tokens here
            inputs_tokenized = self.tokenizer(batch_inputs, return_tensors="pt", padding=True)
            prompt_lens = prompts_tokenized.attention_mask.sum(dim=1)
            labels = inputs_tokenized.input_ids.clone()
            padding_mask = torch.arange(labels.shape[1]).repeat(labels.shape[0], 1) < prompt_lens.unsqueeze(1)
            labels[padding_mask] = -100
            labels = labels.masked_fill(inputs_tokenized.attention_mask == 0, -100)
            return {
                "input_ids": inputs_tokenized.input_ids,
                "attention_mask": inputs_tokenized.attention_mask,
                "labels": labels
            }
    
def validation(model, device, val_dataloader):
    model.eval() # set to eval mode
    total_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            input_ids, attn_masks, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"]
            # move tensors to GPU if CUDA is available
            input_ids = input_ids.to(device)
            attn_masks = attn_masks.to(device) 
            # forward pass and compute loss
            outputs = model(input_ids, labels=labels, attention_mask=attn_masks)  
            loss = outputs[0]  
            # backward pass and update gradients
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

def compute_intermediate_val(batch_num, total_batches, epoch, model, device, val_dataloader, total_train_loss, train_cfg):
    # Since this trains slowly we report metrics at a fixed percentage per epoch
    if train_cfg.eval_frac is None:
        return
    end_ratio = 1 - train_cfg.eval_frac
    if (batch_num + 1) % math.ceil(train_cfg.eval_frac * total_batches) == 0 \
          and batch_num < math.ceil(end_ratio * total_batches):
        percent_complete = math.ceil(((batch_num + 1) / total_batches) * 100)
        print(f"{percent_complete}% of epoch traversed, Computing Validation")
        avg_val_loss = validation(model, device, val_dataloader)
        print(f'Epoch {epoch} and {percent_complete}% - Avg Val Loss: {avg_val_loss:.4f} - Avg Train Loss: {total_train_loss/(batch_num+1):.4f}')
        el.log({'avg_val_loss': avg_val_loss, 'avg_train_loss': total_train_loss/(batch_num+1)})
        model.train()

def fine_tune(train_dataset: Dataset, val_dataset, train_cfg):
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(train_cfg.model_name) # MistralForCausalLM
    # TODO worth tweaking these, see: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
    peft_config = LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r=train_cfg.lora_r,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            task_type="CAUSAL_LM",
            inference_mode=False,
        )   
    model = get_peft_model(base_model, peft_config)

    optimizer = AdamW(model.parameters(), lr=train_cfg.lr) # AdamW optimizer
    # TODO scheduler

    device = f"cuda:{train_cfg.gpu_num}" if train_cfg.gpu_num is not None else "cuda"
    print(f"Sending model to:{device}")
    model = model.to(device)

    # Llama specific tuning, will later be refactored into a TokenizerFactory if needed
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name, use_fast=True, add_eos_token=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token_id = tokenizer.eos_token_id # source: https://www.reddit.com/r/LocalLLaMA/comments/15hz7gl/my_finetuning_based_on_llama27bchathf_model

    # Create data loader
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.train_batch_size, shuffle=True, collate_fn=MathiaResponseDataCollator(tokenizer, inference=False))
    val_dataloader = DataLoader(val_dataset, batch_size=train_cfg.val_batch_size, shuffle=True, collate_fn=MathiaResponseDataCollator(tokenizer, inference=False))

    if train_cfg.init_eval:
        print("Computing initial loss for validation set...")
        avg_val_loss = validation( model, device, val_dataloader)
        print(f"Average initial loss for validation set is {avg_val_loss}.")
        el.log({'avg_val_loss': avg_val_loss})


    min_val_loss = float('inf')
    model.train() # set to train mode
    for epoch in range(train_cfg.num_epochs):
        print(f'Starting Epoch {epoch}')
        total_train_loss = 0.0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids, attn_masks, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"]
            # forward pass and compute loss
            outputs = model(input_ids, labels=labels, attention_mask=attn_masks)  
            loss = outputs[0]  
            # backward pass and update gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            compute_intermediate_val(idx, len(train_dataloader), epoch, model, device, val_dataloader, total_train_loss, train_cfg)
        val_loss = validation(model, device, val_dataloader)
        model.train()
        print(f'Epoch {epoch} completed. Average Validation Loss: {val_loss}. Average Training Loss: {total_train_loss/(idx+1)}')
        el.log({'avg_train_loss': total_train_loss/(idx+1), 'avg_val_loss': val_loss})
        # TODO depending on the results of the experiments we might need this to be done on fraction of epoch as well
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print(f'New best model found. Saving model...')
            postfix = train_cfg.save_override if train_cfg.save_override is not None else el.get_run_name()
            output_dir = f'./saved_models/{train_cfg.model_name}_{postfix}/'
            print(f'Saving model to {output_dir}')
            model.save_pretrained(output_dir)
            if el.cfg.save_model:
                el.save_model_wanb(model, f"{train_cfg.model_name}_{postfix}")

@hydra.main(version_base=None, config_path="conf", config_name="fineTune")
def main(cfg: DictConfig):
    print("Config Dump:\n" + OmegaConf.to_yaml(cfg))
    wandb_hyrda_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    el.init_wandb(cfg.logger, wandb_hyrda_cfg)
    
    # load the dataset
    questions_df = pd.read_csv(cfg.train.questions_path)
    train_responses_df = pd.read_csv(cfg.train.train_responses_path)
    val_responses_df = pd.read_csv(cfg.train.val_responses_path)

    print("Building datasets...")
    train_dataset = MathiaResponseDataset(questions_df, train_responses_df, cfg.template)
    val_dataset = MathiaResponseDataset(questions_df, val_responses_df, cfg.template)

    fine_tune(train_dataset, val_dataset, cfg.train)

if __name__ == '__main__':
    main()