import wandb
import os
import torch
from datetime import datetime

class ExperimentLogger():
    verbose = True
    wandb_run_name = None
    wandb_run = None
    metrics_output_dir = "./analysis/"
    model_output_dir = "./saved_models/"
    cfg = None
    
    @classmethod
    def init_wandb(cls, logger_conf, hydra_cfg, group=None, reinit=False):
        """Expecting wand_group_id, project, entity, and cfg in dictionary"""
        ExperimentLogger.cfg = logger_conf
        if not logger_conf.record:
            print("Record is set to false, not logging to wandb")
            return None
        try:
            if getattr(logger_conf, 'group', None) is None:
                ExperimentLogger.wandb_run = wandb.init(project=logger_conf.project,
                                                        entity=logger_conf.entity,
                                                        config=hydra_cfg,
                                                        reinit=reinit)
            else:
                ExperimentLogger.wandb_run = wandb.init(group=logger_conf.group,
                                                        project=logger_conf.project,
                                                        entity=logger_conf.entity,
                                                        config=hydra_cfg,
                                                        reinit=reinit)
        except:
            raise Exception("Exception occurred initializing Hydra. Probably, missing values from hydra config, check your setup")
        
    @classmethod
    def log(cls, data_dict):
        """
        Logs to the remote logging session
        """
        if ExperimentLogger.wandb_run is None:
            print("Logging locally and NOT to wandb")
            print(data_dict)
            return
        try:
            ExperimentLogger.wandb_run.log(data_dict)
        except:
            raise Exception("You need to initialize a remote logger before you start logging") 
        if ExperimentLogger.verbose:
            print(data_dict)

    @classmethod
    def write_table_wandb(cls, dataFrame, fileName=None, prependRunName=True):
        """
        Converts the dataframe to a csv file and writes it to the wandb run directory
        """
        if ExperimentLogger.wandb_run is None:
            print("Saving locally and NOT to wandb")
            if fileName is None:
                fileName = f'results_table_{datetime.now().strftime("%m_%d_%H_%M")}.csv'
            dataFrame.to_csv(os.path.join(ExperimentLogger.metrics_output_dir, fileName))
            return
        if fileName is None:
            fileName = f'results_table_{cls.wandb_run.name}.csv'
        try:
            if prependRunName:
                fileName = f"{cls.wandb_run.name}_{fileName}"
            print(f"Writing file to: {os.path.join(wandb.run.dir, fileName)}")
            dataFrame.to_csv(os.path.join(wandb.run.dir, fileName))
        except:
            raise Exception("Error when writing table to wandb. Maybe you need to initialize a remote logger?") 
        
    @classmethod
    def get_run_name(cls):
        if ExperimentLogger.wandb_run is None:
            return f"local_run_{datetime.now().strftime('%m_%d_%H_%M')}"
        return ExperimentLogger.wandb_run.name
        
    @classmethod
    def save_model_wanb(cls, model, fileName=None):
        """
        Saves the model to the wandb run directory
        """
        if ExperimentLogger.wandb_run is None:
            print("Saving locally and NOT to wandb")
            if fileName is None:
                fileName = f'final_model_{datetime.now().strftime("%m_%d_%H_%M")}.pt'
            else:
                fileName = f'{fileName}.pt'
            output_dir = os.path.join(ExperimentLogger.model_output_dir, fileName)
            torch.save(model, output_dir)
            return output_dir
        if fileName is None:
            fileName = f'final_model_{cls.wandb_run.name}.pt'
        try:
            fileName = f"{cls.wandb_run.name}_{fileName}.pt"
            output_dir = os.path.join(wandb.run.dir, fileName)
            os.makedirs(output_dir, exist_ok = True)
            # TODO this code needs to be updated to handle LoRA saves
            torch.save(model, output_dir)
            return output_dir
        except:
            raise Exception("Error when saving model to wandb. Maybe you need to initialize a remote logger?")

    @classmethod
    def save_df_to_json(cls, dataFrame, fileName=None):
        """
        Saves the dataframe to the wandb run directory
        """
        if ExperimentLogger.wandb_run is None:
            print("Logging locally and NOT to wandb")
            if fileName is None:
                fileName = f'results_{datetime.now().strftime("%m_%d_%H_%M")}.json'
            dataFrame.to_json(os.path.join(ExperimentLogger.metrics_output_dir, fileName), orient='records', indent=4)
            return
        if fileName is None:
            fileName = f'results_{cls.wandb_run.name}.json'
        try:
            dataFrame.to_json(os.path.join(wandb.run.dir, fileName), orient='records', indent=4)
        except:
            raise Exception("Error when saving model to wandb. Maybe you need to initialize a remote logger?")
        
    @classmethod
    def get_run_name(cls):
        if ExperimentLogger.wandb_run is None:
            return f"local_run_{datetime.now().strftime('%m_%d_%H_%M')}"
        return ExperimentLogger.wandb_run.name
    
    @classmethod
    def finish_run(cls):
        if ExperimentLogger.wandb_run is None:
            return
        ExperimentLogger.wandb_run.finish()
        ExperimentLogger.wandb_run = None
        ExperimentLogger.wandb_run_name = None