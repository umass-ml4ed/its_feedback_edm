# "Can Large Language Models Replicate ITS Feedback on Open-Ended Math Questions?"
Official code repository for "Can Large Language Models Replicate ITS Feedback on Open-Ended Math Questions?" EDM paper.

The code is provided for reference in reproducing the results described in this paper. Please note that the ITS dataset is proprietary and not included. Contact the authors (wmcnichols at umass dot edu) if you are interested.

# Overview
Our codebase is composed of three primary runnable python scripts. The rest of the files support the operations described below.

The file `train.py` runs the fine tuning process for local models (such as Mistral-7B). For our experiments we used a GPU with VRAM of 48GB so the default settings reflect such a hardware envionrment.

The file `inference.py` performs inference on the test split for the fine-tuned model and expects a similar hardware envionrment as above.

Lastly `promptOAI.py` uses the Open AI api to perform inference on fine-tuned and untrained Open-AI models on the test splits.

# Citation
If you found this project useful, please consider citing our work.


```
@misc{mcnichols2024largelanguagemodelsreplicate,
      title={Can Large Language Models Replicate ITS Feedback on Open-Ended Math Questions?}, 
      author={Hunter McNichols and Jaewook Lee and Stephen Fancsali and Steve Ritter and Andrew Lan},
      year={2024},
      eprint={2405.06414},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.06414}, 
}
```