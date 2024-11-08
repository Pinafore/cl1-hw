from functools import partial
from collections import defaultdict

from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer

from buzzer import Buzzer

def initialize_base_model(helper_function=AutoModelForSequenceClassification,
                          model_name="distilbert-base-uncased"):
    """
    Initialize a BERT model and corresponding tokenizer.

    Args:
        helper_function: The huggingface function that returns a BERT model.
        model_name: The name of the BERT model to use.
    """

    model = helper_function.from_pretrained(model_name, num_labels=2)

    # Freeze the model parameters

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        """
        Initialize a LoRA with two weight matrices whose product is the same as the original parameter matrix.
        """
        super().__init__()

        self.A = None
        self.B = None
        self.alpha = 0

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Complete the initialization of the two weight matrices
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the linear layer's original result, then add the low-rank delta
        """
        assert x.shape[-1] == self.in_dim, "Input dimension %s does not match input dimension %i" % (str(x.shape), self.in_dim)

        if len(x.shape) == 1:
            delta = torch.zeros(self.out_dim)
            output_dimension = torch.Size([self.out_dim])
        else:
            delta = torch.zeros((x.shape[0], self.in_dim))
            output_dimension = torch.Size((x.shape[0], self.out_dim))

        # Compute the low-rank delta

        assert delta.shape == output_dimension, "Delta size %s inconsistent with output dimension %i" % (str(delta.shape), self.out_dim)
        return delta


class LinearLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: float):
        """
        Initialize a Linear layer with LoRA adaptation.
        """
        super().__init__()
        self.linear = linear

        # Initialize the LoRA layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adatpation.
        """
        result = self.linear(x)

        # Add the LoRA delta
        return result

# TODO(jbg): Get rid of the hardcoded modules so that it generalizes to other models
def add_lora(model: torch.nn.Module, rank: int, alpha: float, 
             modules_to_adapt: Dict[str, List[str]] = {"attention": ["q_lin", "k_lin", "v_lin", "out_lin"], "ffn": ["lin1", "lin2"]}):
    """
    Add LoRA layers to a PyTorch model.  

    Args:
        model: The PyTorch model to adapt.
        rank: The rank of the LoRA matrices.
        alpha: The scaling factor for the LoRA matrices.
        modules_to_adapt: The key of the dictionary is the model component to adapt (e.g., "attention" or "ffn"), and the values are specific linear layers in that component to adapt.  Anything in this dictionary will be adapted, but anything else will remain frozen.
    """
    

    return model
                

class LoRABertBuzzzer(Buzzer):
    def __init__(self, filename, run_length, num_guesses):
        super().__init__(filename, run_length, num_guesses)

    def initialize_model(self, model_name, rank, alpha):
        """
        Initialize the model and add LoRA layers.
        """

        self.model, self.tokenizer = initialize_base_model(model_name=model_name)
        add_lora(self.model.distilbert.transformer, rank, alpha)

    def dataset_from_questions(self, questions, answer_field="page"):
        from eval import rough_compare
        from datasets import Dataset
        import pandas as pd

        metadata, answers, runs = self._clean_questions(questions, self.run_length, answer_field)
        

        assert self._primary_guesser == "gpr", "This code only works with GPR guessers for the moment"
        guesser = self._guessers[self._primary_guesser]
        dataset = []

        misses = 0
        hits = 0
        for metadatum, answer, run in zip(metadata, answers, runs):
            example = {}

            clean = guesser.clean(run)
            if clean in guesser.cache:
                correct = 0
                if rough_compare(guesser.cache[clean]["guess"], answer):
                    correct = 1
                hits += 1
            else:
                misses += 1
                continue

            if answer is None:
                answer = ""

            example["text"] = run + " [SEP] " + answer
            example["label"] = correct

            dataset.append(example)


        print("Hits: %i, Misses: %i" % (hits, misses))
        dataframe = pd.DataFrame(data=dataset)
        print(dataframe)
        return Dataset.from_pandas(dataframe)

    def train(self, train_questions, eval_questions):
        import numpy as np
        from transformers import DataCollatorWithPadding
        from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
        import evaluate
        
        from tqdm import tqdm

        
        accuracy = evaluate.load("accuracy")

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        
        dataset = {}
        dataset["train"] = train_dataset.map(preprocess_function, batched=True)
        dataset["eval"] = dev_dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)
                    
        training_args = TrainingArguments(  
            output_dir="models/lora_bert_buzzer",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()



if __name__ == "__main__":
    import gzip
    from gpr_guesser import GprGuesser
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold', type=str, default="../data/qanta.buzztrain.json.gz")
    parser.add_argument('--test_fold', type=str, default="../data/qanta.buzzdev.json.gz")
    parser.add_argument('--train_cache', type=str, default="../models/buzztrain_gpr_cache")
    parser.add_argument('--test_cache', type=str, default="../models/buzzdev_gpr_cache")

    parser.add_argument('--run_length', type=int, default=100)  
    parser.add_argument('--limit', type=int, default=-1)

    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--model_name', type=str, default="distilbert-base-uncased")

    flags = parser.parse_args()

    buzzer = LoRABertBuzzzer(filename='', run_length=flags.run_length, num_guesses=1)
    buzzer.initialize_model(flags.model_name, flags.rank, flags.alpha)

    # Train the model
    with gzip.open(flags.train_fold) as infile:
        train_questions = json.load(infile)
        if flags.limit > 0:
            train_questions = train_questions[:flags.limit]

        guesser = GprGuesser(flags.train_cache)   
        guesser.load()     
        buzzer.add_guesser("gpr", guesser, primary_guesser=True)
        train_dataset = buzzer.dataset_from_questions(train_questions)


    with gzip.open(flags.test_fold) as infile:
        dev_questions = json.load(infile)
        if flags.limit > 0:
            dev_questions = dev_questions[:flags.limit]

        guesser = GprGuesser(flags.test_cache)
        guesser.load()
        buzzer.add_guesser("gpr", guesser, primary_guesser=True, replace_guesser=True)
        dev_dataset = buzzer.dataset_from_questions(dev_questions)
        
    buzzer.train(train_questions, dev_questions)
