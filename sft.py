import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from peft import LoraConfig, get_peft_model
import json
from datasets import load_dataset
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser

OUTPUT_DIR = "/scratch/local/ssd/tomlamb/icl-uncertainty"

LABEL_CONVERSIONS = {
    0: "9T",
    1: "3F",
    2: "7K",
}


def soft_accuracy(predictions, references):
    correct = 0
    total = len(predictions)
    for pred, ref in zip(predictions, references):
        if ref in pred:
            correct += 1
    return correct / total


def plot_training_loss(log_history, output_dir):
    df = pd.DataFrame(log_history)
    if "loss" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["step"], df["loss"], label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "training_loss.png"))
        plt.show()
    else:
        print("No loss information found in log history.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="The language model to use for data generation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="The batch size for data generation.",
    )
    parser.add_argument(
        "--train_spurious_correlation",
        type=float,
        default=0.9,
        help="The spurious correlation for the train set.",
    )

    return parser.parse_args()


def main(args):
    different_test_sets = [
        "train",
        "same_iid",
        "reduced_correlation",
        "randomized",
        "flipped",
    ]

    spurious_correlation = args.train_spurious_correlation

    # Load the dataset
    dataset = load_dataset(f"tomalamb/spurious_nli_correlation_{spurious_correlation}")

    train_set = dataset["train"]

    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)

    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    def formatting_prompts_func(example, training=True):
        output_texts = []
        for i in range(len(example["label"])):
            text = f"### Text1: {example['premise'][i]}\n### Text2: {example['hypothesis'][i]}\n ### Output: "
            if training:
                text += (
                    f"{LABEL_CONVERSIONS[example['label'][i]]} ." + tokenizer.eos_token
                )
            output_texts.append(text)
        return output_texts

    response_template = " ### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        do_train=True,
        num_train_epochs=3,
        max_seq_length=400,
        logging_steps=200,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_set,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Make sure the results directory exists
    os.makedirs(
        os.path.join("results", "sft", "spurious_nli", str(spurious_correlation)),
        exist_ok=True,
    )

    # Plot training loss
    plot_training_loss(
        trainer.state.log_history,
        os.path.join("results", "sft", "spurious_nli", str(spurious_correlation)),
    )

    # Save model
    trainer.save_model(sft_config.output_dir)
    del trainer
    del model

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR, torch_dtype=torch.bfloat16
    ).to("cuda")

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    # Tokenize the labels for contrained decoding
    force_token_ids = tokenizer(
        list(LABEL_CONVERSIONS.values()), add_special_tokens=False
    ).input_ids

    # Place in evaluation mode
    model.eval()
    with torch.no_grad():
        # Evaluate the model on each test set
        results = {}
        for test_set in different_test_sets:
            test_dataset = dataset[test_set]
            print(f"Loaded test set {test_set}")

            test_dataset = test_dataset.map(
                lambda x: {"formatted_text": formatting_prompts_func(x, False)},
                batched=True,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=16)

            # Make predictions
            predictions = []
            for batch in tqdm(test_dataloader):
                inputs = tokenizer(
                    batch["formatted_text"], return_tensors="pt", padding=True
                ).to("cuda")
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=3,
                    num_return_sequences=1,
                    max_new_tokens=3,
                    force_words_ids=force_token_ids,
                )
                prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(prediction)

            # Use soft accuracy for evaluation
            labels = [LABEL_CONVERSIONS[label] for label in test_dataset["label"]]
            preds = [
                prediction.split("### Output:")[1].strip() for prediction in predictions
            ]

            accuracy = soft_accuracy(preds, labels)
            results[f"test_set_{test_set}"] = accuracy

            print(f"Accuracy for {test_set}: {accuracy}")

        # Make sure the results directory exists
        os.makedirs(
            os.path.join("results", "sft", "spurious_nli", str(spurious_correlation)),
            exist_ok=True,
        )

        # Save results to a file
        with open(
            os.path.join(
                "results",
                "sft",
                "spurious_nli",
                str(spurious_correlation),
                "evaluation_results.json",
            ),
            "w",
        ) as f:
            json.dump(results, f, indent=4)

        # Report results
        print(
            f"Evaluation Results for spurious correlation {str(spurious_correlation)}:"
        )
        for test_set, accuracy in results.items():
            print(f"{test_set}: {accuracy}")

        print("_" * 80)


if __name__ == "__main__":
    args = parse_args()

    main(args)