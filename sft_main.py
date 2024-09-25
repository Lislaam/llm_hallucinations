from sft import *
from constants import SYSTEM_INSTRUCTION, BINARY_INSTRUCTION, OLD_SYSTEM_INSTRUCTION
from datasets import concatenate_datasets, load_dataset, dataset_dict, DatasetDict
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig
from collections import Counter
from transformers import DataCollatorForCompletionOnlyLM
from transformers import SFTTrainer, SFTConfig, LoraConfig
import os
import json
import ast
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args):

    def formatting_prompts_func(example, training=True):
        instruction = SYSTEM_INSTRUCTION
        output_texts = []
        for i in range(len(example["error_type"])):
            text = f"{instruction}\n ### ORIGINAL_TEXT: {example['doc'][i]}\n ### SUMMARY: {example['summ'][i]}\n ### ERROR_LOCATIONS: {example['annotated_span'][i]}\n ### ERROR_CORRECTIONS: {example['annotated_corrections'][i]}\n ### Output: " #
            if training:
                if instruction == SYSTEM_INSTRUCTION:
                    text += (
                        f"{LABEL_CONVERSIONS[example['error_type'][i]]}." + tokenizer.eos_token
                    )
                else:
                    label = ast.literal_eval(example["error_type"][i])
                    text += f"{str(len(label))}, "
                    for l in label:
                        text += f"{LABEL_CONVERSIONS[l]}, "
                    text.removesuffix(', ')
                    text += '.'
                    text += tokenizer.eos_token

            output_texts.append(text)
        
        return output_texts

    # # Load the dataset
    dataset = Dataset.from_file('rag_data/train/data-00000-of-00001.arrow')
    dataset = dataset.remove_columns([col for col in dataset.column_names if dataset.filter(lambda x: x[col] is None or x[col] == '').num_rows > 0])

    train_size = 240
    val_size = 30
    test_size = 30

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

    dir = args.dir

    # dataset = load_dataset(args.dataset, split=['validation[:20]', 'test[:20]'])
    # dataset = concatenate_datasets([dataset[0], dataset[1]]) # Turn into one dataset to make new split
    # dataset = dataset.filter(lambda x: error_type_map(x) is not None) # reformat_data_split_labels(dataset, args.dataset) # Get rid of non-standard error_type examples and split data
    # dataset = dataset.map(error_type_map)
    # # dataset = dataset.filter(lambda x: x is not None and None not in x.values())

    # if args.sampling == None:
    #     dir = "whole_dataset"
    # elif args.sampling == 'oversampling':
    #     dataset = oversampling(dataset)
    #     dir = "naive_oversampling"
    # elif args.sampling == 'undersampling':
    #     dataset = undersampling(dataset)
    #     dir = "naive_undersampling"
    # elif args.sampling == 'binary':
    #     dataset = make_binary_dataset(dataset)
    #     dir = "binary"
    # else:
    #     print("Sampling not supported. Choose oversampling, undersampling or binary")
    #     exit

    # # Split the dataset into train and test sets (80% train, 20% test)
    # train_test = dataset.train_test_split(test_size=0.2)

    # # Further split the train set into train and validation sets (75% train, 25% validation of the original 80%)
    # train_valid = train_test['train'].train_test_split(test_size=0.25)

    # # Combine the splits into a DatasetDict
    # dataset = DatasetDict({
    #     'train': train_valid['train'],
    #     'validation': train_valid['test'],
    #     'test': train_test['test']
    # })

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16, device_map='auto', quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)

    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=8,  # Was 16 for mistral and llama trainings oversample/undersample. NOT during binary dataset
        lora_alpha=8,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Test formatting for 1st and 2nd example:
    # print(formatting_prompts_func(dataset['train'][:1], True))

    # Tokenize the labels for constrained decoding
    force_tokens = list(LABEL_CONVERSIONS.values())
    force_tokens = [f" {token}" for token in force_tokens]
    force_token_ids = tokenizer(
        force_tokens, add_special_tokens=False
    ).input_ids

    response_template = " ### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        do_train=True,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=2500,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        bf16=True,
        metric_for_best_model="eval_loss",
        per_device_train_batch_size=args.batch_size,
        load_best_model_at_end = True,
    )

    trainer = SFTTrainer( 
        model=model,
        train_dataset= train_dataset, #dataset['train'],
        eval_dataset= val_dataset, #dataset['validation'],
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=lora_config,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )

    if args.train:
        # Train the model
        trainer.train()

        # Make sure the results directory exists
        os.makedirs(
            os.path.join("different_metrics", str(args.llm), dir),
            exist_ok=True,
        )
        # Plot training loss
        plot_training_loss(
            trainer.state.log_history,
            os.path.join("different_metrics", str(args.llm), dir),
        )

        # Save model
        trainer.save_model(sft_config.output_dir)
        del trainer
        del model

        # Load the model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch.bfloat16, device_map='auto', quantization_config=quantization_config)

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    # Place in evaluation mode
    model.eval()
    with torch.no_grad():
        dataset = test_dataset
        dataset = dataset.map(
            lambda x: {"formatted_text": formatting_prompts_func(x, False)},
            batched=True,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size)

        # Make predictions
        predictions = []
        for batch in tqdm(dataloader):
            inputs = tokenizer(batch["formatted_text"], return_tensors="pt", padding=True)

            outputs = model.generate(
                #**inputs,
                inputs=inputs["input_ids"].to("cuda"),
                attention_mask=inputs["attention_mask"].to("cuda"),
                do_sample=False,
                num_beams=3,
                num_return_sequences=1,
                max_new_tokens=15,
                force_words_ids=force_token_ids,
            )
            logits = outputs.logit
            pred_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(prediction)

        # Use soft accuracy for evaluation
        labels = dataset["error_type"]
        preds = [
            prediction.split("### Output:")[1].strip() for prediction in predictions
        ]
        
        score = get_score(preds, labels)
        entropy = get_average_entropy(pred_probs)
        ece = get_ece(pred_probs, labels)
        brier = get_brier(pred_probs, labels)

        print(f"Total accuracy: {score['total']}")
        print(f"Entropy: {entropy}")
        print(f"ECE: {ece}")
        print(f"Brier: {brier}")
        for error_type in ['extrinsic', 'intrinsic']:
            print(f"{error_type} class accuracy: {score[error_type]}")

        # if args.sampling == 'binary':
        #     score = get_single_label_score(preds, labels, binary=True)
        #     print(f"Total accuracy: {score['total']}")
        #     for error_type in ['correct', 'incorrect']:
        #         print(f"{error_type} class accuracy: {score[error_type]}")
        # else:
        #     score = get_score(preds, labels, reverse_labels=LABEL_CONVERSIONS)
        #     print(f"Detecting # errors accuracy: {score['accuracy detecting # errors']}")
        #     print(f"Total accuracy: {score['total class accuracy']}")
        #     for error_type in ['correct', 'extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']:
        #         print(f"{error_type} class accuracy: {score[error_type]}")

        # Make sure the results directory exists
        os.makedirs(
            os.path.join("different_metrics", str(args.llm), dir), 
            exist_ok=True,
        )
        # Save the predictions
        with open(os.path.join("different_metrics", str(args.llm), dir, f"summary.json"), "w") as f:
            json.dump([{"prediction": col1, "label": col2} for col1, col2 in zip(preds, labels)],
                        f, indent=4)
        # Save results to a file
        with open(os.path.join("different_metrics", str(args.llm), dir, "evaluation_results.json"), "w",) as f:
            json.dump(score, f, indent=4)
            json.dump(entropy, f, indent=4)
            json.dump(ece, f, indent=4)
            json.dump(brier, f, indent=4)

        print("_" * 80)


if __name__ == "__main__":
    args = parse_args()

    main(args)