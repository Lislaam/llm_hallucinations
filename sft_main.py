from sft import *
from constants import SYSTEM_INSTRUCTION
from datasets import concatenate_datasets
from utils import reformat_data, get_score, undersampling, oversampling
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

def main(args):

    def formatting_prompts_func(example, training=True):
        output_texts = []
        for i in range(len(example["error_type"])):
            text = f"{SYSTEM_INSTRUCTION}\n ### Text1: {example['doc'][i]}\n ### Text2: {example['summ'][i]}\n ### Output: "
            if training:
                text += (
                    f"{LABEL_CONVERSIONS[example['error_type'][i]]} ." + tokenizer.eos_token
                )
            output_texts.append(text)
        return output_texts

    # Load the dataset
    dataset = load_dataset(args.dataset, split=['validation[:]', 'test[:]'])
    dataset = concatenate_datasets([dataset[0], dataset[1]]) # Turn into one dataset to make new split
    dataset = reformat_data(dataset, args.dataset) # Get rid of non-standard error_type examples and split data

    if args.sampling == None:
        dir = "whole_dataset"
    elif args.sampling == 'oversampling':
        dataset = oversampling(dataset)
        dir = "naive_oversampling"
    elif args.sampling == 'undersampling':
        dataset = undersampling(dataset)
        dir = "naive_undersampling"

    # Split the dataset into train and test sets (80% train, 20% test)
    train_test = dataset.train_test_split(test_size=0.2)

    # Further split the train set into train and validation sets (75% train, 25% validation of the original 80%)
    train_valid = train_test['train'].train_test_split(test_size=0.25)

    # Combine the splits into a DatasetDict
    dataset = DatasetDict({
        'train': train_valid['train'],
        'validation': train_valid['test'],
        'test': train_test['test']
    })

    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.llm)

    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

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
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
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

    # Train the model
    trainer.train()

    # Make sure the results directory exists
    os.makedirs(
        os.path.join("fine_tuning", str(args.llm)),
        exist_ok=True,
    )

    # Plot training loss
    plot_training_loss(
        trainer.state.log_history,
        os.path.join("fine_tuning", str(args.llm), dir),
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
        list(LABEL_CONVERSIONS.values()), add_special_tokens=False # These are numbers corresponding to the error types
    ).input_ids

    # Place in evaluation mode
    model.eval()
    with torch.no_grad():
        dataset = dataset.map(
            lambda x: {"formatted_text": formatting_prompts_func(x, False)},
            batched=True,
        )
        dataloader = DataLoader(dataset['test'], batch_size=4)

        # Make predictions
        predictions = []
        for batch in tqdm(dataloader):
            inputs = tokenizer(
                batch["formatted_text"], return_tensors="pt", padding=True
            ).to("cuda")
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=3,
                num_return_sequences=1,
                max_new_tokens=1, # Force to only give the number corresponding to the error type
                force_words_ids=force_token_ids,
            )
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(prediction)

        # Use soft accuracy for evaluation
        labels = dataset['test']["error_type"]
        preds = [
            prediction.split("### Output:")[1].strip() for prediction in predictions
        ]

        score = get_sft_score(preds, labels)
        print(f"Total accuracy: {score['total']}")
        for error_type in ['correct', 'extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']:
            print(f"{error_type} class accuracy: {score[error_type]}")

        # Make sure the results directory exists
        os.makedirs(
            os.path.join("fine_tuning", str(args.llm), dir), 
            exist_ok=True,
        )
        # Save the predictions
        with open(os.path.join("fine_tuning", str(args.llm), dir, f"summary.json"), "w") as f:
            json.dump([{"prediction": col1, "label": col2} for col1, col2 in zip([REVERSE_LABEL_CONVERSIONS[i] for i in preds], labels)],
                        f, indent=4)
        # Save results to a file
        with open(
            os.path.join(
                "fine_tuning", str(args.llm), dir,
                "evaluation_results.json",
            ),
            "w",
        ) as f:
            json.dump(score, f, indent=4)

        print("_" * 80)


if __name__ == "__main__":
    args = parse_args()

    main(args)