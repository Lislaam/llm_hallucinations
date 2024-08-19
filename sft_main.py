from sft import *

def main(args):

    def formatting_prompts_func(example, training=True):
        output_texts = []
        for i in range(len(example["error_type"])):
            text = f"### Text1: {example['doc'][i]}\n### Text2: {example['summ'][i]}\n ### Output: "
            if training:
                text += (
                    f"{LABEL_MAP[example['error_type'][i]]} ." + tokenizer.eos_token
                )
            output_texts.append(text)
        return output_texts

    # Load the dataset
    dataset = load_dataset("Lislaam/AggreFact")
    dataset = reformat_data(dataset, "Lislaam/AggreFact")

    train_set = dataset["validation"]

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

    response_template = " ### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        do_train=True,
        num_train_epochs=3,
        max_seq_length=2000,
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
        os.path.join("fine_tuning", str(args.llm)),
        exist_ok=True,
    )

    # Plot training loss
    plot_training_loss(
        trainer.state.log_history,
        os.path.join("fine_tuning", str(args.llm)),
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
        dataset = dataset.map(
            lambda x: {"formatted_text": formatting_prompts_func(x, False)},
            batched=True,
        )
        dataloader = DataLoader(dataset, batch_size=16)

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
                max_new_tokens=3,
                force_words_ids=force_token_ids,
            )
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(prediction)

        # Use soft accuracy for evaluation
        labels = [LABEL_CONVERSIONS[label] for label in dataset["label"]]
        preds = [
            prediction.split("### Output:")[1].strip() for prediction in predictions
        ]

        score = get_score(preds, labels)
        print(f"Total accuracy: {score['total']}")
        for error_type in ['correct', 'extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']: #DATASET_LABELS[args.dataset].values():
            print(f"{error_type} class accuracy: {score[error_type]}")

        # Make sure the results directory exists
        os.makedirs(
            os.path.join("sft_results", args.llm),
            exist_ok=True,
        )

        # Save results to a file
        with open(
            os.path.join(
                "sft_results",
                args.llm,
                "evaluation_results.json",
            ),
            "w",
        ) as f:
            json.dump(score, f, indent=4)

        print("_" * 80)


if __name__ == "__main__":
    args = parse_args()

    main(args)