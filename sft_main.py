from sft import *
from constants import SYSTEM_INSTRUCTION, BINARY_INSTRUCTION, OLD_SYSTEM_INSTRUCTION
from datasets import concatenate_datasets, load_dataset, dataset_dict, DatasetDict
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig
from collections import Counter

def main(args):

    # Define a custom Trainer class to implement weighted loss
    class WeightedSFTTrainer(SFTTrainer):
        def __init__(self, *args, class_weights, **kwargs):
            super().__init__(*args, **kwargs)
            # Set up CrossEntropyLoss with class weights
            self.class_weights = class_weights
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        def compute_loss(self, model, inputs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            
            # Assuming logits has size [batch_size, seq_len, vocab_size]
            batch_size, seq_len, vocab_size = logits.size()
            import pdb; pdb.set_trace()
            constrained_logits = torch.index_select(logits, dim=2, index=torch.tensor(force_token_ids).to(logits.device))

            # Flatten the logits and labels for token-level cross-entropy
            logits_flat = constrained_logits.view(-1, constrained_logits.size(-1))  # [batch_size * seq_len, num_error_types]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]

            # Apply class weights for error type tokens
            token_loss = F.cross_entropy(logits_flat, labels_flat, weight=self.class_weights, ignore_index=tokenizer.pad_token_id)

            return token_loss

    def formatting_prompts_func(example, training=True):
        instruction = BINARY_INSTRUCTION if args.sampling=='binary' else SYSTEM_INSTRUCTION
        output_texts = []
        for i in range(len(example["error_type"])):
            text = f"{instruction}\n ### Text1: {example['doc'][i]}\n ### Text2: {example['summ'][i]}\n ### Output: "
            if training:
                if instruction != SYSTEM_INSTRUCTION:
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

    # Load the dataset
    dataset = load_dataset(args.dataset, split=['validation[:20]', 'test[:20]'])
    dataset = concatenate_datasets([dataset[0], dataset[1]]) # Turn into one dataset to make new split
    dataset = dataset.filter(lambda x: error_type_map(x) is not None) # reformat_data_split_labels(dataset, args.dataset) # Get rid of non-standard error_type examples and split data
    dataset = dataset.map(error_type_map)
    # dataset = dataset.filter(lambda x: x is not None and None not in x.values())

    if args.sampling == None:
        dir = "whole_dataset"
    elif args.sampling == 'oversampling':
        dataset = oversampling(dataset)
        dir = "naive_oversampling"
    elif args.sampling == 'undersampling':
        dataset = undersampling(dataset)
        dir = "naive_undersampling"
    elif args.sampling == 'binary':
        dataset = make_binary_dataset(dataset)
        dir = "binary"
    else:
        print("Sampling not supported. Choose oversampling, undersampling or binary")
        exit

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
    print(formatting_prompts_func(dataset['train'][:1], True))

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

    # Assuming `dataset` contains the `error_type` labels
    error_type_counts = Counter(dataset['train']['error_type'])

    # Calculate total samples and class weights
    total_samples = sum(error_type_counts.values())
    class_weights = {label: total_samples / count for label, count in error_type_counts.items()}

    # Convert the class weights to a tensor
    class_weights_tensor = torch.tensor([class_weights[label] for label in sorted(class_weights.keys())], dtype=torch.float32).to('cuda')  # Ensure to match the label indices

    # Initialize the WeightedSFTTrainer
    trainer = WeightedSFTTrainer(
        model=model,
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
        class_weights=class_weights_tensor  # Pass class weights here
    )

    # Train the model
    trainer.train()

    # Make sure the results directory exists
    os.makedirs(
        os.path.join("count_errors_sft", str(args.llm), dir),
        exist_ok=True,
    )
    # Plot training loss
    plot_training_loss(
        trainer.state.log_history,
        os.path.join("count_errors_sft", str(args.llm), dir),
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
        dataset = Dataset.from_file('data/eval/data-00000-of-00001.arrow')
        dataset = dataset.map(
            lambda x: {"formatted_text": formatting_prompts_func(x, False)},
            batched=True,
        )
        dataloader = DataLoader(dataset['test'], batch_size=args.batch_size)

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
                max_new_tokens=10,
                force_words_ids=force_token_ids,
            )
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(prediction)

        # Use soft accuracy for evaluation
        labels = dataset['test']["error_type"]
        preds = [
            prediction.split("### Output:")[1].strip() for prediction in predictions
        ]
        
        if args.sampling == 'binary':
            score = get_single_label_score(preds, labels, binary=True)
            print(f"Total accuracy: {score['total']}")
            for error_type in ['correct', 'incorrect']:
                print(f"{error_type} class accuracy: {score[error_type]}")
        else:
            score = get_score(preds, labels, reverse_labels=LABEL_CONVERSIONS)
            print(f"Detecting # errors accuracy: {score['accuracy detecting # errors']}")
            print(f"Total accuracy: {score['total class accuracy']}")
            for error_type in ['correct', 'extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']:
                print(f"{error_type} class accuracy: {score[error_type]}")

        # Make sure the results directory exists
        os.makedirs(
            os.path.join("count_errors_sft", str(args.llm), dir), 
            exist_ok=True,
        )
        # Save the predictions
        with open(os.path.join("count_errors_sft", str(args.llm), dir, f"summary.json"), "w") as f:
            json.dump([{"prediction": col1, "label": col2} for col1, col2 in zip(preds, labels)],
                        f, indent=4)
        # Save results to a file
        with open(os.path.join("count_errors_sft", str(args.llm), dir, "evaluation_results.json"), "w",) as f:
            json.dump(score, f, indent=4)

        print("_" * 80)


if __name__ == "__main__":
    args = parse_args()

    main(args)