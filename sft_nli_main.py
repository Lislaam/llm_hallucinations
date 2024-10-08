from sft import *
from constants import SYSTEM_INSTRUCTION, BINARY_INSTRUCTION, SYSTEM_INSTRUCTION2, GET_ERROR_SPAN
from datasets import concatenate_datasets, load_dataset, dataset_dict, DatasetDict, Dataset
from utils import error_type_map, reformat_data_split_labels, oversampling, undersampling, make_binary_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from peft import LoraConfig
import evaluate
import os
import json
import ast
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from huggingface_hub import login
# login()

# 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
# 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

def main(args):

    def formatting_prompts_func(example, training=True):
        instruction = GET_ERROR_SPAN
        output_texts = []
        for i in range(len(example["error_type"])):
            text = f"{instruction}\n ### ORIGINAL_TEXT: {example['doc'][i]}\n ### SUMMARY: {example['summ'][i]}\n ### Output: " #{example['annotated_span'][i]}\n ### ERROR_CORRECTIONS: {example['annotated_corrections'][i]}\n
       
            if training:
                if instruction == SYSTEM_INSTRUCTION or instruction == BINARY_INSTRUCTION:
                    text += (
                        f"{LABEL_CONVERSIONS[example['error_type'][i]]}." + tokenizer.eos_token
                    )
                else:
                    try:
                        error_locations = ast.literal_eval(example["annotated_span"][i])
                    except:
                        import pdb; pdb.set_trace()
                    for loc in error_locations:
                        text += f"{loc}, "

                    text.removesuffix(', ')
                    text += '.'
                    text += tokenizer.eos_token

            output_texts.append(text)
        
        return output_texts

    # # Load the dataset
    dataset = Dataset.from_file('error_span_data/data-00000-of-00001.arrow')
    dataset = dataset.remove_columns([col for col in dataset.column_names if dataset.filter(lambda x: x[col] is None or x[col] == '').num_rows > 0])
    #dataset = dataset.select(range(10))

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(dataset))

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))

    dir = args.dir
    base_tuned = 'tuned' if args.do_fine_tune=='y' else 'baseline'

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

    response_template = " ### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        do_train=True,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=3000,
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

    if args.do_fine_tune == 'y':
        # Train the model
        trainer.train()

        # Make sure the results directory exists
        os.makedirs(
            os.path.join(dir, str(args.llm), base_tuned),
            exist_ok=True,
        )
        # Plot training loss
        plot_training_loss(
            trainer.state.log_history,
            os.path.join(dir, str(args.llm), base_tuned),
        )

        # Save model
        trainer.save_model(sft_config.output_dir)
        del trainer
        del model

        # Load the model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, torch_dtype=torch.bfloat16, 
                                                     device_map='auto', 
                                                     quantization_config=quantization_config)

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
        logs = []
        predictions = []
        i=0
        for batch in tqdm(dataloader):
            #if i <=5 :
            inputs = tokenizer(batch["formatted_text"], return_tensors="pt", padding=True)

            outputs = model.generate(
                #**inputs,
                inputs=inputs["input_ids"].to("cuda"),
                attention_mask=inputs["attention_mask"].to("cuda"),
                do_sample=False,
                num_beams=3,
                num_return_sequences=1,
                max_new_tokens=50,
            )

            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(prediction)
            i +=1

        # Use soft accuracy for evaluation
        labels = dataset["annotated_span"]
        preds = [prediction.split("### Output:")[1].strip().split('\n')[0] for prediction in predictions]

        # Cleaning the prediction lists
        i = 0
        for pred in preds:
            if "'" in pred[:2]:
                if pred[-2:] != "']":
                    preds[i] = pred + "']"
            elif '"' in pred[:2]:
                if pred[-2:] != '"]':
                    preds[i] = pred + '"]'
        
            try:
                ast.literal_eval(preds[i])
            except:
                import pdb; pdb.set_trace()

            i += 1


        labels = [ast.literal_eval(label) for label in labels]
        preds = [ast.literal_eval(pred) for pred in preds]
        
        for label, pred in zip(labels, preds):
            while len(pred) < len(label):
                pred.append('')
            while len(label) < len(pred):
                label.append('')
        
        flat_labels = [item for sublist in labels for item in sublist] 
        flat_preds = [item for sublist in preds for item in sublist] 

        bleu = evaluate.load("google_bleu")
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")
        meteor = evaluate.load("meteor")

        bleu_score = bleu.compute(predictions=flat_preds, references=flat_labels)
        rouge_score = rouge.compute(predictions=flat_preds, references=flat_labels)
        bertscore_score = bertscore.compute(predictions=flat_preds, references=flat_labels, lang="en")
        meteor_score = meteor.compute(predictions=flat_preds, references=flat_labels)

        # Make sure the results directory exists
        os.makedirs(
            os.path.join(dir, str(args.llm), base_tuned), 
            exist_ok=True,
        )
        # Save the predictions
        with open(os.path.join(dir, str(args.llm), base_tuned, f"summary.json"), "w") as f:
            json.dump([{"prediction": col1, "label": col2} for col1, col2 in zip(preds, labels)],
                        f, indent=4)
            
            results = {
            **bleu_score,
            **rouge_score,
            **meteor_score,
            **bertscore_score,
        }
        # Save results to a file
        with open(os.path.join(dir, str(args.llm), base_tuned, "evaluation_results.json"), "w",) as f:
            json.dump(results, f, indent=4)

        print("_" * 80)


if __name__ == "__main__":
    args = parse_args()

    main(args)
