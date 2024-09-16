from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, BitsAndBytesConfig, BartTokenizer
from constants import SYSTEM_INSTRUCTION
from datasets import concatenate_datasets, load_dataset, dataset_dict, DatasetDict, Dataset
import ast
from rag_utils import *


def main(args):
    def formatting_prompts_func(example, training=True):
        instruction = SYSTEM_INSTRUCTION
        output_texts = []
        for i in range(len(example["error_type"])):
            text = f"{instruction}\n ### ORIGINAL_TEXT: {example['doc'][i]}\n ### SUMMARY: {example['summ'][i]}\n ### ERROR_LOCATIONS: {example['annotated_span']}\n ### Output: "
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
    dataset = load_dataset('csv', data_files='rag_test_data.csv')
    dataset = concatenate_datasets([dataset['train']])

    dataset.add_faiss_index(column='doc')  # Indexes document embeddings using FAISS
    dataset.save_to_disk('rag_test_data222.csv')  # Save the dataset
    dataset.get_index('doc').save('rag_test_indexxxxx.csv')  # Save the FAISS index
    
    #dataset = load_dataset('rag_test_data.csv', split=['validation[:]', 'test[:]'])
    #dataset = concatenate_datasets([dataset[0], dataset[1]]) # Turn into one dataset to make new split
    #model_name = "HuggingFaceH4/zephyr-7b-beta"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_bart=True)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom", passages=dataset['train']['doc'])

    # Tokenize the labels for constrained decoding
    force_tokens = list(LABEL_CONVERSIONS.values())
    force_tokens = [f" {token}" for token in force_tokens]
    force_token_ids = tokenizer(
        force_tokens, add_special_tokens=False
    ).input_ids

    tokenizer.padding_side = "left"

    # Place in evaluation mode
    model.eval()
    with torch.no_grad():
        dataset = dataset.map(
            lambda x: {"formatted_text": formatting_prompts_func(x, False)},
            batched=True,
        )
        dataloader = DataLoader(dataset['train'], batch_size=args.batch_size)

        # Make predictions
        predictions = []
        for batch in tqdm(dataloader):
            # Tokenize inputs
            inputs = tokenizer(batch["formatted_text"], return_tensors="pt", padding=True)

            # Retrieve documents for RAG
            retrieved_docs = retriever(batch['formatted_text'])

            # Generate outputs with retrieved context
            outputs = model.generate(
                input_ids=inputs["input_ids"].to("cuda"),
                attention_mask=inputs["attention_mask"].to("cuda"),
                context_input_ids=retrieved_docs['input_ids'].to('cuda'),  # Documents retrieved
                context_attention_mask=retrieved_docs['attention_mask'].to('cuda'),
                do_sample=False,
                num_beams=3,
                num_return_sequences=1,
                max_new_tokens=10,
                force_words_ids=force_token_ids,
            )

            # Decode predictions
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(prediction)

        # Use soft accuracy for evaluation
        labels = dataset['test']["error_type"]
        preds = [
            prediction.split("### Output:")[1].strip() for prediction in predictions
        ]

        # Save the predictions
        output_dir = os.path.join("rag_test", str(args.llm))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"summary.json"), "w") as f:
            json.dump([{"prediction": col1, "label": col2} for col1, col2 in zip(preds, labels)], f, indent=4)

        # Save results to a file (replace `score` with actual evaluation logic if necessary)
        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump({}, f, indent=4)  # Update with the actual score if needed

        print("_" * 80)

if __name__ == "__main__":
    args = parse_args()
    main(args)