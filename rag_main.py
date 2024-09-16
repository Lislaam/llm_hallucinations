from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from constants import LABEL_CONVERSIONS, SYSTEM_INSTRUCTION
from datasets import concatenate_datasets, load_dataset, dataset_dict, DatasetDict
import ast

def preprocess_rag(examples):
    # Preprocess and tokenize documents and atomic facts for retrieval. Tokenize the documents and facts
    inputs = rag_tokenizer(examples['facts'], truncation=True, padding=True, return_tensors='pt')
    outputs = rag_tokenizer(examples['doc'], truncation=True, padding=True, return_tensors='pt')
    return inputs, outputs

def formatting_prompts_func(example, training=True):
    instruction = SYSTEM_INSTRUCTION
    output_texts = []
    
    # Split summary into atomic facts
    facts = split_into_facts(example['summ'])
    
    for i, fact in enumerate(facts):
        text = f"{instruction}\n ### Text1: {example['doc']}\n ### Text2: {fact}\n ### Output: "
        if training:
            if instruction != SYSTEM_INSTRUCTION:
                text += f"{LABEL_CONVERSIONS[example['error_type'][i]]} ." + tokenizer.eos_token
            else:
                label = ast.literal_eval(example["error_type"][i])
                text += f"{str(len(label))} "
                for l in label:
                    text += f"{LABEL_CONVERSIONS[l]} "
                text += tokenizer.eos_token
        output_texts.append(text)
    
    return output_texts

# Initialize tokenizer and retriever for RAG
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)

# Initialize RAG model
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=rag_retriever)

# Apply preprocessing to dataset
dataset = dataset.map(preprocess_rag, batched=True)

# Tokenize the labels for constrained decoding
force_token_ids = rag_tokenizer(list(LABEL_CONVERSIONS.values()), add_special_tokens=False).input_ids

# Set the model to evaluation mode
rag_model.eval()

with torch.no_grad():
    dataset = dataset.map(lambda x: {"formatted_text": formatting_prompts_func(x, False)}, batched=True)
    dataloader = DataLoader(dataset['test'], batch_size=args.batch_size)

    # Make predictions
    predictions = []
    for batch in tqdm(dataloader):
        inputs = rag_tokenizer(batch["formatted_text"], return_tensors="pt", padding=True)

        outputs = rag_model.generate(
            input_ids=inputs["input_ids"].to("cuda"),
            attention_mask=inputs["attention_mask"].to("cuda"),
            do_sample=False,
            num_beams=3,
            num_return_sequences=1,
            max_new_tokens=10,
            force_words_ids=force_token_ids,
        )

        prediction = rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(prediction)

    # Evaluate predictions
    labels = dataset['test']["error_type"]
    preds = [prediction.split("### Output:")[1].strip() for prediction in predictions]

    score = get_score(preds, labels, reverse_labels=LABEL_CONVERSIONS)
    print(f"Total accuracy: {score['total class accuracy']}")
    for error_type in ['correct', 'extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']:
        print(f"{error_type} class accuracy: {score[error_type]}")