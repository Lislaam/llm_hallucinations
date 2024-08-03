from datasets import load_dataset
from datasets import DatasetDict
from openicl import PromptTemplate
from transformers import AutoTokenizer
from openicl import (
    DatasetReader,
    GenInferencer,
    RandomRetriever,
    AccEvaluator,
    VotekRetriever,
    TopkRetriever,
)
from setup import *
from constants import DATASET_LABELS, DATASET_PROMPTS
from utils import reformat_data

from huggingface_hub import login
# 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
#login()

def main(args):
    #test_split = TEST_SPLIT[args.dataset]

    scores = {}
    #for model in args.llms:
    model = args.llms
    print(f"Starting inference routine on {model} on the dataset {args.dataset}...")
    
    for num_ice in args.num_icl_examples:
        print(
            f"Model: {model}, n_shots: {num_ice}, dataset: {args.dataset}, inference routine started..."
        )
    
        if args.dataset == "Lislaam/AggreFact":
            # Loading dataset from huggingface
            datasets = load_dataset(args.dataset) # Contains validation and test sets
        else:
            print("Must use Lislaam/AggreFact")


        """ reformatted_datasets = [
            reformat_data(
                dataset, args.dataset
            )
            for dataset in datasets
        ]
"""
        # Create a DatasetDict object
        dataset = DatasetDict(
            {"val": datasets['validation'], "test": datasets['test']} # Checked Correct
        )

        # Define a DatasetReader, with specified column names where input and output are stored.
        data = DatasetReader(
            dataset, input_columns=["input_text"], output_column="error_type"
        )

        # Load tokenizer to set chat template.
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

        input_start = DATASET_PROMPTS[args.dataset]["input"]
        output_start = DATASET_PROMPTS[args.dataset]["output"]

        messages = []
        messages.append(
            {
                "role": "user",
                "content": f"{input_start[0]}: " + "{/input_text}" + f"\n{input_start[1]}:" + "{/input_text}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"{output_start}: " + "{/output}",
            }
        )

        column_token_map = {"input_text": "{/input_text}"}

        # # Define a DatasetReader, with specified column names where input and output are stored.
        # data = DatasetReader(
        #     dataset, input_columns=["doc", "summ"], output_column="error_type" # Checked Correct
        # )

        # # Load tokenizer to set chat template.
        # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct') 
        
        # messages = []
        # # messages.append(
        # #         {
        # #         "role": "system",
        # #         "content": "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
        # #         },
        # # )
        # messages.append(
        #     {
        #         "role": "user",
        #         "content": r"Document: {/doc}\nSummary: {/summ}", # Added start token /E
        #     }
        # )
        # messages.append(
        #     {
        #         "role": "assistant",
        #         "content": r"Error Type: {/error_type}"
        #     }
        # )

        # column_token_map = {
        #                         "doc": r"{/doc}",
        #                         "summ": r"{/summ}",
        #                     }

        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        #print("Initial Prompt:\n", prompt)
        prompt = add_ic_token_and_remove_sos_token(prompt, model)
        #print("Processed Prompt:\n", prompt)

        # Create prompt template dictionary.
        tp_dict = {
            label: f"{prompt.replace('{/error_type}', label)}"
            for label in DATASET_LABELS[args.dataset].values()
        }

        #def generate_tp_dict(prompt_template, labels):
           # return {
          #      label: prompt_template.replace("</error_type>", label)
         #       for label in labels.values()
        #    }

        # Generate the tp_dict]
        #tp_dict = generate_tp_dict(prompt, DATASET_LABELS[args.dataset])

        # Print the resulting dictionary to verify
        for key, value in tp_dict.items():
            print(f"'{key}':\n{value}\n")

        template = PromptTemplate(
            tp_dict, column_token_map=column_token_map, ice_token="</E>"
        )

        retriever = select_retriever(
            args.retriever, # Defaults to random
            data,
            num_ice,
            "val",
            "test",
            "",
            "",
        )

        inferencer = GenInferencer(
            model_name=model,
            batch_size=args.batch_size,
            device="cuda",
            dataset_name=args.dataset, # Defaults to AggreFact
            num_icl_examples=num_ice,
        )

        # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
        predictions, summary = inferencer.inference(
            retriever,
            ice_template=template,
        )

        # Save the label tensor.
        os.makedirs("results", exist_ok=True)
        dataset_name = args.dataset.split("/")[-1]
        model_name = model.split("/")[-1]
        retriever = args.retriever

        # # Save the tensor containing the logits at context label positions.
        # save_dir = os.path.join(
        #     args.scratch_dir,
        #     dataset_name,
        #     ins_name,
        #     retriever,
        #     model_name,
        #     f"verbalised_labels_{args.verbalised_labels}",
        # )
        # os.makedirs(save_dir, exist_ok=True)
        # # torch.save(
        # #     label_tensor, os.path.join(save_dir, f"label_tensor_{num_ice}.pt")
        # # )

        # Save the summary.
        save_dir = os.path.join(
            "results",
            retriever,
            model_name,
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"summary_{num_ice}.json"), "w") as f:
            json.dump(summary, f, indent=4)

        # Compute accuracy for the prediction
        score = AccEvaluator().score(
            predictions=predictions, references=data.references
        )

        print(f"Accuracy for {num_ice} ICL examples: {score}")

        results_file_path = os.path.join(save_dir, "scores.json")

        # Check if results file exists and load it
        if os.path.exists(results_file_path):
            with open(results_file_path, "r") as f:  # open the file in read mode
                scores = json.load(f)

        scores[num_ice] = score

        # Delete the retriever and inferencer objects to save on memory.
        del retriever
        del inferencer

        print(
            f"Finished inference routine for {model} with n-shots {num_ice} on {args.dataset}..."
        )
        print("__________________________")

    # Save the scores
    os.makedirs("results", exist_ok=True)
    dataset_name = args.dataset.split("/")[-1]
    model_name = model.split("/")[-1]
    retriever = args.retriever

    save_dir = os.path.join(
        "results",
        retriever,
        model_name,
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    main(args)