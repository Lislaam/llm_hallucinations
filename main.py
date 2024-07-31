from datasets import load_dataset
from datasets import DatasetDict
from openicl import PromptTemplate
from transformers import AutoTokenizer
from openicl import (
    DatasetReader,
    PPLInferencer,
    RandomRetriever,
    AccEvaluator,
    VotekRetriever,
    TopkRetriever,
)
from setup import *
from llm_icl.constants import DATASET_PROMPTS, TEST_SPLIT, DATASET_LABELS
from llm_icl.utils import reformat_data


def main(args):
    assert not ((args.focus_addition  or args.prohibit_addition) and not args.use_instruction), "Prompt additions can only be used with instruction."
    
    test_split = TEST_SPLIT[args.dataset]

    if args.use_instruction == "True":
        args.use_instruction = True
    elif args.use_instruction == "False":
        args.use_instruction = False
    else:
        raise ValueError("Use intruction not set properly.")

    scores = {}
    for model in args.llms:
        print(f"Starting inference routine on {model} on the dataset {args.dataset}...")
        for num_ice in args.num_icl_examples:
            print(
                f"Model: {model}, n_shots {num_ice}, dataset: {args.dataset}, focus_addition: {args.focus_addition}, prohibit_addition {args.prohibit_addition} inference routine started..."
            )
        
            if args.dataset == "google-research-datasets/paws":
                # Loading dataset from huggingface
                datasets = load_dataset(args.dataset , 'labeled_final', split=["train", test_split])
            else:
                datasets = load_dataset(args.dataset ,  split=["train", test_split])


            reformatted_datasets = [
                reformat_data(
                    dataset, args.dataset, verbalised_labels=args.verbalised_labels
                )
                for dataset in datasets
            ]

            # Just extract the first 4000 examples for the test set. Fairer comparison.
            if args.dataset in ["jhu-cogsci/hans", "DT4LM/qqp", "google-research-datasets/paws", "SetFit/mnli"]:
                reformatted_datasets[1] = reformatted_datasets[1].shuffle(seed=42).select(range(1000))

            # Create a DatasetDict object
            dataset = DatasetDict(
                {"train": reformatted_datasets[0], "test": reformatted_datasets[1]}
            )

            # Define a DatasetReader, with specified column names where input and output are stored.
            data = DatasetReader(
                dataset, input_columns=["input_text"], output_column="output"
            )

            # Load tokenizer to set chat template.
            tokenizer = AutoTokenizer.from_pretrained(model)

            input_start = DATASET_PROMPTS[args.dataset]["input"]
            output_start = DATASET_PROMPTS[args.dataset]["output"]

            messages = []
            messages.append(
                {
                    "role": "user",
                    "content": f"{input_start}: " + "{/input_text}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"{output_start}: " + "{/output}",
                }
            )

            column_token_map = {"input_text": "{/input_text}"}

            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt = add_ic_token_and_remove_sos_token(prompt, model)

            # Create prompt template dictionary.
            tp_dict = {
                label: f"{prompt.replace('{/output}', label)}"
                for label in DATASET_LABELS[args.verbalised_labels][
                    args.dataset
                ].values()
            }

            template = PromptTemplate(
                tp_dict, column_token_map=column_token_map, ice_token="</E>"
            )

            retriever = select_retriever(
                args.retriever,
                data,
                num_ice,
                "train",
                "test",
                "",
                "",
            )

            inferencer = PPLInferencer(
                model_name=model,
                batch_size=args.batch_size,
                device="cuda",
                use_instruction=args.use_instruction,
                dataset_name=args.dataset,
                num_icl_examples=num_ice,
                verbalised_labels=args.verbalised_labels,
                focus_addition = args.focus_addition,
                prohibit_addition = args.prohibit_addition
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
            if args.use_instruction:
                ins_name = "instruction"
            else:
                ins_name = "no_instruction"
            additional_prompt_instructions = f"focus_addition_{args.focus_addition}_prohibit_addition_{args.prohibit_addition}"

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
                additional_prompt_instructions,
                dataset_name,
                ins_name,
                retriever,
                model_name,
                f"verbalised_labels_{args.verbalised_labels}",
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
        if args.use_instruction:
            ins_name = "instruction"
        else:
            ins_name = "no_instruction"
        save_dir = os.path.join(
            "results",
            additional_prompt_instructions,
            dataset_name,
            ins_name,
            retriever,
            model_name,
            f"verbalised_labels_{args.verbalised_labels}",
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "scores.json"), "w") as f:
            json.dump(scores, f, indent=4)


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    main(args)