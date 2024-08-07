"""PPL Inferencer"""

import json
import torch
from openicl import PromptTemplate, ChatTemplate
from openicl.icl_retriever import BaseRetriever
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, PPLInferencerOutputHandler
from openicl.utils.logging import get_logger
from openicl.utils.api_service import *
from typing import List, Union, Optional, List
from tqdm import tqdm
from tqdm import trange
from transformers import PretrainedConfig
from accelerate import Accelerator
import re
from llm_icl.constants import PROMPT_INSTRUCTIONS, ASSISTANT_PROMPTS, DATASET_LABELS, PRE_POST_LABEL_TOKENS, FOCUS_ADDITIONS, PROHIBIT_ADDITIONS
from llm_icl.utils import find_ordered_label_positions, extract_logits, get_max_token_in_label_length, pad_logits, extract_labels_from_prompt, find_answer_label_positions

logger = get_logger(__name__)


class PPLInferencer(BaseInferencer):
    """PPL In-context Learning Inferencer Class
        Perplexity-based In-context Learning Inferencer.
        
    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file. 
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file. 
        api_name (:obj:`str`, optional): Name of API service. 
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`.   
        labels (:obj:`List`, optional): A list of labels for all classes.
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 labels: Optional[List] = None,
                 model_parallel: Optional[bool] = False,
                 use_instruction: Optional[bool] = False,
                 dataset_name: Optional[str] = "stanfordnlp/sst2",
                 num_icl_examples: Optional[int] = 1,
                 verbalised_labels = False,
                 focus_addition = False,
                 prohibit_addition = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.labels = labels
        self.use_instruction = use_instruction
        self.dataset_name = dataset_name
        self.num_icl_examples = num_icl_examples
        self.model_name = model_name
        self.verbalised_labels = verbalised_labels
        self.focus_addition = focus_addition
        self.prohibit_addition = prohibit_addition


        dataset_labels = list(DATASET_LABELS[self.verbalised_labels][dataset_name].values())

        # Add eos or eot token to get answer label tokens and to avoid any extra tokens within the context being considered as part of the label.
        if "mistral" in model_name or "llama" in model_name:
            extra_token = self.tokenizer.eos_token
        elif "gemma" in model_name:
            extra_token = '<end_of_turn>'
        else:
            raise ValueError("Model not supported...")

        # Add a space as matters at least for the Gemma and LLama tokenizers.
        self.tokenized_labels = [
            self.tokenizer(' ' + label + extra_token, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].squeeze(0).to(self.model.device)
            for label in dataset_labels
        ]

        if not self.verbalised_labels:
            # Tokens to ignore in matching labels
            if "mistral" in model_name:
                # Ignore " " and eos token, found through experimentation with the Mistral tokenizer.
                self.ignore_tokens = [torch.tensor([29473]).to(self.model.device), torch.tensor([2]).to(self.model.device)]

            elif "llama" in model_name:
                self.ignore_tokens = [torch.tensor([220]).to(self.model.device), torch.tensor([128009]).to(self.model.device)]
            else: 
                raise ValueError("Model not supported...")
        else:
            # Tokens to ignore in matching labels
            if "mistral" in model_name:
                # Ignore " " and eos token, found through experimentation with the Mistral tokenizer.
                self.ignore_tokens = [torch.tensor([]).to(self.model.device), torch.tensor([2]).to(self.model.device)]
            elif "llama" in model_name:
                self.ignore_tokens = [torch.tensor([]).to(self.model.device), torch.tensor([128009]).to(self.model.device)]
            else: 
                raise ValueError("Model not supported...")


        num_start_ignore_tokens = len(self.ignore_tokens[0])
        num_end_ignore_tokens = len(self.ignore_tokens[1])

        tokenised_raw_labels = [self.tokenizer(label, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].squeeze(0).to(self.model.device)
            for label in dataset_labels
        ]

        # Find max_legnth 
        self.max_tokenised_label_length = max(label.size(0) - num_start_ignore_tokens - num_end_ignore_tokens for label in self.tokenized_labels)


    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                prompt_template: Optional[PromptTemplate] = None, chat_template: Optional[ChatTemplate] = None,
                output_json_filepath: Optional[str] = None, output_json_filename: Optional[str] = None,
                normalizing_str: Optional[str] = None) -> Union[List, torch.Tensor, dict]:
        # 1. Preparation for output logs
        output_handler = PPLInferencerOutputHandler(self.accelerator)

        sub_predictions = []
        ppl = []
        ice = []
        logits_dict = {}  # Dictionary to store logits
        log_likelihoods_dict = {}  # Dictionary to store log likelihoods
        summary = {}  # Dictionary to store additional information

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Get ground truth labels
        ground_truth_labels = retriever.ground_truth_labels()

        # 4. Get labels of all the classes
        if self.labels is None:
            labels = retriever.get_labels(ice_template=ice_template, prompt_template=prompt_template)
        else:
            labels = self.labels

        # 5. Generate in-context examples for testing inputs
        for idx in range(len(ice_idx_list)):
            ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template, chat_template=chat_template, tokenizer=self.tokenizer))
        output_handler.save_ice(ice)

        # 6. Calculating PPL for prompts in each label's class
        for label_idx, label in enumerate(labels):
            index = 0
            prompt_list = []
            sub_ppl_list = []
            normalizing_prompt_list = []
            context_length_list = []

            # 6.1 Generate prompts of current label and truncate if needed
            for idx in range(len(ice_idx_list)):
                prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                        prompt_template=prompt_template, chat_template=chat_template,
                                                        tokenizer=self.tokenizer, remain_sep=normalizing_str is not None)
                if self.max_model_token_num is not None and self.api_name != 'gpt3':
                    prompt_token_num = self.get_input_token_num(prompt)
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                        ice_idx_list[idx] = ice_idx_list[idx][:-1]
                        ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template, chat_template=chat_template, tokenizer=self.tokenizer)
                        prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                                prompt_template=prompt_template, chat_template=chat_template,
                                                                tokenizer=self.tokenizer)
                        prompt_token_num = self.get_input_token_num(prompt)

                if normalizing_str is not None:
                    prompt_sep = prompt
                    if prompt_template is not None:
                        sep_token = prompt_template.sep_token
                    else:
                        sep_token = ice_template.sep_token
                    sep_pos = prompt_sep.find(sep_token)

                    context = prompt_sep[0:sep_pos]
                    answer = prompt_sep[sep_pos:].replace(sep_token, '')
                    prompt = context + answer
                    normalizing_prompt = normalizing_str + answer

                    context_length_list.append(self.get_input_token_num(context))
                    normalizing_prompt_list.append(normalizing_prompt)

                prompt_list.append(prompt)

            if normalizing_str is not None:
                normalizing_str_len = self.get_input_token_num(normalizing_str)

            # 6.2 Get PPL and logits
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list_raw = prompt_list[idx:idx + self.batch_size]
                # Add the instruction to prompt if using this. 
                if self.use_instruction:
                    sub_prompt_list = self.add_instruction_propmt_additions_and_sos_token(sub_prompt_list_raw)
                else:
                    sub_prompt_list = sub_prompt_list_raw
                    
                if normalizing_str is not None:
                    sub_context_length_list = context_length_list[idx:idx + self.batch_size]
                    sub_normalizing_prompt_list = normalizing_prompt_list[idx:idx + self.batch_size]
                    # TODO: Maybe have to add the instruction prompt here as well.
                with torch.no_grad():
                    if normalizing_str is not None:
                        sub_res, extracted_logits_tensor, extracted_log_likelihoods_tensor = self.__get_ppl_and_logits(input_texts=sub_prompt_list, mask_length=sub_context_length_list)
                        sub_res, extracted_logits_tensor, extracted_log_likelihoods_tensor = self.__get_ppl_and_logits(input_texts=sub_normalizing_prompt_list,
                                                            mask_length=[normalizing_str_len for _ in range(len(sub_prompt_list))])
                        sub_res = res1 - res2
                    else:
                        sub_res, extracted_log_likelihoods_tensor = self.__get_ppl_and_logits(sub_prompt_list)

                    for i, (res, log_likelihood_tensor) in enumerate(zip(sub_res, extracted_log_likelihoods_tensor)):
                        sub_ppl_list.append(res)
                        output_handler.save_prompt_and_ppl(label, sub_prompt_list[i], sub_prompt_list[i], res, index)
                        log_likelihoods_dict[(idx + i, label_idx)] = log_likelihood_tensor.cpu()  # Store log likelihoods similarly
                        
                        # Cleanup to save memory
                        del log_likelihood_tensor
                        torch.cuda.empty_cache()

                        # Extract and save additional details
                        sub_prompt = sub_prompt_list[i]
                        icl_labels = extract_labels_from_prompt(sub_prompt, self.model_name)
                        summary[idx + i] = {
                            'prompt_id': idx + i,
                            'prompt': sub_prompt,
                            'icl_labels': icl_labels,
                            'prediction': None,  # To be updated
                            'label': ground_truth_labels[idx + i]
                        }
                        index += 1

            ppl.append(sub_ppl_list)

        # 7. Get lowest PPL class as predictions and store final logits
        ppl = list(zip(*ppl))
        for idx, single_ppl in enumerate(ppl):
            selected_label_idx = single_ppl.index(min(single_ppl))
            selected_label = labels[selected_label_idx]
            sub_predictions.append(selected_label)

            # Retrieve the logits for the best prompt + answer
            # best_prompt_logits = logits_dict[(idx, selected_label_idx)]
            best_log_likelihoods = log_likelihoods_dict[(idx, selected_label_idx)]


            # Now we have the predictive log_likelihood, we can record the log-likelihood.
            summary[idx]['pred_log_likelihood'] = best_log_likelihoods.item()

            query_label_log_probs = []
            for label_idx, label in enumerate(labels):
                query_label_log_probs.append(log_likelihoods_dict[(idx, label_idx)])

            # Normalise the exp of these to get a distrbiution over labels.
            query_label_logits = torch.exp(torch.stack(query_label_log_probs))

            # Softmax the log-likelihoods to get a distribution over labels.
            normalized_label_probs = torch.softmax(query_label_logits, dim=0).squeeze()

            # Now compute the entropy of this distribution.
            semantic_entropy = -torch.sum(normalized_label_probs * torch.log(normalized_label_probs), dim=0)
            
            # Record the entropy.
            summary[idx]['semantic_entropy'] = semantic_entropy.item()

            # Store the predictive distribution
            summary[idx]['predictive_dist'] = normalized_label_probs.tolist()

            # Update the prediction in summary
            summary[idx]['prediction'] = selected_label

            # Adjust the prompt final label to be the selected label.
            prompt = summary[idx]['prompt']

            # Get pre and post tokens for the specified model
            pre_token, post_token = PRE_POST_LABEL_TOKENS[self.model_name]

            # Define the pattern to find the labels in the prompt
            pattern = re.escape(pre_token) + r"(.*?)" + re.escape(post_token)

            # Find all matches and their positions
            matches = list(re.finditer(pattern, prompt))

            # Find the final match
            if matches:
                final_match = matches[-1]
                final_match_text = final_match.group(1)
                
                # Split the final match on ": "
                parts = final_match_text.split(": ")
                
                if len(parts) > 1:
                    # Replace the final part with the selected label
                    parts[-1] = selected_label
                    modified_final_label = ": ".join(parts)
                    
                    # Replace only the final label in the prompt
                    start, end = final_match.span(1)
                    prompt = prompt[:start] + modified_final_label + prompt[end:]

            summary[idx]['prompt'] = prompt

            # Add correctness to output logs.
            summary[idx]['correct'] = int(selected_label == ground_truth_labels[idx])

        output_handler.save_predictions(sub_predictions)

        # 8. Stack and save logits tensor
        # # stacked_logits_tensor = torch.stack(final_logits_list)
        # stacked_log_likelihood_tensor = torch.stack(final_log_likelihood_list)

        # 9. Output
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return [sample['prediction'] for sample in output_handler.results_dict.values()], summary

    @torch.no_grad()
    def __get_ppl_and_logits(self, input_texts: List[str], mask_length=None):
        if self.call_api:
            return api_get_ppl(self.api_name, input_texts)
        self.tokenizer.padding_side = "right"
        
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)
            for i in range(len(mask)):
                mask[i, mask_length[i] - 1:] = 1
            loss *= mask

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens

        # Extract logits and likelihoods at label positions
        extracted_likelihoods = []

        for i in range(len(inputs["input_ids"])):
            # Extract the ICL label positions from the input sequence.
            answer_positions, matched_tokens = find_answer_label_positions(
                inputs["input_ids"][i], self.tokenized_labels, self.ignore_tokens
            )

            # Extract the logits used for the label predictions.
            raw_logits = extract_logits(outputs.logits[i], [answer_positions])

            # Pad the logits to ensure that we can stack them later.
            logits = pad_logits(raw_logits, self.max_tokenised_label_length)

            # Sotmax to get log distribtuion over vocabulary.
            softmax_log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            label_log_likelihoods = []

        
            # Extract the softmax probabilities for each token in the matched label sequence
            # Here, pos is expected to be a tensor with the specific indices for tokens
            position_seq = torch.arange(0, len(matched_tokens))
            token_probs = softmax_log_probs[position_seq, matched_tokens]

            label_log_probs = torch.mean(token_probs, dim=0)

            label_log_likelihoods.append(label_log_probs)

            # extracted_logits.append(logits)
            extracted_likelihoods.append(torch.stack(label_log_likelihoods))  # Stack for each sequence

        # extracted_logits_tensor = torch.stack(extracted_logits)
        extracted_log_likelihoods_tensor = torch.stack(extracted_likelihoods)

        return ce_loss, extracted_log_likelihoods_tensor

 
    def add_instruction_propmt_additions_and_sos_token(self, input_texts):
        # Add on system prompts and special tokens if needed.
        if "mistral" in self.model_name:
            if self.use_instruction:
                messages = [
                    {"role": "user", "content": PROMPT_INSTRUCTIONS[self.dataset_name]},
                    {"role": "assistant", "content": ASSISTANT_PROMPTS[self.dataset_name]},
                ]

                instruction = self.tokenizer.apply_chat_template(messages, tokenize=False)
                prompts = [instruction + text for text in input_texts]

            else:
                prompts = ["<s>" + text for text in input_texts]
        elif "llama" in self.model_name:
            if self.use_instruction:
                if self.focus_addition and not self.prohibit_addition:
                    messages = [
                        {"role": "system", "content": PROMPT_INSTRUCTIONS[self.dataset_name] + FOCUS_ADDITIONS[self.dataset_name]},
                    ]
                elif self.prohibit_addition and not self.focus_addition:
                    messages = [
                        {"role": "system", "content": PROMPT_INSTRUCTIONS[self.dataset_name] + PROHIBIT_ADDITIONS[self.dataset_name]},
                    ]
                elif self.prohibit_addition and self.focus_addition:
                    messages = [
                        {"role": "system", "content": PROMPT_INSTRUCTIONS[self.dataset_name] + FOCUS_ADDITIONS[self.dataset_name] + PROHIBIT_ADDITIONS[self.dataset_name]},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": PROMPT_INSTRUCTIONS[self.dataset_name]},
                    ]

                instruction = self.tokenizer.apply_chat_template(messages, tokenize=False)
                prompts = [instruction + text for text in input_texts]

            else:
                prompts = ["<|begin_of_text|>" + text for text in input_texts]
        elif "gemma" in self.model_name:
            if self.use_instruction:
                messages = [
                    {"role": "user", "content": PROMPT_INSTRUCTIONS[self.dataset_name]},
                    {"role": "assistant", "content": ASSISTANT_PROMPTS[self.dataset_name]},
                ]

                instruction = self.tokenizer.apply_chat_template(messages, tokenize=False)
                prompts = [instruction + text for text in input_texts]

            else:
                prompts = ["<bos>" + text for text in input_texts]
        else:
            raise ValueError("Model not supported...")

        return prompts