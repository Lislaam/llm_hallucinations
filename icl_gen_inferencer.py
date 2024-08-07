"""Direct Generation Inferencer"""

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.utils.api_service import *
from openicl.utils.icl_common_utils import get_dataloader, get_generation_prompt_list_from_retriever_indices
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import PretrainedConfig
from accelerate import Accelerator

from constants import PROMPT_INSTRUCTIONS, DATASET_LABELS, PRE_POST_LABEL_TOKENS
from utils import find_ordered_label_positions, extract_logits, get_max_token_in_label_length, pad_logits, extract_labels_from_prompt, find_answer_label_positions


logger = get_logger(__name__)


class GenInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.
        
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
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method. 
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs={"max_new_tokens": 100},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,

                 labels: Optional[List] = None,
                 dataset_name: Optional[str] = "Lislaam/AggreFact",
                 num_icl_examples: Optional[int] = 0,

                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs

        self.labels = labels
        self.dataset_name = dataset_name
        self.num_icl_examples = num_icl_examples
        self.model_name = model_name

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

        #if not self.verbalised_labels:
            # Tokens to ignore in matching labels
        if "mistral" in model_name:
            # Ignore " " and eos token, found through experimentation with the Mistral tokenizer.
            self.ignore_tokens = [torch.tensor([29473]).to(self.model.device), torch.tensor([2]).to(self.model.device)]

        elif "llama" in model_name:
            self.ignore_tokens = [torch.tensor([220]).to(self.model.device), torch.tensor([128009]).to(self.model.device)]
        else: 
            raise ValueError("Model not supported...")
        # else:
        #     # Tokens to ignore in matching labels
        #     if "mistral" in model_name:
        #         # Ignore " " and eos token, found through experimentation with the Mistral tokenizer.
        #         self.ignore_tokens = [torch.tensor([]).to(self.model.device), torch.tensor([2]).to(self.model.device)]
        #     elif "llama" in model_name:
        #         self.ignore_tokens = [torch.tensor([]).to(self.model.device), torch.tensor([128009]).to(self.model.device)]
        #     else: 
        #         raise ValueError("Model not supported...")

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
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, force_words=None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0

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

        # 4. Generate prompts for testing input 
        prompt_list_raw = get_generation_prompt_list_from_retriever_indices(ice_idx_list, retriever, self.tokenizer,
                                                                        self.gen_field_replace_token,
                                                                        max_model_token_num=self.max_model_token_num,
                                                                        ice_template=ice_template,
                                                                        prompt_template=prompt_template,
                                                                        chat_template=chat_template)
        
        prompt_list = self.add_instruction_prompt_additions_and_sos_token(prompt_list_raw)
        output_handler.save_orgin_prompts(prompt_list)

        # 5. Wrap prompts with Dataloader
        dataloader = get_dataloader(prompt_list, self.batch_size)

        # 6. Inference for prompts in each batch 
        logger.info("Starting inference process...")
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            # 6-1. Inference with local model
            if not self.call_api:
                with torch.no_grad():
                    tokenized_data = self.tokenizer.batch_encode_plus(entry, padding=True, return_tensors='pt').to(
                        self.device)
                    prompt_len = int(tokenized_data.attention_mask.shape[1])
                    if 't5' in self.model_name:
                        prompt_len = 0
                    if force_words is not None:
                        force_words_ids = [
                            self.tokenizer(force_words).input_ids,
                        ]
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      force_words_ids=force_words_ids,
                                                      num_beams=10,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    else:
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                    outputs = outputs.tolist()
                    complete_output = self.tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
                    generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                                            skip_special_tokens=True)
            # 5-2. Inference with remote API
            else:
                complete_output, generated = api_get_tokens(self.api_name, entry)

            # 5-3. Save current output
            for prediction, output in zip(generated, complete_output):
                output_handler.save_prediction_and_output(prediction, output, index)
                index = index + 1

        # 6. Output 
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return [sample['prediction'] for sample in output_handler.results_dict.values()], summary


    def add_instruction_prompt_additions_and_sos_token(self, input_texts):
        # Add on system prompts and special tokens if needed.
        if "mistral" in self.model_name:
            if self.use_instruction:
                messages = [
                    {"role": "user", "content": PROMPT_INSTRUCTIONS[self.dataset_name]},
                    #{"role": "assistant", "content": ASSISTANT_PROMPTS[self.dataset_name]},
                ]

                instruction = self.tokenizer.apply_chat_template(messages, tokenize=False)
                prompts = [instruction + text for text in input_texts]

            else:
                prompts = ["<s>" + text for text in input_texts]
        elif "llama" in self.model_name:
                messages = [
                    {"role": "system", "content": PROMPT_INSTRUCTIONS[self.dataset_name]},
                ]

                instruction = self.tokenizer.apply_chat_template(messages, tokenize=False)
                prompts = [instruction + text for text in input_texts]

        elif "gemma" in self.model_name:
            if self.use_instruction:
                messages = [
                    {"role": "user", "content": PROMPT_INSTRUCTIONS[self.dataset_name]},
                    #{"role": "assistant", "content": ASSISTANT_PROMPTS[self.dataset_name]},
                ]

                instruction = self.tokenizer.apply_chat_template(messages, tokenize=False)
                prompts = [instruction + text for text in input_texts]

            else:
                prompts = ["<bos>" + text for text in input_texts]
        else:
            raise ValueError("Model not supported...")

        return prompts