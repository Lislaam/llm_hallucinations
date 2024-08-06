SYSTEM_INSTRUCTION = """
You are a language model asked to determine if a given summary is valid with respect to some source text.
If the summary is valid, please return "correct".

If the summary is invalid, you must categorise and return the type of error choosing from the below:
### Categories
1. Intrinsic: Summary text that directly conflicts with the source text.
2. Extrinsic: Summary text that is not found in the source text.
3. Noun-Phrase: Summary text contains new noun-phrases or incorrect use of noun-phrases. We will abbreviate the word Noun-Phrase to NP.
4. Predicate: Summary text contains an incorrect use of predicates.

The following errors can be combined:
### Rules
1. Intrinsic and NP.
2. Intrinsic and Predicate.
3. Extrinsic and NP.
4. Extrinsic and Predicate.

The following errors MUST NOT be combined:
1. Intrinsic and Extrinsic.
2. NP and Predicate.

Keep in mind that the given summary may contain more than one error. You must identify all errors.
You should output the answer from the following: ["correct", "intrinsic-NP", "intrinsic-predicate", "extrinsic-NP", "extrinsic-predicaate"].
"""

PROMPT_INSTRUCTIONS = {"Lislaam/AggreFact": SYSTEM_INSTRUCTION }

DATASET_PROMPTS = {"Lislaam/AggreFact": {"input": ["doc", "summ"], "output": "error_type"}}

DATASET_LABELS = {"Lislaam/AggreFact": {
                    0: "correct",
                    1: "intrinsic-NP",
                    2: "intrinsic-predicate",
                    3: "extrinsic-NP",
                    4: "extrinsic-predicate" 
                    }
                  }

PRE_POST_LABEL_TOKENS = {
    "mistralai/Mistral-7B-Instruct-v0.3": ["[/INST]", "</s>"],
    "meta-llama/Meta-Llama-3-8B-Instruct": [
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|eot_id|>",
    ],
    "google/gemma-1.1-7b-it": ["<start_of_turn>model\n", "<end_of_turn>"],
}

UNCERTAINTY_DOMAINS = [
    "log_token_neg_likelihoods",
    "token_entropies",
    "log_label_neg_likelihoods",
    "label_entropies",
]  # "embeddings",

BASELINE_METRICS = [
    "token_entropies_baseline",
    "label_entropies_baseline",
]

######################################
# Mr Lamb's stuff
######################################
# ERROR_DETECTION_INSTRUCTION = "You are asked to determine if the given text is a valid summary of the source text. "

# "Determine if the given summary contains an error with respect to the given source text. If an error is present, categorise it using one or more of these options [intrinsic, extrinsic, NP, predicate]"

# ### Mr Lamb stuff I don't need vvvvvvv
# SENTIMENT_ANALYSIS_INSTRUCTION = "Determine the sentiment of the given text. Choose the answer from the options [9Y, 3A]."
# BOOLQ_INSTRUCTION = "Answer the question. Choose the answer from the options [4I, 1O]."
# MNLI_INSTRUCTION = "Determine the relationship between the premise and hypothesis. Choose the answer from the options [5U, 3B, 6K]."
# AG_NEWS_INSTRUCTION = "Determine the category of the news article. Choose the answer from the options [8J, 2V, 9E, 5P]."
# COLA_INSTRUCTION = "Determine the grammaticality of the sentence. Choose the answer from the options [1C, 2A]."
# SST5_INSTRUCTION = "Determine the sentiment of the given text. Choose the answer from the options [4F, 6H, 9O, 3X, 5W]."
# IMDB_INSTRUCTION = "Determine the sentiment of the given text. Choose the answer from the options [8R, 7Z]."
# HANS_INSTRUCTION = "Determine the relationship between the premise and hypothesis. Choose the answer from the options [E7, J2]."
# QQP_INSTRUCTION = "Determine if the two questions are paraphrases. Choose the answer from the options [Z0, D3]."
# PAWS_INSTRUCTION = "Determine if the two sentences are paraphrases. Choose the answer from the options [J7, O1]."

# PROMPT_INSTRUCTIONS = {
#     "stanfordnlp/sst2": SENTIMENT_ANALYSIS_INSTRUCTION,
#     "google/boolq": BOOLQ_INSTRUCTION,
#     "SetFit/mnli": MNLI_INSTRUCTION,
#     "SetFit/ag_news": AG_NEWS_INSTRUCTION,
#     "linxinyuan/cola": COLA_INSTRUCTION,
#     "SetFit/sst5": SST5_INSTRUCTION,
#     "ajaykarthick/imdb-movie-reviews": IMDB_INSTRUCTION,
#     "jhu-cogsci/hans": HANS_INSTRUCTION,
#     "DT4LM/qqp": QQP_INSTRUCTION,
#     "google-research-datasets/paws": PAWS_INSTRUCTION,
#     ##### ^^^^^^ Mr Lamb stuff I throw away

#     "Lislaam/AggreFact": ERROR_DETECTION_INSTRUCTION
# }

# ERROR_DETECTION_ASSISTANT_PROMPT = "You have asked me to identify and categorise errors in the given summary. I will categorise the errors using one or more of [intrinsic, extrinsic, NP, predicate]"

# # Mr Lamb stuff: Prompts used when we don't have system prompts available for the model.
# SENTIMENT_ANALYSIS_ASSISTANT_PROMPT = "You have asked me to determine the sentiment of the given text. I will choose the answer from the options [9Y, 3A]."
# BOOLQ_ASSISTANT_PROMPT = "You have asked me to answer the question. I will choose the answer from the options [4I, 1O]."
# MNLI_ASSISTANT_PROMPT = "You have asked me to determine the relationship between the premise and hypothesis. I will choose the answer from the options [5U, 3B, 6K]."
# AG_NEWS_ASSISTANT_PROMPT = "You have asked me to determine the category of the news article. I will choose the answer from the options [8J, 2V, 9E, 5P]."
# COLA_ASSISTANT_PROMPT = "You have asked me to determine the grammaticality of the sentence. I will choose the answer from the options [1C, 2A]."
# SST5_ASSISTANT_PROMPT = "You have asked me to determine the sentiment of the given text. I will choose the answer from the options [4F, 6H, 9O, 3X, 5W]."
# IMDB_ASSISTANT_PROMPT = "You have asked me to determine the sentiment of the given text. I will choose the answer from the options [8R, 7Z]."
# HANS_ASSISTANT_PROMPT = "You have asked me to determine the relationship between the premise and hypothesis. I will choose the answer from the options [E7, J2]."
# QQP_ASSISTANT_PROMPT = "You have asked me to determine if the two questions are paraphrases. I will choose the answer from the options [Z0, D3]."
# PAWS_ASSISTANT_PROMPT = "You have asked me to determine if the two sentences are paraphrases. I will choose the answer from the options [J7, O1]."


# # Mr Lamb stuff. Ignore it:
# NLI_ADDITION = """
# ### Pay attention to:
# - Ensuring that entities (people, places, things), their actions and events described in the hypothesis match those in the premise without overgeneralizing or misrepresenting.
# - Recognizing and correctly interpreting negations and their implications.
# - Identifying and correctly interpreting conditional (if-then) and hypothetical statements.
# - Verifying that inferred actions or events logically follow from the premise.
# - Understanding complex sentence structures and relationships between clauses.
# - Recognizing and correctly interpreting idiomatic and colloquial expressions, understanding their intended meanings rather than their literal interpretations.
# - Understanding and correctly interpreting the temporal relationships between events mentioned in the premise and hypothesis.
# - Recognizing and interpreting connectives (e.g., "because," "although," "therefore") and understand how they shape the logical relationship between clauses.
# - Applying commonsense reasoning to assess the plausibility of statements, leveraging background knowledge about the world.

# ### DO NOT:
# - Assume that a premise entails all hypotheses constructed from words in the premise.
# - Assume that a premise entails all of its contiguous subsequences.
# - Assume that a negation word in the premise but not in the hypothesis leads to a prediction of contradiction. 
# - Assume that if the hypothesis is a constituent of the premise, the prediction is entailment."""


# SENTIMENT_ADDITION = """
# ### Pay attention to:
# - Understanding the entire text and not just focussing on the presence of individual words to determine the overall sentiment.
# - Balancing and taking into account the positive and negative parts of a statement to decide if the sentiment is positive or negative overall.
# - Recognizing context to accurately interpret the sentiment, especially in cases of sarcasm or irony.
# - Handling negations properly, understanding how they modify the meaning of surrounding words.
# - Recognizing and correctly interpreting domain-specific jargon and idiomatic expressions.
# - Analyzing the syntactic structure to understand the relationship between words and phrases, especially paying attention to the sentiment within and across clauses.

# ### DO NOT:
# - Assume that the pressence of a negation (not, none, 't) is indicative of negative sentiment. 
# - Focus solely on using the sentiment of indvidual words (e.g. creepy, horrifies, ) to determine sentiment
# - Assume the precences of certain punctuation (e.g. ...) is indictavtive of sentiment (e.g. negative).
# - Misinterpret sarcasm or irony by taking words at their face value.
# - Depend exclusively on the frequency of positive or negative words to judge sentiment without understanding their contextual usage."""


# PARAPHRASE_ADDITION = """
# ### Pay attention to:
# - Ensure entities (people, places, things) and their actions/events are consistently represented in both sentences.
# - Recognize and correctly interpret negations and their implications in both sentences.
# - Identify and correctly interpret conditional ("if-then") and hypothetical statements.
# - Verify that inferred actions or events logically align between the two sentences.
# - Understand and correctly interpret complex sentence structures and relationships between clauses.
# - Recognize and correctly interpret idiomatic and colloquial expressions, ensuring intended meanings are understood.
# - Correctly interpret temporal relationships between events in both sentences.

# ### DO NOT:
# - Assume that if the two sentences have a high degree of lexical overlap, they are necessarily paraphrases.
# - Assume that if two sentences have the same named entities (names, places, etc.), they are necessarily paraphrases.
# - Miss subtleties in negation or sarcasm that could change the meaning of a sentences.
# """


# NLI_FOCUS_ADDITION = """
# ### Pay attention to:
# - Ensuring that entities (people, places, things), their actions and events described in the hypothesis match those in the premise without overgeneralizing or misrepresenting.
# - Recognizing and correctly interpreting negations and their implications.
# - Identifying and correctly interpreting conditional (if-then) and hypothetical statements.
# - Verifying that inferred actions or events logically follow from the premise.
# - Understanding complex sentence structures and relationships between clauses.
# - Recognizing and correctly interpreting idiomatic and colloquial expressions, understanding their intended meanings rather than their literal interpretations.
# - Understanding and correctly interpreting the temporal relationships between events mentioned in the premise and hypothesis.
# - Recognizing and interpreting connectives (e.g., "because," "although," "therefore") and understand how they shape the logical relationship between clauses.
# - Applying commonsense reasoning to assess the plausibility of statements, leveraging background knowledge about the world."""



# SENTIMENT_FOCUS_ADDITION = """
# ### Pay attention to:
# - Understanding the entire text and not just focussing on the presence of individual words to determine the overall sentiment.
# - Balancing and taking into account the positive and negative parts of a statement to decide if the sentiment is positive or negative overall.
# - Recognizing context to accurately interpret the sentiment, especially in cases of sarcasm or irony.
# - Handling negations properly, understanding how they modify the meaning of surrounding words.
# - Recognizing and correctly interpreting domain-specific jargon and idiomatic expressions.
# - Analyzing the syntactic structure to understand the relationship between words and phrases, especially paying attention to the sentiment within and across clauses."""


# PARAPHRASE_FOCUS_ADDITION = """
# ### Pay attention to:
# - Ensure entities (people, places, things) and their actions/events are consistently represented in both sentences.
# - Recognize and correctly interpret negations and their implications in both sentences.
# - Identify and correctly interpret conditional ("if-then") and hypothetical statements.
# - Verify that inferred actions or events logically align between the two sentences.
# - Understand and correctly interpret complex sentence structures and relationships between clauses.
# - Recognize and correctly interpret idiomatic and colloquial expressions, ensuring intended meanings are understood.
# - Correctly interpret temporal relationships between events in both sentences."""


# NLI_PROHIBIT_ADDITION = """
# ### DO NOT:
# - Assume that a premise entails all hypotheses constructed from words in the premise.
# - Assume that a premise entails all of its contiguous subsequences.
# - Assume that a negation word in the premise but not in the hypothesis leads to a prediction of contradiction. 
# - Assume that if the hypothesis is a constituent of the premise, the prediction is entailment."""

# SENTIMENT_PROHIBIT_ADDITION = """
# ### DO NOT:
# - Assume that the pressence of a negation (not, none, 't) is indicative of negative sentiment. 
# - Focus solely on using the sentiment of indvidual words (e.g. creepy, horrifies, ) to determine sentiment
# - Assume the precences of certain punctuation (e.g. ...) is indictavtive of sentiment (e.g. negative).
# - Misinterpret sarcasm or irony by taking words at their face value.
# - Depend exclusively on the frequency of positive or negative words to judge sentiment without understanding their contextual usage."""

# PARAPHRASE_PROHIBIT_ADDITION = """
# ### DO NOT:
# - Assume that if the two sentences have a high degree of lexical overlap, they are necessarily paraphrases.
# - Assume that if two sentences have the same named entities (names, places, etc.), they are necessarily paraphrases.
# - Miss subtleties in negation or sarcasm that could change the meaning of a sentences."""


# FOCUS_ADDITIONS = {
#     "stanfordnlp/sst2": SENTIMENT_FOCUS_ADDITION,
#     "SetFit/mnli": NLI_FOCUS_ADDITION,
#     "SetFit/sst5": SENTIMENT_FOCUS_ADDITION,
#     "ajaykarthick/imdb-movie-reviews": SENTIMENT_FOCUS_ADDITION,
#     "jhu-cogsci/hans": NLI_FOCUS_ADDITION,
#     "DT4LM/qqp": PARAPHRASE_FOCUS_ADDITION,
#     "google-research-datasets/paws": PARAPHRASE_FOCUS_ADDITION,
# }

# PROHIBIT_ADDITIONS = {
#     "stanfordnlp/sst2": SENTIMENT_PROHIBIT_ADDITION,
#     "SetFit/mnli": NLI_PROHIBIT_ADDITION,
#     "SetFit/sst5": SENTIMENT_PROHIBIT_ADDITION,
#     "ajaykarthick/imdb-movie-reviews": SENTIMENT_PROHIBIT_ADDITION,
#     "jhu-cogsci/hans": NLI_PROHIBIT_ADDITION,
#     "DT4LM/qqp": PARAPHRASE_PROHIBIT_ADDITION,
#     "google-research-datasets/paws": PARAPHRASE_PROHIBIT_ADDITION,
# }

# ASSISTANT_PROMPTS = {
#     "stanfordnlp/sst2": SENTIMENT_ANALYSIS_ASSISTANT_PROMPT,
#     "google/boolq": BOOLQ_ASSISTANT_PROMPT,
#     "SetFit/mnli": MNLI_ASSISTANT_PROMPT,
#     "SetFit/ag_news": AG_NEWS_ASSISTANT_PROMPT,
#     "linxinyuan/cola": COLA_ASSISTANT_PROMPT,
#     "SetFit/sst5": SST5_ASSISTANT_PROMPT,
#     "ajaykarthick/imdb-movie-reviews": IMDB_ASSISTANT_PROMPT,
#     "jhu-cogsci/hans": HANS_ASSISTANT_PROMPT
# }

# # DATASET_LABELS = {True:{
# #     "stanfordnlp/sst2": {0: "negative", 1: "positive"},
# #     "google/boolq": {0: "false", 1: "true"},
# #     "SetFit/mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
# #     "SetFit/ag_news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
# #     "linxinyuan/cola": {0: "ungrammatical", 1: "grammatical"},
# #     "stanfordnlp/imdb": {0: "negative", 1: "positive"},
# #     "SetFit/sst5": {
# #         0: "very negative",
# #         1: "negative",
# #         2: "neutral",
# #         3: "positive",
# #         4: "very positive",
# #     },
# # }, False:{
# #     "stanfordnlp/sst2": {"negative": "0", "positive": "1"},
# #     "google/boolq": {"false": "0", "true": "1"},
# #     "SetFit/mnli": {"entailment": "0", "neutral": "1", "contradiction": "2"},
# #     "SetFit/ag_news": {"World": "0", "Sports": "1", "Business": "2", "Sci/Tech": "3"},
# #     "linxinyuan/cola": {"ungrammatical": "0", "grammatical": "1"},
# #     "stanfordnlp/imdb": {"negative": "0", "positive": "1"},
# #     "SetFit/sst5": {
# #         "very negative": "0",
# #         "negative": "1",
# #         "neutral": "2",
# #         "positive": "3",
# #         "very positive": "4",
# #     },
# # }}


# DATASET_LABELS = {
#     # Two types of labels, binary (symbolic) or 'verbalised labels' (+ve / -ve sentiment)
#     # I don't care about verbalised labels
#     True: {
#         "stanfordnlp/sst2": {0: "9Y", 1: "3A"},
#         "google/boolq": {0: "4I", 1: "1O"},
#         "SetFit/mnli": {0: "5U", 1: "3B", 2: "6K"},
#         "SetFit/ag_news": {0: "8J", 1: "2V", 2: "9E", 3: "5P"},
#         "linxinyuan/cola": {0: "1C", 1: "2A"},
#         "ajaykarthick/imdb-movie-reviews": {0: "8R", 1: "7Z"},
#         "SetFit/sst5": {
#             0: "4F",
#             1: "6H",
#             2: "9O",
#             3: "3X",
#             4: "5W",
#         },
#         "jhu-cogsci/hans": {0: "E7", 1: "J2"},
#         "DT4LM/qqp": {0: "Z0", 1: "D3"},
#         "google-research-datasets/paws": {0: "J7", 1: "O1"},
#         # Mr Lamb ^
        
#         "Lislaam/AggreFact": {0: "correct"}
#     },
#     False: {
#         "stanfordnlp/sst2": {"negative": "9Y", "positive": "3A"},
#         "google/boolq": {"false": "4I", "true": "1O"},
#         "SetFit/mnli": {"entailment": "5U", "neutral": "3B", "contradiction": "6K"},
#         "SetFit/ag_news": {
#             "World": "8J",
#             "Sports": "2V",
#             "Business": "9E",
#             "Sci/Tech": "5P",
#         },
#         "linxinyuan/cola": {"ungrammatical": "1C", "grammatical": "2A"},
#         "ajaykarthick/imdb-movie-reviews": {"negative": "8R", "positive": "7Z"},
#         "SetFit/sst5": {
#             "very negative": "4F",
#             "negative": "6H",
#             "neutral": "9O",
#             "positive": "3X",
#             "very positive": "5W",
#         },
#         "jhu-cogsci/hans": {"E7": "entailment", "J2": "non-entailment"},
#         "DT4LM/qqp": {"Z0": "paraphrase", "D3": "not paraphrase"},
#         "google-research-datasets/paws": {"J7": "paraphrase", "O1": "not paraphrase"},
#     },
# }

# REVERSED_DATASET_LABELS = {
#     True: {
#         "stanfordnlp/sst2": {"9Y": 0, "3A": 1},
#         "google/boolq": {"4I": 0, "1O": 1},
#         "SetFit/mnli": {"5U": 0, "3B": 1, "6K": 2},
#         "SetFit/ag_news": {"8J": 0, "2V": 1, "9E": 2, "5P": 3},
#         "linxinyuan/cola": {"1C": 0, "2A": 1},
#         "ajaykarthick/imdb-movie-reviews": {"8R": 0, "7Z": 1},
#         "SetFit/sst5": {
#             "4F": 0,
#             "6H": 1,
#             "9O": 2,
#             "3X": 3,
#             "5W": 4,
#         },
#         "jhu-cogsci/hans": {"E7": 0, "J2": 1},
#         "DT4LM/qqp": {"Z0": 0, "D3": 1},
#         "google-research-datasets/paws": {"J7": 0, "O1": 1},
#     },
#     False: {
#         "stanfordnlp/sst2": {"9Y": "negative", "3A": "positive"},
#         "google/boolq": {"4I": "false", "1O": "true"},
#         "SetFit/mnli": {"5U": "entailment", "3B": "neutral", "6K": "contradiction"},
#         "SetFit/ag_news": {
#             "8J": "World",
#             "2V": "Sports",
#             "9E": "Business",
#             "5P": "Sci/Tech",
#         },
#         "linxinyuan/cola": {"1C": "ungrammatical", "2A": "grammatical"},
#         "ajaykarthick/imdb-movie-reviews": {"8R": "negative", "7Z": "positive"},
#         "SetFit/sst5": {
#             "4F": "very negative",
#             "6H": "negative",
#             "9O": "neutral",
#             "3X": "positive",
#             "5W": "very positive",
#         },
#         "jhu-cogsci/hans": {"E7": "entailment", "J2": "non-entailment"},
#         "DT4LM/qqp": {"Z0": "paraphrase", "D3": "not paraphrase"},
#         "google-research-datasets/paws": {"J7": "paraphrase", "O1": "not paraphrase"},
#     },
# }

# UNCERTAINTY_DOMAINS = [
#     "log_token_neg_likelihoods",
#     "token_entropies",
#     "log_label_neg_likelihoods",
#     "label_entropies",
# ]  # "embeddings",

# BASELINE_METRICS = [
#     "token_entropies_baseline",
#     "label_entropies_baseline",
# ]

# TEST_SPLIT = {
#     "SetFit/ag_news": "test",
#     "SetFit/mnli": "validation",
#     "google/boolq": "validation",
#     "stanfordnlp/sst2": "validation",
#     "linxinyuan/cola": "test",
#     "SetFit/sst5": "test",
#     "ajaykarthick/imdb-movie-reviews": "test",
#     "jhu-cogsci/hans": "validation",
#     "DT4LM/qqp": "test",
#     "google-research-datasets/paws": "test",
#     # ^^^ Mr Lamb

#     "Lislaam/AggreFact": "val"
# }

# PRE_POST_LABEL_TOKENS = {
#     "mistralai/Mistral-7B-Instruct-v0.3": ["[/INST]", "</s>"],
#     "meta-llama/Meta-Llama-3-8B-Instruct": [
#         "assistant<|end_header_id|>\n\n",
#         "<|eot_id|>",
#     ],
#     "google/gemma-1.1-7b-it": ["<start_of_turn>model\n", "<end_of_turn>"],
# }

# DATASET_PROMPTS = {
#     "stanfordnlp/sst2": {"input": "Input", "output": "Sentiment"},
#     "google/boolq": {"input": "Question", "output": "Answer"},
#     "SetFit/mnli": {"input": "Premise and Hypothesis", "output": "Relationship"},
#     "SetFit/ag_news": {"input": "News Article", "output": "Category"},
#     "linxinyuan/cola": {"input": "Sentence", "output": "Grammaticality"},
#     "SetFit/sst5": {"input": "Input", "output": "Sentiment"},
#     "ajaykarthick/imdb-movie-reviews": {"input": "Review", "output": "Sentiment"},
#     "jhu-cogsci/hans": {"input": "Premise and Hypothesis", "output": "Relationship"},
#     "DT4LM/qqp": {"input": "Question Pair", "output": "Paraphrase"},
#     "google-research-datasets/paws": {"input": "Sentence Pair", "output": "Paraphrase"},

#     # Give the model some writing and a summary. Want it to guess what type of error is present.
#     "Lislaam/AggreFact": {"input": ["doc", "summ"], "output": "error_type"}
# }
