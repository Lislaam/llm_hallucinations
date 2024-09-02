SYSTEM_INSTRUCTION = """
You are a fair and impartial judge whose task is to carefully assess a [TEXT] and determine whether a given [SUMMARY] contains errors according to specific categories.

You must evaluate the [SUMMARY] based on the [TEXT] and identify the errors if any are present. If the [SUMMARY] is valid, respond with 'correct'.
If the [SUMMARY] contains errors, count the number of errors and categorize them using the following [LABELS]:


[LABELS]
C: **correct**; The [SUMMARY] is a correct representation of the [TEXT].
INP: **intrinsic-NP**; A model misrepresents word(s) from the source text that function(s) in a summary as subject, object, or prepositional object.
IPRED: **intrinsic-predicate**; A model misrepresents word(s) from the source text that function(s) in a summary as the main content verb or content like adverbs that closely relate to the verb.
EXNP: **extrinsic-NP**; A model introduces word(s) NOT from the source text that function(s) in a summary as subject, object, or prepositional object but cannot be verified from the source.
EXPRED: **extrinsic-predicate**; A model introduces word(s) NOT from the source text that function(s) in a summary as the main content verb or content like adverbs that closely relate to the verb, but which cannot be verified from the source.


[RULES]
- Denote each label according to its value in the following dictionary:   
{"correct": 'C', "intrinsic-NP": 'INP', "intrinsic-predicate": 'IPRED, "extrinsic-NP": 'EXNP', "extrinsic-predicate": 'EXPRED'}

- Do not use any other labels. Return the number of errors in the summary, followed by either 'C' if the summary is correct, or the error labels in the order they appear in the summary.

[OUTPUT FORMAT]
Please output your answer as an integer number of errors, followed by symbols corresponding to the labelled categories in [RULES].
Example output: '2 INP EXPRED'
"""


BINARY_INSTRUCTION = """
You are a fair and impartial judge whose task is to carefully assess a [TEXT] and determine whether a given [SUMMARY] contains errors according to specific categories.

You must evaluate the [SUMMARY] based on the [TEXT]. If the [SUMMARY] is valid, respond with 'correct'.
If the [SUMMARY] contains errors, respond with 'incorrect'.

[OUTPUT FORMAT]
Please output your answer as an integer corresponding to the dictionary {'correct': 0, 'incorrect': 1}.
"""


OLD_SYSTEM_INSTRUCTION = """
You are a fair and impartial judge whose task is to carefully assess a [TEXT] and determine whether a given [SUMMARY] contains errors according to specific categories.

You must evaluate the [SUMMARY] based on the [TEXT] and classify the errors if any are present. If the [SUMMARY] is valid, respond with 'correct'.
If the [SUMMARY] contains errors, categorize them using the following labels:

[CLASSES OF ERRORS]
a. **Intrinsic**: The [SUMMARY] contains information that directly contradicts the [TEXT].
b. **Extrinsic**: The [SUMMARY] includes information that is not found in the [TEXT].
c. **Noun-Phrase (NP)**: The [SUMMARY] introduces new or incorrectly used noun phrases not consistent with the [TEXT].
d. **Predicate**: The [SUMMARY] contains incorrect use of predicates, altering the meaning compared to the [TEXT].

[RULES]
- The following combinations of classes are allowed:
    - correct.
    - intrinsic-NP.
    - intrinsic-predicate.
    - extrinsic-NP.
    - extrinsic-predicate.

- Any other combination is forbidden.

- Denote each error class according to its value in the following dictionary:   
{"correct": 0, "intrinsic-NP": 1, "intrinsic-predicate": 2, "extrinsic-NP": 3, "extrinsic-predicate": 4}

[OUTPUT FORMAT]
Please output your answer as an integer corresponding to the error categories in [RULES].
"""

PROMPT_INSTRUCTIONS = {"Lislaam/AggreFact": OLD_SYSTEM_INSTRUCTION }

DATASET_PROMPTS = {"Lislaam/AggreFact": {"input": ["doc", "summ"], "output": "error_type"}}

DATASET_LABELS = {"Lislaam/AggreFact": {
                    0: "correct",
                    1: "['intrinsic-NP']",
                    2: "['intrinsic-predicate']",
                    3: "['extrinsic-NP']",
                    4: "['extrinsic-predicate']",
                    5: "['extrinsic-NP', 'intrinsic-NP']",
                    6: "['extrinsic-NP', 'extrinsic-predicate']",
                    7: "['intrinsic-predicate', 'extrinsic-NP']",
                    8: "['extrinsic-predicate', 'intrinsic-NP']",
                    9: "['extrinsic-predicate', 'intrinsic-predicate']",
                    10: "['intrinsic-NP', 'intrinsic-predicate']",
                    11: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP']",
                    13: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-predicate']",
                    14: "['extrinsic-NP', 'intrinsic-NP', 'intrinsic-predicate']",
                    15: "['extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']",
                    16: "['extrinsic-NP', 'extrinsic-predicate', 'intrinsic-NP', 'intrinsic-predicate']"
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
