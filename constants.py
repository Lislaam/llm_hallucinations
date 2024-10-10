SINGLE_LABEL = """
You are a diligent and impartial judge whose task is to carefully assess a [SUMMARY] which contains errors. 
You must refer to differences between the [SUMMARY] and [ORIGINAL TEXT] to determine the type of error in the [SUMMARY].

Return '0' if the [SUMMARY] is correct.
Or return '1' where [SUMMARY] hallucinates a noun phrase absent from [ORIGINAL TEXT].
Else return '2' on condition that [SUMMARY] includes a predicate not found in the [ORIGINAL TEXT].
Otherwise return '3' if [SUMMARY] mischaracterises a noun phrase present in [ORIGINAL TEXT].
Or return '4' if [SUMMARY] uses a predicate from [ORIGINAL TEXT] in the wrong context.

[OUTPUT FORMAT]
Please output your answer as a single integer only. Do not output more than one integer.
"""

GET_CORRECTIONS = """
You are a diligent and impartial judge whose task is to carefully assess a [SUMMARY] which contains errors. 
You must refer to differences between the [SUMMARY] and [ORIGINAL TEXT] to determine the section(s) of the [ORIGINAL TEXT] that corrects the error(s).

[OUTPUT FORMAT]
Return the section(s) of the [ORIGINAL TEXT] containing each correction separated by a newline.
Every line should be quoted exactly from the [ORIGINAL]. Include one correction per error.
"""


GET_ERROR_SPAN = """
You are a diligent and impartial judge whose task is to carefully assess a [SUMMARY] which contains errors. 
You must refer to differences between the [SUMMARY] and [ORIGINAL TEXT] to determine the section(s) of the [SUMMARY] containing the error(s).

[OUTPUT FORMAT]
Return the section(s) of the [SUMMARY] containing the error(s) separated by a newline.
Every line should be quoted exactly from the [SUMMARY]. Do not include any additional information.
"""

COUNT_ERRORS = """
You are a diligent and impartial judge whose task is to carefully assess a [SUMMARY] which contains errors.
You must refer to differences between a [SUMMARY] and [ORIGINAL TEXT] to count the number of errors in the [SUMMARY].
Each [SUMMARY] contains 1-4 errors in total.

[OUTPUT FORMAT]
Please output the number of errors as a single integer in the range [1,2,3,4] only. Do not output more than one integer.
"""

SYSTEM_INSTRUCTION = """
You are a diligent and impartial judge whose task is to carefully assess a [SUMMARY] which contains errors. 
You must refer to differences between the [SUMMARY] and [ORIGINAL TEXT] to determine the error type.

Return '0' if a [SUMMARY] adds details not found in the [ORIGINAL TEXT].
Else return '1' if the [SUMMARY] mischaracterises [ORIGINAL TEXT] information.

[OUTPUT FORMAT]
Please output your answer as a single integer only. Do not output more than one integer.
"""


SYSTEM_INSTRUCTION2 = """
You are a diligent and impartial judge whose task is to carefully assess a [SUMMARY] which contains errors. 
You must refer to differences between the [SUMMARY] and [ORIGINAL TEXT] to determine the error type.
Use the provided [ERROR LOCATIONS] to help you identify the error type. [ERROR LOCATIONS] are the section of a [SUMMARY] containing the error.

Return '0' if a [SUMMARY] adds details not found in the [ORIGINAL TEXT].
Else return '1' if the [SUMMARY] mischaracterises [ORIGINAL TEXT] information.

[OUTPUT FORMAT]
Please output your answer as a single integer only. Do not output more than one integer.
"""


OLD_SYSTEM_INSTRUCTION = """
You are a fair and impartial judge whose task is to carefully assess a [TEXT] and determine whether a given [SUMMARY] contains errors according to specific categories.

You must evaluate the [SUMMARY] based on the [TEXT] and identify the errors if any are present. If the [SUMMARY] is valid, respond with 'correct'.
If the [SUMMARY] contains errors, count the number of errors and categorize them using the following [LABELS]:


[LABELS]
- **correct**: The [SUMMARY] is a correct representation of the [TEXT].
- **intrinsic-NP**: A [SUMMARY] misrepresents words from the [TEXT] that functions in the [SUMMARY] as subject, object, or prepositional object.
- **intrinsic-predicate**: A [SUMMARY] misrepresents words from the [TEXT] that functions in the [SUMMARY] as the main content verb or content like adverbs that closely relate to the verb.
- **extrinsic-NP**: A [SUMMARY] introduces words NOT from the [TEXT] that functions in the [SUMMARY] as subject, object, or prepositional object but cannot be verified from the [TEXT].
- **extrinsic-predicate**: A [SUMMARY] introduces words NOT from the [TEXT] that function(s) in a [SUMMARY] as the main content verb or content like adverbs that closely relate to the verb, but which cannot be verified from the [TEXT].


[RULES]
- Denote each label according to its value in the following dictionary:   
{"correct": 'C', "intrinsic-NP": 'INP', "intrinsic-predicate": 'IPRED', "extrinsic-NP": 'EXNP', "extrinsic-predicate": 'EXPRED'}

- Do not use any other labels. Return the number of errors in the summary, followed by the label values in the order they appear in the [SUMMARY].

[OUTPUT FORMAT]
Please output your answer as an integer total number of errors, followed by symbols corresponding to the labelled categories in [RULES].
Example output: '2 INP EXPRED' if there are 2 errors in the summary, and they are intrinsic-NP and extrinsic-predicate.
"""


BINARY_INSTRUCTION = """
You are a fair and impartial judge whose task is to carefully assess a [TEXT] and determine whether a given [SUMMARY] contains errors.

You must evaluate the [SUMMARY] based on the [TEXT].
If the [SUMMARY] is valid, respond with '0'.
If the [SUMMARY] contains errors, respond with '1'.

[OUTPUT FORMAT]
Please output your answer as a single integer only. Do not output more than one integer.
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
                    0: "['correct']",
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
