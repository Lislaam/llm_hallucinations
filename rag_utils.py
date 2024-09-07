def split_into_facts(summary):
    # Split the summary into sentence-level facts, use your custom logic if needed
    facts = summary.split('. ')  # Example: Split by period to get atomic facts
    return facts