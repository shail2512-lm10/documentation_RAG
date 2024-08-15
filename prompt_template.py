qa_prompt_tmpl_str = (
"I want you to think step by step to answer the below query in a crisp manner.\n"
"Query: {query}\n"
"And if you require any help from context to answer the query, below is some context for you.\n"
"{context}"
"----------"
"Incase case you don't know the answer only say 'I don't know!'.\n"
"Answer: "
)