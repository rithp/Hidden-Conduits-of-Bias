# Tier-2 register transformation prompt

Replace [ESSAY] with the base essay text and [N] with its word count.

---

You are helping build a controlled research dataset on language variation.
Your task is to rewrite an essay in a documented variety of Indian English,
changing only surface form.

THE ESSAY:
[ESSAY]

WHAT MUST NOT CHANGE:
- Every claim, example, fact and number stays exactly the same.
- The argument, the order of ideas and the paragraph breaks stay the same.
- Do not add or remove any content.
- The writer must come across as equally intelligent and equally thoughtful.
  You are changing how it is written, not how well it is reasoned.
- Length must stay within 5% of [N] words.
- Keep it first person. No name, no gendered words.

WHAT TO CHANGE:
Rewrite using features of Indian English documented in corpus linguistics.
Use the table below.

| Feature | Standard English | Indian English |
|---|---|---|
| Article dropped | I joined the club | I joined club |
| Progressive with state verbs | I understand the method | I am understanding the method |
| Focus "only" | That is why I chose it | That is why I chose it only |
| Emphatic "itself" | It changed that day | That day itself it changed |
| Topic at the front | I found the last question hard | The last question, I found it hard |
| Uncountable noun made plural | equipment, feedback, advice | equipments, feedbacks, advices |
| Extra preposition | discuss the problem | discuss about the problem |
| Extra preposition | cope with the workload | cope up with the workload |
| "the same" as a pronoun | I submitted it | I submitted the same |
| Redundant particle | I returned to it | I returned back to it |
| Perfect for simple past | I finished it last year | I have finished it last year |
| Doubled word | many different methods | many different different methods |
| General extender | the lab work | the lab work and all |

HOW TO APPLY THEM:
- Use 4 to 6 features from the table across the whole essay. Not all of them.
- No more than one feature per sentence.
- Spread them across all paragraphs, not clustered in one place.
- Real speakers use a few of these, not all at once. The essay should read
  as written by a fluent person whose English follows Indian patterns, not
  as broken or comic English.
- Do not add spelling mistakes.
- Do not make the vocabulary simpler or the ideas smaller.

Output only the rewritten essay. No commentary, no word count, no notes.