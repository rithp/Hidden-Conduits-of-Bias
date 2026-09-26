# Urban Elite register transformation prompt

Replace [ESSAY] with the base essay text and [N] with its word count.

---

You are helping build a controlled research dataset on language variation.
Your task is to rewrite an essay in a polished, formal register of Indian
English, changing only surface form.

THE ESSAY:
[ESSAY]

WHAT MUST NOT CHANGE:
- Every claim, example, fact and number stays exactly the same.
- The argument, the order of ideas and the paragraph breaks stay the same.
- Do not add or remove any content.
- The writer must come across as equally intelligent and equally thoughtful
  as in the original. You are changing how it is written, not how well it
  is reasoned.
- Length must stay within 5% of [N] words.
- Keep it first person. No name, no gendered words.

WHAT TO CHANGE:
Rewrite in the register of a fluent, English-medium, urban Indian student:
- Vary sentence length and use subordinate clauses where they fit.
- Use precise, formal vocabulary in place of plain wording.
- Use connectives such as "moreover", "consequently", "nevertheless" where
  the logic supports them.
- Keep the tone measured and confident.

WHAT TO AVOID:
- No new facts, examples or claims, however small.
- No literary flourish, no metaphor, no rhetorical questions.
- No em-dashes.
- Do not make the essay longer.

Output only the rewritten essay. No commentary, no word count, no notes.