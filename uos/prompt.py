UOS_LINGUISTIC_PROMPT = """You split hotel review sentences into Unit Aspect Sentences (UAS).

A UAS contains exactly ONE aspect term and ONE sentiment orientation toward that aspect.

Return a JSON array of strings only.
No explanation.
No extra text.

## GOAL

Split review text into minimal aspect-focused sentences suitable for Aspect-Based Sentiment Analysis (ABSA).

Split a sentence whenever:

1. The aspect term changes.
2. The sentiment toward the same aspect changes.

Do NOT split merely because multiple descriptive words are used for the same aspect if they express the same overall sentiment.

Rewrite fragments into complete standalone sentences when necessary:

* add articles ("The", "A")
* add a copula ("is", "was", "are", "were")
* preserve the original wording

Never invent aspects, sentiments, attributes, or details.

---

## HARD CONSTRAINTS

* Do NOT add information not present in the input.

* Do NOT invent aspect terms.

* Do NOT invent sentiment.

* Do NOT convert factual statements into sentiment statements.

* Do NOT split nouns that appear only inside:

  * location phrases
  * example lists
  * amenities lists
  * causal explanations
  * prepositional phrases

* Split only when:

  * aspect terms differ, OR
  * sentiment polarity differs.

---

## SPLIT

### Rule 1 — Different aspects

"The room was clean but breakfast was cold."

Output:

[
"The room was clean.",
"The breakfast was cold."
]

---

"The beds were comfortable and the room was spacious."

Output:

[
"The beds were comfortable.",
"The room was spacious."
]

---

"The room and bathroom were clean."

Output:

[
"The room was clean.",
"The bathroom was clean."
]

---

### Rule 2 — Shared sentiment across different aspects

"Beautiful room and pool."

Output:

[
"The room was beautiful.",
"The pool was beautiful."
]

---

"Very comfortable room and bed."

Output:

[
"The room was very comfortable.",
"The bed was very comfortable."
]

---

"The mattress and pillows are comfortable."

Output:

[
"The mattress was comfortable.",
"The pillows were comfortable."
]

---

### Rule 3 — Same aspect but different sentiment

"The breakfast was delicious but expensive."

Output:

[
"The breakfast was delicious.",
"The breakfast was expensive."
]

---

"The room was spacious but noisy."

Output:

[
"The room was spacious.",
"The room was noisy."
]

---

"The location was convenient but crowded."

Output:

[
"The location was convenient.",
"The location was crowded."
]

---

### Rule 4 — Distinct aspect fragments

"Lovely hotel room, good beds, nice staff."

Output:

[
"Lovely hotel room.",
"Good beds.",
"Nice staff."
]

---

"Nice bungalow and clean beach. Good food in the restaurant. Beautiful sunset."

Output:

[
"Nice bungalow.",
"Clean beach.",
"Good food in the restaurant.",
"Beautiful sunset."
]

---

### Rule 5 — Separate physical spaces introduced by 'with'

"Very spacious room with a huge bathroom."

Output:

[
"The room was very spacious.",
"The bathroom was huge."
]

---

"The room was lovely with a beautiful view and a large balcony."

Output:

[
"The room was lovely.",
"The view was beautiful.",
"The balcony was large."
]

---

"The room is comfortable, quiet, with a very large bed and a renovated bathroom."

Output:

[
"The room is comfortable and quiet.",
"The bed was very large.",
"The bathroom was renovated."
]

---

## DO NOT SPLIT

### Rule 1 — Same aspect, same sentiment

"The room was spacious, clean and comfortable."

Output:

[
"The room was spacious, clean and comfortable."
]

---

"The hotel is clean, nice, safe and quiet."

Output:

[
"The hotel is clean, nice, safe and quiet."
]

---

"The bathroom was modern and very clean."

Output:

[
"The bathroom was modern and very clean."
]

---

### Rule 2 — Holistic review statements

"Loved the whole experience, from the decor to the food to the room."

Output:

[
"Loved the whole experience, from the decor to the food to the room."
]

---

### Rule 3 — Context plus one aspect statement

"We arrived at 3pm but the room was dirty."

Output:

[
"We arrived at 3pm but the room was dirty."
]

---

### Rule 4 — Pure facts

"There are two elevators."

Output:

[
"There are two elevators."
]

---

### Rule 5 — Causal explanation

"We couldn't shower because the shower head was above the toilet and sink."

Output:

[
"We couldn't shower because the shower head was above the toilet and sink."
]

---

### Rule 6 — Amenities and accessories

"The bathroom was clean with shampoo, shower gel and hand soap."

Output:

[
"The bathroom was clean with shampoo, shower gel and hand soap."
]

---

"A large swimming pool with sun loungers."

Output:

[
"A large swimming pool with sun loungers."
]

---

### Rule 7 — View and location context

"Fantastic view over the river and promenade."

Output:

[
"The view over the river and promenade was fantastic."
]

---

"There was a rooftop pool with a great view."

Output:

[
"There was a rooftop pool with a great view."
]

---

## DECISION GUIDE

Split if:

✓ aspect term changes

✓ sentiment changes for the same aspect

Keep together if:

✓ same aspect

✓ same sentiment

✓ multiple adjectives support the same overall sentiment

✓ accessories or amenities belong to the same aspect

✓ contextual information is attached to the same aspect

---

## FINAL RULES

* Preserve original wording whenever possible.
* Never invent aspects.
* Never invent sentiment.
* Keep modifiers attached to their aspects.
* Do not classify aspect categories.
* Do not classify sentiment labels.
* Do not over-split coherent descriptions of the same aspect.

Input:
{sentence}

Output:
"""
