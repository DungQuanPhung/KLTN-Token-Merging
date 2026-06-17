UOS_LINGUISTIC_PROMPT = """Split a hotel review sentence into unit sentences — one unit per aspect term.

Return a JSON array of strings. No explanation, no extra text.


## DEFINITION

An **aspect term** is a noun or noun phrase that refers to a specific hotel entity or feature,
such as: room, staff, service, food, breakfast, pool, view, location, price, bed, bathroom, lobby, elevator.


## GOAL

Given a hotel review sentence, identify every distinct aspect term mentioned.
For each aspect term, extract the clause that describes it as a complete standalone sentence.


## RULES

1. One unit per aspect term — each output sentence must focus on exactly ONE aspect term.
2. Preserve the original words — do NOT paraphrase or substitute any word.
3. Make each unit a complete standalone sentence if it is a fragment:
   - Add "The" or "A" as a subject if missing
   - Add a copula ("is", "was", "are", "were") if missing
4. If a shared modifier applies to multiple aspect terms, copy it to each relevant unit.
5. If the sentence mentions only ONE aspect term, return a single-element array.
6. Never invent words that are not present in the input.
7. If a single modifier describes a group of aspect terms joined by "and" that together form one compound concept (e.g. "staff and customer service", "food and drinks"), do NOT split — keep the whole group as one unit.
   - Split when each noun can stand alone as an independent aspect (e.g. "room and bed", "pool and gym").
   - Do NOT split when the nouns form a tightly coupled pair that is evaluated together as a single entity.


## EXAMPLES

Input: "The room was clean but the breakfast was cold."
Output: ["The room was clean.", "The breakfast was cold."]

Input: "The room was spacious but noisy."
Output: ["The room was spacious but noisy."]

Input: "Lovely hotel room, good beds, nice staff."
Output: ["Lovely hotel room.", "Good beds.", "Nice staff."]

Input: "The restaurant is spacious, the view is beautiful, and the food is amazing."
Output: ["The restaurant is spacious.", "The view is beautiful.", "The food is amazing."]

Input: "Very comfortable room and bed."
Output: ["Very comfortable room.", "Very comfortable bed."]

Input: "Exceptional staff and customer service."
Output: ["Exceptional staff and customer service."]

Input: "Great food and drinks."
Output: ["Great food and drinks."]

Input: "Good service."
Output: ["Good service."]

Input: "Staff is friendly from the reception to the restaurant."
Output: ["Staff is friendly."]

Input: "The room was lovely with a beautiful view and a large balcony."
Output: ["The room was lovely.", "The view was beautiful.", "The balcony was large."]

Input: "they upgraded us to a beautiful suite without us even asking."
Output: ["They upgraded us to a beautiful suite without us even asking."]

Input: "The pool is clean and the gym is well-equipped."
Output: ["The pool is clean.", "The gym is well-equipped."]


Input:
{sentence}

Output:"""
