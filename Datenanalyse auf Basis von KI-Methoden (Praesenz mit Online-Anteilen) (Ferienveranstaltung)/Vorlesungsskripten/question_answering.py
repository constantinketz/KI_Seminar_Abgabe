
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

path = "deutsche-telekom/bert-multi-english-german-squad2"
tokenizer = AutoTokenizer.from_pretrained(path)

model = AutoModelForQuestionAnswering.from_pretrained(path)

text="<p>Sie können einen Zuschuss beantragen, wenn Sie nicht über ausreichendes Einkommen verfügen, um Ihren Wohnraum zu bezahlen.</p>\n<p><span class=\"sbw-langtext\">Das Wohngeld für Mieterinnen und Mieter heißt Mietzuschuss, das Wohngeld für Eigentümerinnen und Eigentümer von selbst genutztem Wohnraum heißt Lastenzuschuss.</span></p>\n<p><strong>Höhe:</strong></p>\n<p>Abhängig vom Einzelfall.<br>Es orientiert sich an der Haushaltsgröße, dem Einkommen und der Miete beziehungsweise Belastung.</p>\n<p><strong>Dauer:</strong></p>\n<p>In der Regel für 12 Monate.</p>\n<p>Im Einzelfall kann dieser Zeitraum länger oder kürzer sein. <br>Wollen Sie Wohngeld nach diesem Zeitraum weiter beziehen, müssen Sie es neu beantragen.</p>"

questions = [
    "für welche Dauer kann ich Wohngeld beantragen",
    "Wohngeld Bewilligungszeitraum"

]

for question in questions:
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)[0], model(**inputs)[1]

    answer_start = torch.argmax(
        answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")