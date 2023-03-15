from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

path="Einmalumdiewelt/T5-Base_GNAD"

tokenizer = AutoTokenizer.from_pretrained(path)

model = AutoModelForSeq2SeqLM.from_pretrained(path)



text_example='''
Das Elterngeld ersetzt einen Teil des entfallenden Einkommens, wenn Sie nach der Geburt für Ihr Kind da sein wollen und Ihre berufliche Arbeit unterbrechen oder einschränken. Beantragen Sie Basiselterngeld oder ElterngeldPlus oder kombiniren Sie beides. Entscheiden Sie, welcher Elternteil für welchen Zeitraum das Elterngeld in Anspruch nimmt. Geben Sie beides im Antrag an. Rückwirkende Zahlungen gewährt die L-Bank nur für die letzten drei Lebensmonate vor Antragseingang. Basiselterngeld beziehen Sie innerhalb der ersten 14 Lebensmonate Ihres Kindes. ElterngeldPlus können Sie auch über die ersten 14 Lebensmonate hinaus beziehen. Für besonders früh geborene Kinder können Sie zusätzliche Elterngeldmonate bekommen, wenn das Kind ab dem 01.09.21 geboren ist. Die L-Bank zahlt Elterngeld auch Elternteilen, die nicht erwerbstätig sind. Das Basiselterngeld für nicht Erwerbstätige beträgt mindestens 300 Euro monatlich, das Elterngeld Plus 150 Euro monatlich.

'''


tokens_input = tokenizer.encode("summarize: "+text_example, return_tensors='pt', max_length=100000000, truncation=True)
summary_ids = model.generate(tokens_input, max_length=75 + 5, min_length=75 - 20)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)

