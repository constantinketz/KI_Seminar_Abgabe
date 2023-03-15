
from transformers import pipeline


path="Sahajtomar/German_Zeroshot"
zershot_pipeline = pipeline("zero-shot-classification",
                             model=path)

sequence = "Das Informationssicherheits-Managementsystem ist in die bestehenden Strukturen der Komm.ONE integriert und trägt wesentlich zu einer hohen Qualität und Sicherheit der Produkte bei. Diese Sicherheitsleistung erstreckt sich über das gesamte Verbandsnetz und schützt die angebundenen Mitglieder und Kunden."
labels = ["Service", "Sicherheit", "Beratung"]


hypothesis_template = "In diesem Satz geht es um das Thema {}."

prediction=zershot_pipeline(sequence, labels, hypothesis_template=hypothesis_template)
print(prediction)
print(prediction['labels'][0])