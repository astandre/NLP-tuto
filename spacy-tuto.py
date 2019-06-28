import spacy

nlp = spacy.load("es_core_news_sm")

text = """El caso Receta de Arroz Verde 50 es una investigación publicada por el portal digital Mil Hojas. El portal digital reveló un correo electrónico recibido por Pamela Martínez supuesta asesora del expresidente Rafael Correa según Mil Hoja con un documento titulado Receta de Arroz Verde 502.  Según la investigación, el remitente del correo electrónico sería Geraldo Luiz Pereira de Souza- encargado de la administración y finanzas de Odebrecht en Ecuador. El mail demuestra presuntos aportes entregados por empresas multinacionales como Odebrecht al movimiento Alianza País desde noviembre de 2013 a febrero de 201 periodo en el que el expresidente Rafael Correa lideraba esa organización política. Según Mil Hojas, las donaciones alcanzarían los 11,6 millones de dólares. Las empresas que habrían realizado los aportes son: Constructora Norberto Odebrecht, SK Engineering & Construction, Sinohydro Corporation, Grupo Azul, Telconet, China International Water & Electric Corp-CWE."""

text = nlp(text)

# Tokenizar sentencias
tokenized_sentences = [sentence.text for sentence in text.sents]

print(tokenized_sentences)

# Tokenizar sentencias
tokenized_words = [word.text for word in nlp(tokenized_sentences[0])]
print(tokenized_words)

# POS tagging
tokens_pos = [word.pos_ for word in nlp(tokenized_sentences[0])]
print(tokens_pos)

# Lemmas
stemmas = [word.lemma_ for word in nlp(tokenized_sentences[0])]
print(stemmas)


# Linguistic features
for sentence in tokenized_sentences:
    for token in nlp(sentence):
        print("WORD: %s | POS: %s | LEMMA: %s" % (token.text, token.pos_, token.lemma_))
    # NER
    for entity in nlp(sentence).ents:
        print("Entity: %s | Label: %s " % (entity.text, entity.label_))

