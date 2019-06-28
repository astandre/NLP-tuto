import nltk
from nltk.tag import StanfordPOSTagger, StanfordNERTagger
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stemmer = SnowballStemmer('spanish')

stop_words = set(stopwords.words('spanish'))

text = """El caso Receta de Arroz Verde 50 es una investigación publicada por el portal digital Mil Hojas. El portal digital reveló un correo electrónico recibido por Pamela Martínez supuesta asesora del expresidente Rafael Correa según Mil Hoja con un documento titulado Receta de Arroz Verde 502.  Según la investigación, el remitente del correo electrónico sería Geraldo Luiz Pereira de Souza- encargado de la administración y finanzas de Odebrecht en Ecuador. El mail demuestra presuntos aportes entregados por empresas multinacionales como Odebrecht al movimiento Alianza País desde noviembre de 2013 a febrero de 201 periodo en el que el expresidente Rafael Correa lideraba esa organización política. Según Mil Hojas, las donaciones alcanzarían los 11,6 millones de dólares. Las empresas que habrían realizado los aportes son: Constructora Norberto Odebrecht, SK Engineering & Construction, Sinohydro Corporation, Grupo Azul, Telconet, China International Water & Electric Corp-CWE."""

jar_postagger = "resources\\pos_tagger\\stanford-postagger.jar"
tagger_model = "resources\\pos_tagger\\spanish.tagger"

pos_tagger = StanfordPOSTagger(tagger_model, jar_postagger)

jar_nertagger = "resources\\ner_tagger\\stanford-ner.jar"
ner_model = "resources\\ner_tagger\\spanish.ancora.distsim.s512.crf.ser.gz"

ner_tagger = StanfordNERTagger(ner_model, jar_nertagger)

# Tokenizar sentencias
tokenized_sentences = nltk.sent_tokenize(text)

print(tokenized_sentences)

# Tokenizar sentencias
tokenized_words = [nltk.word_tokenize(sentence) for sentence in tokenized_sentences]

print(tokenized_words)

# POS Tagger
tokens_pos = [pos_tagger.tag(word) for word in tokenized_words]
print(tokens_pos)

# Stemmas
stemmas = [stemmer.stem(word) for word in tokenized_words[0]]
print(stemmas)

# NER Tagger
entities = [ner_tagger.tag(word) for word in tokenized_words]
print(entities)

# Stop words
print(stop_words)

# Removing using stop words
# for word in tokenized_words:
#     if word not in tokenized_words:
#         print(word)

