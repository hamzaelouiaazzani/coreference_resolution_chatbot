import stanfordnlp
import json
def getIndex(phrase, index):
    limits = []
    file = open('output.jsonlines', 'r')
    file_content = file.read()
    indices = []

    json_content = json.loads(file_content)
    paragraphs = json_content['paragraphs']

    phrase_indices = paragraphs[index]

    sentence = json_content['sentences'][index]
    
    doc = stanfordnlp.Document(phrase)
    nlp = stanfordnlp.Pipeline(lang="fr", processors="tokenize")
    doc = nlp(doc)
    sents = [
    [ token[1] for token in sent if '-' not in token[0] ]
    for sent in doc.conll_file.sents]

    
    words = sents[0]

    for word in words:
        for index, elem in enumerate(sentence):
            if word == elem:
                indices.append(phrase_indices[0] + index)
    limits.append(indices[0])
    limits.append(indices[-1])
    return limits