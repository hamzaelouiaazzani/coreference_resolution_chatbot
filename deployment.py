import json
import stanfordnlp
import stanza



lang = "fr"
nlp1 = stanfordnlp.Pipeline(lang=lang, processors="tokenize,mwt,pos")
nlp2 = stanza.Pipeline(lang=lang, processors='tokenize,ner')







#Function that assingn every indexed mention to its textual form
def indexed_mention_2_textual_mention(json_object , indexed_mention):
    
    sentences = json_object["sentences"]
    start , end = indexed_mention[0] , indexed_mention[1]
    tokens = [t for sent in sentences for t in sent]
    
    string = tokens[start]
    for i in range(start+1 , end+1):
        string = string + " " + tokens[i]
    
    return string


#Entity recognition
def NER(text):
    doc = nlp2(text)
    for ent in doc.ents:
        entities = [ent.text for ent in doc.ents]
    return entities




#Is_entity : the function that decides whether a mention is an entity or not
def is_entity_indexed(json_object , entities , indexed_mention): 
    text_mention = indexed_mention_2_textual_mention(json_object , indexed_mention)
    return is_entity_text(entities , text_mention)
   
    
def is_entity_text(entities , text_mention):
    if text_mention in entities:
        return True
    else:
        return False


    
#the longest noun phrase mention      
def longest_NP_mention_indexed(cluster):
    l = []
    for i in cluster:
        start , end = i[0] , i[1]
        l.append(end-start+1)    
    return cluster[l.index(max(l))]

def longest_NP_mention_text(cluster , json_object):
    indexed_mention = longest_NP_mention_indexed(cluster)
    return indexed_mention_2_textual_mention(json_object , indexed_mention)


#List of sentences to text
def list_sents_2_text(list_sents):
    text = list_sents[0]
    for i in range(len(list_sents)-1) :
        text = text + "\n" + list_sents[i+1]
    return text



#Function to assign a representative entity for a given cluster
def rep_entity_of_cluster(cluster , entities , json_object):
    for i in cluster:
        if (is_entity_indexed(json_object , entities , i)): 
            return indexed_mention_2_textual_mention(json_object , i)
    
    return longest_NP_mention_text(cluster , json_object) 



#Function that search which cluster a given mention belongs to
def search(indexed_mention , clusters):
    for i in clusters:
        if indexed_mention in i:
            return i
    return None


#Function that affect to a mention the entity it belongs to
def affecting_entity_mention(indexed_mention , entities , json_object):
    clusters = json_object["clusters"]
    if search(indexed_mention , clusters)==None:
        #print("it is not in a cluster")
        return indexed_mention_2_textual_mention(json_object , indexed_mention)
    else:
        cluster = search(indexed_mention , clusters)
        #print("it is in a cluster")
        return rep_entity_of_cluster(cluster , entities , json_object)


#Function that transform a list of succesive tokens to a string sentence
def list_tokens_2_string(list_tokens):
    string = list_tokens[0]
    for i in range(len(list_tokens)-1):
        string = string + " " + list_tokens[i+1]  
    return string


#Function that take a sentence and affect every every pronoun to its representative entity
def affecting_pronouns_by_entities(entities , json_object , index=-1):
    start , end = json_object["paragraphs"][index]
    pos = [pos for list_pos in json_object["pos"] for pos in list_pos][start:end+1]
    sent = [sent for list_sent in json_object["sentences"] for sent in list_sent][start:end+1]
    list_to_detect = ["DET" , "PRON"]
    for i in range(len(pos)):
        if pos[i] in list_to_detect:
            #print(indexed_mention_2_textual_mention(json_object , [i+start , i+start]))
            string = affecting_entity_mention([i+start , i+start] , entities , json_object)
            sent[i] = string
    return list_tokens_2_string(sent)


#the final function
def return_coreferenced_sentence(json_object , list_sents):
    text = list_sents_2_text(list_sents)
    entities = NER(text)
    string = affecting_pronouns_by_entities(entities , json_object , index=-1)
    return string

#function that take the output of tokenize_text function and make the json object that will be exploited by Bert and coreference model
def make_json(sents, pos, pars, fpath, genre):
    doc = dict(
        doc_key = f"{genre[:2]}:{fpath}",
        sentences = sents,
        speakers = [ [ "_" for tok in sent ] for sent in sents ],
        clusters = [],
        pos = pos,
        paragraphs = pars,
    )
    return doc
    

    
 
#tokenize the text
def tokenize_text(list_text, lang = "fr"):
    res_sents = []
    res_pars = []
    res_pos = []
    start_par = 0
    for par in list_text:
        par = par.strip()
        if not par:
            continue
        doc = stanfordnlp.Document(par)
        
        doc = nlp1(doc)
        #print(doc.conll_file.conll_as_string())
        #print(doc.conll_file.sents)
        sents = [
            [ token[1] for token in sent if '-' not in token[0] ]
            for sent in doc.conll_file.sents
        ]
        pos = [
            [ token[3] for token in sent if '-' not in token[0] ]
            for sent in doc.conll_file.sents
        ]
        res_sents.extend(sents)
        res_pos.extend(pos)
        length = sum((len(s) for s in sents))
        res_pars.append([start_par, start_par+length-1])
        start_par = start_par+length
    return res_sents, res_pos, res_pars