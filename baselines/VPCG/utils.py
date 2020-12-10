# encoding=utf8
useless_words = ['-','——','_','【','】','(',')','.',',','《','》','?','、','（','）','。',':','，','・']

def filter_title(doc_words):
    words = []
    for w in doc_words:
        if len(w) == 0 or w in useless_words:
            continue
        words.append(w)

    return words[:20]

def find_id(word_dict,word):
    return word_dict[word] if word in word_dict else 1


def model2id(model_name):
    #print 'model_name: ',model_name
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)