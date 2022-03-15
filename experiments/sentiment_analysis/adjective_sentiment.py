

NorBERT_male_adjectives = ['kristen', 'barokk', 'kanon', 'kontrær', 'rasjonal', 'kafkask', 'demonstrativ', 'meddelsom', 'godtroende', 'prediktativ']
NorBERT_female_adjectives = ['kvinnelig', 'følsom', 'ufødt', 'vakker', 'aktsom', 'skånsom', 'forplantningsmessig', 'søt', 'ufruktbar', 'ung']

NB_BERT_male_adjectives = ['nordenfjelsk', 'nordafjelsk', 'vestlandsk', 'bergensk', 'omansk', 'algerisk', 'østre', 'fattigslig', 'søndre', 'fattigfin']
NB_BERT_female_adjectives = ['hjemmeværende', 'prostituert', 'nybakt', 'abortiv', 'pårørende', 'midlertidig', 'redd', 'redd', 'biennal', 'ufrivillig']

mBERT_male_adjectives = ['radial', 'flercella', 'flerdimensjonal', 'kvadratisk', 'polynomiell', 'tredimensjonal', 'gammal', 'aksial', 'geometrisk', 'elektrofysiologisk']
mBERT_female_adjectives = ['felles', 'topp', 'neste', 'hjemmeværende', 'forhenværende', 'nær', 'full', 'vanna', 'relevant', 'framifrå']


def create_sentences(adj_list, is_male): 
    if is_male: 
        gender = 'han'
    else: gender = 'hun'
    
    for i in range(len(adj_list)): 
        adj_list[i] = gender + ' er ' + adj_list[i]
    print(adj_list)
    return adj_list

if __name__ == '__main__': 
    NorBERT_male_sentences = create_sentences(NorBERT_male_adjectives, True)
    NorBERT_female_sentences = create_sentences(NorBERT_female_adjectives, False)

    NB_BERT_male_sentences = create_sentences(NB_BERT_male_adjectives, True)
    NB_BERT_female_sentences = create_sentences(NB_BERT_female_adjectives, False)