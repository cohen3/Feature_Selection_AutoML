from gensim.models.doc2vec import Doc2Vec as d2v, TaggedDocument


class Doc2Vec:

    def __init__(self,**kwargs):
        self.list_of_RW = []
        self.listOfTaggedDocuments = []
        self.vector_size=int(kwargs['vector_size'])
        self.window_size=int(kwargs['window_size'])
        self.learning_rate=float(kwargs['learning_rate'])
        self.min_learning_rate=float(kwargs['min_learning_rate'])
        self.max_vocab_size=kwargs['min_vocab_size']
        self.min_count=int(kwargs['min_count'])
        self.workers=int(kwargs['workers'])
        self.epochs=int(kwargs['epochs'])


    def fit(self):
        model = d2v(self.listOfTaggedDocuments ,
                    vector_size=self.vector_size,
                    window=self.window_size,
                    min_count=self.min_count,
                    workers=self.workers,
                    epochs=self.epochs)
        return model.docvecs

    def transform(self, subGraphsList):
        for rwList in subGraphsList:
            tmp = []
            for graph in rwList:
                tmp.append(list(graph.nodes))
            self.list_of_RW.append(tmp)
        self.createTaggedDocuments()

    def createTaggedDocuments(self):
        for i, doc in enumerate(self.list_of_RW):
            for j in doc:
                self.listOfTaggedDocuments.append(TaggedDocument(j, [i]))


