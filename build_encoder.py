import os
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32,exception_verbosity=high"
import skipthoughts

import sys
sys.path.append('training')


import cPickle as pkl
import numpy
import nltk

from collections import OrderedDict, defaultdict
from nltk.tokenize import word_tokenize
from scipy.linalg import norm
from gensim.models import Word2Vec as word2vec
from sklearn.linear_model import LinearRegression
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as tensor
import numpy

from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
import train
import tools

class Encoder(object):
    def __init__(self):
        pass
    def encode(self, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):
        """
        Encode sentences in the list X. Each entry will return a vector
        """
        model = self.model
        # first, do preprocessing
        X = preprocess(X)

        # word dictionary and init
        d = defaultdict(lambda : 0)
        for w in model['table'].keys():
            d[w] = 1
        features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')

        # length dictionary
        ds = defaultdict(list)
        captions = [s.split() for s in X]
        for i,s in enumerate(captions):
            ds[len(s)].append(i)

        # Get features. This encodes by length, in order to avoid wasting computation
        for k in ds.keys():
            numbatches = len(ds[k]) / batch_size + 1
            for minibatch in range(numbatches):
                caps = ds[k][minibatch::numbatches]

                if use_eos:
                    embedding = numpy.zeros((k+1, len(caps), model['options']['dim_word']), dtype='float32')
                else:
                    embedding = numpy.zeros((k, len(caps), model['options']['dim_word']), dtype='float32')
                for ind, c in enumerate(caps):
                    caption = captions[c]
                    for j in range(len(caption)):
                        if d[caption[j]] > 0:
                            embedding[j,ind] = model['table'][caption[j]]
                        else:
                            embedding[j,ind] = model['table']['UNK']
                    if use_eos:
                        embedding[-1,ind] = model['table']['<eos>']
                if use_eos:
                    ff = model['f_w2v'](embedding, numpy.ones((len(caption)+1,len(caps)), dtype='float32'))
                else:
                    ff = model['f_w2v'](embedding, numpy.ones((len(caption),len(caps)), dtype='float32'))
                if use_norm:
                    for j in range(len(ff)):
                        ff[j] /= norm(ff[j])
                for ind, c in enumerate(caps):
                    features[c] = ff[ind]
        return features
        
    @property
    def model(self):
        try:
            return self._model
        except:
            return None

    def generate_model(self, path_to_word2vec='w2v_vocab_expansion.bin',
                       path_to_dictionary='../debate_speech.pkl',
                       path_to_model='model.npz', 
                       pickle_table='table.npy'):
        embed_map = word2vec.load_word2vec_format(path_to_word2vec, binary=True)    
        # Load the worddict
        print 'Loading dictionary...'
        with open(path_to_dictionary, 'rb') as f:
            worddict = pkl.load(f)

        # Create inverted dictionary
        print 'Creating inverted dictionary...'
        word_idict = dict()
        for kk, vv in worddict.iteritems():
            word_idict[vv] = kk
        word_idict[0] = '<eos>'
        word_idict[1] = 'UNK'

        # Load model options
        print 'Loading model options...'
        with open('%s.pkl'%path_to_model, 'rb') as f:
            options = pkl.load(f)

        # Load parameters
        print 'Loading model parameters...'
        params = init_params(options)
        params = load_params(path_to_model, params)
        tparams = init_tparams(params)

        # Extractor functions
        print 'Compiling encoder...'
        trng = RandomStreams(1234)
        trng, x, x_mask, ctx, emb = build_encoder(tparams, options)
        f_enc = theano.function([x, x_mask], ctx, name='f_enc')
        f_emb = theano.function([x], emb, name='f_emb')
        trng, embedding, x_mask, ctxw2v = build_encoder_w2v(tparams, options)
        f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')

        # Load word2vec, if applicable
        if embed_map == None:
            print 'Loading word2vec embeddings...'
            embed_map = load_googlenews_vectors(path_to_word2vec)

        # Lookup table using vocab expansion trick
        if pickle_table:
            t = numpy.load(pickle_table)
            table = OrderedDict()
            for k,v in t:
                table[k]=v
        else:
            print 'Creating word lookup tables...'
            table = lookup_table(options, embed_map, worddict, word_idict, f_emb)

        # Store everything we need in a dictionary
        print 'Packing up...'
        model = {}
        model['options'] = options
        model['table'] = table
        model['f_w2v'] = f_w2v
        
        self._model = model
        return model

def init_tparams(params):
    """
    Initialize Theano shared variables according to the initial parameters
    """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def load_params(path, params):
    """
    Load parameters
    """
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive'%kk)
            continue
        params[kk] = pp[kk]
    return params

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()
    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    # Encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',nin=options['dim_word'], dim=options['dim'])
    # Decoder: next sentence
    params = get_layer(options['decoder'])[0](options, params, prefix='decoder_f',nin=options['dim_word'], dim=options['dim'])
    # Decoder: previous sentence
    params = get_layer(options['decoder'])[0](options, params, prefix='decoder_b',nin=options['dim_word'], dim=options['dim'])
    # Output layer
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim'], nout=options['n_words'])
    return params

def build_encoder(tparams, options):
    """
    Computation graph, encoder only
    """
    opt_ret = dict()
    trng = RandomStreams(1234)
    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    # word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=x_mask)
    ctx = proj[0][-1]
    return trng, x, x_mask, ctx, emb

def build_encoder_w2v(tparams, options):
    """
    Computation graph for encoder, given pre-trained word embeddings
    """
    opt_ret = dict()
    trng = RandomStreams(1234)
    # word embedding (source)
    embedding = tensor.tensor3('embedding', dtype='float32')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    # encoder
    proj = get_layer(options['encoder'])[1](tparams, embedding, None, options,
                                            prefix='encoder',
                                            mask=x_mask)
    ctx = proj[0][-1]
    return trng, embedding, x_mask, ctx

def ortho_weight(ndim):
    """
    Orthogonal weight init, for recurrent layers
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.1, ortho=True):
    """
    Uniform initalization from [-scale, scale]
    If matrix is square and ortho=True, use ortho instead
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')

def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X

def lookup_table(options, embed_map, worddict, word_idict, f_emb, use_norm=False):
    """
    Create a lookup table from linear mapping of word2vec into RNN word space
    """
    wordvecs = get_embeddings(options, word_idict, f_emb)
    clf = train_regressor(options, embed_map, wordvecs, worddict)
    table = apply_regressor(clf, embed_map, use_norm=use_norm)
    for i in range(options['n_words']):
        w = word_idict[i]
        table[w] = wordvecs[w]
        if use_norm:
            table[w] /= norm(table[w])
    return table

def get_embeddings(options, word_idict, f_emb, use_norm=False):
    """
    Extract the RNN embeddings from the model
    """
    d = OrderedDict()
    for i in range(options['n_words']):
        caption = [i]
        ff = f_emb(numpy.array(caption).reshape(1,1)).flatten()
        if use_norm:
            ff /= norm(ff)
        d[word_idict[i]] = ff
    return d

def train_regressor(options, embed_map, wordvecs, worddict):
    """
    Return regressor to map word2vec to RNN word space
    """
    # Gather all words from word2vec that appear in wordvecs
    d = defaultdict(lambda : 0)
    for w in embed_map.vocab.keys():
        d[w] = 1
    shared = OrderedDict()
    count = 0
    for w in worddict.keys()[:options['n_words']-2]:
        if d[w] > 0:
            shared[w] = count
            count += 1

    # Get the vectors for all words in 'shared'
    w2v = numpy.zeros((len(shared), 300), dtype='float32')
    sg = numpy.zeros((len(shared), options['dim_word']), dtype='float32')
    for w in shared.keys():
        w2v[shared[w]] = embed_map[w]
        sg[shared[w]] = wordvecs[w]

    clf = LinearRegression()
    clf.fit(w2v, sg)
    return clf

def apply_regressor(clf, embed_map, use_norm=False):
    """
    Map words from word2vec into RNN word space
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wordvecs = OrderedDict()
        for i, w in enumerate(embed_map.vocab.keys()):
            if '_' not in w:
                wordvecs[w] = clf.predict(embed_map[w]).astype('float32')
                if use_norm:
                    wordvecs[w] /= norm(wordvecs[w])
        return wordvecs