import argparse
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run the message generator')
    parser.add_argument('n', type=int, help='The number of sentences to generate')
    parser.add_argument('seed', help='Sentence to seed the decoder')
    parser.add_argument('-s', '--stochastic', action='store_true', help='Use stochastic selection for decoder')
    args = parser.parse_args()
    import os
    os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32,exception_verbosity=high"
    import skipthoughts
    import sys
    sys.path.append('decoding')
    import decode_tools
    decoder_model = decode_tools.load_model()
    import pickle
    import build_encoder
    
    clean_tweet_sentences = pickle.load(open('clean_tweet_sentences.pkl', 'r'))
    facebook_sentences = pickle.load(open('facebook_sentences.pkl', 'r'))
    encoder = build_encoder.Encoder()
    encoder_model = encoder.generate_model()

    def generate(n, seed=None):
        import random
        if not seed:
            encoded_tweets = encoder.encode(clean_tweet_sentences)
            seeds = decode_tools.run_sampler(decoder_model, encoded_tweets.mean(0))
            seed = random.choice(seeds)
        print 'Working with seed: %s '% seed, type(seed)
        sentences = [seed]
        for i in range(n):
            print 'Working with: %s '% sentences[i]
            choices = decode_tools.run_sampler(decoder_model, encoder.encode([sentences[i]]), beam_width=10)

            choice = random.choice(choices)
            sentences.append(choice)
        return ' '.join(sentences)
    def stocastic_generate(n, seed=None):
        import random
        if not seed:
            encoded_tweets = encoder.encode(clean_tweet_sentences)
            seeds = decode_tools.run_sampler(decoder_model, encoded_tweets.mean(0))
            seed = random.choice(seeds)
        print 'Working with seed: %s '% seed, type(seed)
        sentences = [seed]
        for i in range(n):
            print 'Working with: %s '% sentences[i]
            choices = decode_tools.run_sampler(decoder_model, encoder.encode([sentences[i]]), stochastic=True)

            choice = random.choice(choices)
            sentences.append(choice)
        return ' '.join(sentences)
    
    if args.stochastic:
        print(stocastic_generate(args.n, args.seed))
    else:
        print(generate(args.n, args.seed))


