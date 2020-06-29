def setup(opt, vocab):
    try:
        mod = __import__('.'.join(['models', opt.model]), fromlist=['Model'])
        model = getattr(mod, 'CapModel')(opt, vocab)
    except:
        raise Exception("Model not supported: {}".format(opt.model))

    return model
