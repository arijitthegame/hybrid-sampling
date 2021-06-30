import numpy as np 

''' Code by Tamas Nepusz, Denis Bzowy
'''
class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probablities. Usage case is the dot product between positive random features.
    """
    
    def __init__(self, weights, keys=None):
        """Builds the Walker tables ``prob`` and ``inx`` for calls to `random()`.
        The weights (a list or tuple or iterable) can be in any order and they
        do not even have to sum to 1."""
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = np.array(keys)

        if isinstance(weights, (list, tuple)):
            weights = np.array(weights, dtype=float)
        elif isinstance(weights, np.ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = np.array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a vector")

        weights = weights * n / weights.sum()

        inx = -np.ones(n, dtype=int)
        short = np.where(weights < 1)[0].tolist()
        long = np.where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= (1 - weights[j])
            if weights[k] < 1:
                short.append( k )
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, count=None):
        """Returns a given number of random integers or keys, with probabilities
        being proportional to the weights supplied in the constructor.
        When `count` is ``None``, returns a single integer or key, otherwise
        returns a NumPy array with a length given in `count`.
        """
        if count is None:
            u = np.random.random()
            j = np.random.randint(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = np.random.random(count)
        j = np.random.randint(self.n, size=count)
        k = np.where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k

