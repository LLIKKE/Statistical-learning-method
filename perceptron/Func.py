import numpy as np

def seed_up(seed=0):
    np.random.seed(seed)


class loss_function:
    def __init__(self, mode='01loss'):
        self.mode = mode

    def __call__(self, x, y):
        if self.mode == '01loss':
            if x == y:
                return 0.
            else:
                return 1.0
        elif self.mode == 'l2loss':
            return (x - y) ** 2
        elif self.mode == 'l1loss':
            return abs(x - y)
        elif self.mode == 'lloss':
            return False


class model_error(loss_function):
    def __init__(self):
        super().__init__()

    def __call__(self, x: list, y: list):
        N = len(x)
        x = np.array(x)
        y = np.array(y)
        s = np.sum(x == y)
        return s / N

#class generalization_error(loss_function):


if __name__ == '__main__':
    mystr = "python"
    myit = iter(mystr)

    print(next(myit))
    print(next(myit))
    print(next(myit))
    print(next(myit))
    print(next(myit))
    print(next(myit))
