

class Abstract():


    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)


    def solve(self, data):
        raise NotImplementedError