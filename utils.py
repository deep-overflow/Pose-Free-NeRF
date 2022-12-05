class Config:
    def __init__(self, args_dict):
        for key, value in args_dict.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value