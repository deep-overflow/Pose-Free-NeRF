
class Config:
    def __init__(self, configs_dict):
        for key, value in configs_dict.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value