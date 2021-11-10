import configparser

# 実行されるmain.pyから見た相対パス
CONFIG_FILE = './config.ini'


class ParamsBase:
    def __init__(self, env_name: str) -> None:
        config = configparser.ConfigParser()
        config.read_file(open(CONFIG_FILE))
        self.params = config[env_name]
