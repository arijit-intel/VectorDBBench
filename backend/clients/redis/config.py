from pydantic import SecretStr
from ..api import DBConfig

class RedisConfig(DBConfig):
    #password: SecretStr
    host: SecretStr
    port: int = None 
    db_label: str

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "db_label": self.db_label,
            #"password": self.password.get_secret_value(),
        }