from pydantic import SecretStr
from ..api import DBConfig

class vdmsConfig(DBConfig):
    #password: SecretStr
    host: SecretStr
    port: int
    db_label: str
    distance_strategy: str
    engine: str

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "db_label": self.db_label,
            "distance_strategy": self.distance_strategy,
            "engine": self.engine
            #"password": self.password.get_secret_value(),
        }