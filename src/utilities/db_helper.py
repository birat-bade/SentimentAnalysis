import MySQLdb

from config.config import Config


class DbHelper:
    def __init__(self):
        self.connection_mysql = MySQLdb.connect(Config.server, Config.user, Config.password, Config.db,
                                                use_unicode=True, charset="utf8")
