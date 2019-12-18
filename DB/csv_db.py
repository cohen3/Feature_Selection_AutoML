from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
import pandas as pd
from os import listdir
from pandasql import sqldf


# import csvquerytool


class CSV_DB(AbstractController):

    def __init__(self):
        self.data_path = getConfig().eval(self.__class__.__name__, "csv_path")
        self.pysqldf = lambda q: sqldf(q, self.data_path)
        self.is_csv = getConfig().eval(self.__class__.__name__, "is_csv")

    def setUp(self):
        # configInst = getConfig()
        self._date = getConfig().eval(self.__class__.__name__, "start_date")
        # self._pathToEngine = configInst.get(self.__class__.__name__, "DB_path") + \
        #                      configInst.get(self.__class__.__name__, "DB_name_prefix") + \
        #                      configInst.get(self.__class__.__name__, "DB_name_suffix")

        # if configInst.eval(self.__class__.__name__, "remove_on_setup"):
        #     self.deleteDB()
        #
        # self.engine = create_engine("sqlite:///" + self._pathToEngine, echo=False)
        # self.Session = sessionmaker()
        # self.Session.configure(bind=self.engine)
        #
        # self.session = self.Session()

        # @event.listens_for(self.engine, "connect")
        # def connect(dbapi_connection, connection_rec):
        #    dbapi_connection.enable_load_extension(True)
        #    dbapi_connection.execute(
        #        'SELECT load_extension("{0}{1}")'.format(configInst.get("DB", "DB_path_to_extension"), '.dll'))
        #
        #     dbapi_connection.enable_load_extension(False)
        #
        # if getConfig().eval(self.__class__.__name__, "dropall_on_setup"):
        #     Base.metadata.drop_all(self.engine)
        #
        # Base.metadata.create_all(self.engine)
        pass

    def commit(self):
        # self.session.commit()
        pass

    def df_to_table(self, df, name='mytable', mode='append'):
        """

        This method writes a dataframe to the database, the mode and name of the table can be modified.

        :param df: dataframe to write
        :param name: the name of the table
        :param mode: {'fail', 'replace', 'append'}
        :return: None
        """

        df.to_csv(self.data_path + name + ".csv", index=False, na_rep=0)

    def execQuery(self, q):
        # print(q)
        # if "SELECT name FROM sqlite_master WHERE type='table" in q:
        #     lst = listdir(self.data_path)
        #     print(lst)
        #     return lst
        #
        # #query = text(q)
        # results = self.pysqldf(q)
        # #result = self.session.execute(query)
        # #cursor = result.cursor
        # print("11111111"*10)
        # print(results)
        # #records = list(cursor.fetchall())
        return 0

    def create_table(self, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        # try:
        #     self.session.execute(create_table_sql)
        # except:
        #     pass
        return
