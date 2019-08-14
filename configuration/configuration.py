import configparser
import datetime
import sys
import logging
__config =None
__config_file = None

def getConfig():
    global __config, __config_file
    if __config and __config_file == sys.argv[1]:
        return __config
    else:
        try:
            __config_file = sys.argv[1]
            logging.info("config file %s"%__config_file)
            __config = configparser.ConfigParser()
            __config.read(__config_file)
            __config.eval = lambda sec,key: eval(__config.get(sec,key))
            __config.getfilename = lambda: __config_file
            return __config
        except:
            logging.exception( "Usage: %s <path to config .ini file>" , (sys.argv[0]))
            exit(-1)

def date(datestring, formate="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.strptime(datestring, formate)
