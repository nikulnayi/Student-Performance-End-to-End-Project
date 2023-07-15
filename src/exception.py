import sys
import logging
from src.logger import logging
def errorMessegeDetail(error,errorDetail:sys):
    _,_,exe_tb  = errorDetail.exc_info()
    fileName = exe_tb.tb_frame.f_code.co_filename
    errorMessege = "Error Occured in python script name [{0}] line number [{1}] error messege[{2}]".format(
        fileName,exe_tb.tb_lineno,str(error))
    return errorMessege
class customException(Exception):
    def __init__(self, errorMessege,errorDetail:sys):
        super().__init__(errorMessege)
        self.errorMessege = errorMessegeDetail(errorMessege,errorDetail=errorDetail)
    def __str__(self):
        return self.errorMessege
