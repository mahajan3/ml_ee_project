import sys
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_td=error_detail.exc_info()
    filename=exc_td.tb_frame.f_code.co_filename
    error_message='The error in file [{0}] in line [{1}] is [{2}]'.format(
        filename,exc_td.tb_lineno,error
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)
    def __str__(self) -> str:
        return self.error_message