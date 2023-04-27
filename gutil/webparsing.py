from datetime import datetime
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException

def wait(driver,tag, attrib, wait_time = 10):
    '''
    waits until element is loaded
    if more than wait_time passed throws an error

    '''
    cur_time = datetime.now()
    while (datetime.now()-cur_time).total_seconds() < wait_time:
        bs = BeautifulSoup(driver.page_source)
        el = bs.find(tag, attrib)
        if el is not None:
            return el
    raise ValueError("element did not appear in time")

def find_element(driver, by= "id", value = None):
    '''
    selenium throws by default, here I will return nan if not found
    '''
    try:
        el = driver.find_element(by, value)
        return el
    except NoSuchElementException:
        return None

