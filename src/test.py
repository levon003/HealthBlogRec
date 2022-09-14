"""
Single entry point for doctest and unittest.

"""


import cbrec.coverage

import doctest

def main():
    doctest.testmod(cbrec.coverage)
    
if __name__ == '__main__':
    main()