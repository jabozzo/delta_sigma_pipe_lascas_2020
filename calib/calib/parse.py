#! /usr/bin/env python

import parsec as psc


def lexme(parser):
    return parser << psc.spaces()


def int_number(self):
    '''Parse int number.'''
    return psc.regex(r'-?(0|[1-9][0-9]*)').parsecmap(int)


def float_number(self):
    '''Parse float number.'''
    return psc.regex('-?(0|[1-9][0-9]*)([.][0-9]+)?([eE][+-]?[0-9]+)?').parsecmap(float)


def word(self):
    return psc.many1(psc.letter()).parsecmap(lambda x: ''.join(x))


def words(self, n=1):
    return psc.separated(self.word(), psc.spaces(), n, n, end=False).parsecmap(lambda x: ' '.join(x))
