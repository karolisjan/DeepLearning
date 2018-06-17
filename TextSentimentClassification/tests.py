# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:27:37 2017

@author: Karolis
"""

from unittest import TestCase, main
from preprocessing import (clean, 
                           tokenize, 
                           tokenize_n_stem, 
                           tokenize_n_lemmatize)


class Test(TestCase):
    def setUp(self):
        pass
    
    
    def test_clean_text_fun(self):
        self.assertEqual(clean("</a>This :) is :( a test :-)!"), 'this is a test :) :( :)')
    
    
    def test_tokenize_fun(self):
        self.assertEqual(tokenize('runners like running and thus they run'),
                         ['runners', 'like', 'running', 'thus', 'run'])
        
        
    def test_tokenize_n_stem_fun(self):
        self.assertEqual(tokenize_n_stem('runners like running and thus they run'),
                         ['runner', 'like', 'run', 'thus', 'run'])
        
        
    def test_tokenize_n_lemmatize_fun(self):
        self.assertEqual(tokenize_n_lemmatize('runners like running and thus they run'),
                         ['runner', 'like', 'running', 'thus', 'run'])    
    
    
if __name__ == "__main__":
    main()