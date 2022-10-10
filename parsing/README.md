Hooking up words and phrases and clauses
================

Introduction
----------------------

As always, check out the Github repository with the course homework templates:

git://github.com/ezubaric/cl1-hw.git

The code for this homework is in the parsing directory.

Treebank Probabilities (4 points)
--------------------------

To warm up, you'll first compute PCFG probabilities from
data.  You'll create a class that can read in sentences (through the
`add_sentence` function and then answer queries using the
`query` function.

Add the required functionality to the file `treebank.py`.  You
can see how the code will be called:


```
>>> tb_probs = PcfgEstimator()
>>> from nltk.corpus import treebank
>>> for ii in treebank.parsed_sents():
...         tb_probs.add_sentence(ii)
...
>>> tb_probs.query('NN', 'man')
0.0009114385538508279
```

Shift-Reduce Parsers (40 points)
--------------------------------------

Next, you'll create sequences of shift-reduce actions that can produce
dependency parses from a string of words.  You'll implement the
`transition_sequence` function in the `oracle.py` file.
Your code will be much simpler if you use
[generators](https://wiki.python.org/moin/Generators).

An example of the shift-reduce sequence is:

```
>>> from nltk.corpus import conll2007
>>> print(" ".join(conll2007.sents('esp.train')[1]))
El Banco_Central_Europeo - BCE - fijo el cambio oficial
                       del euro en los 0,9355_dolares .
>>> s = conll2007.parsed_sents('esp.train')[1]
>>> for x in transition_sequence(s):
...     print(x.pretty_print(s))
...
s
s
l	(Banco_Central_Europeo, El)
s
s
l	(BCE, -)
s
r	(BCE, -)
r	(Banco_Central_Europeo, BCE)
s
l	(fijo, Banco_Central_Europeo)
s
s
l	(cambio, el)
s
r	(cambio, oficial)
s
s
r	(del, euro)
r	(cambio, del)
r	(fijo, cambio)
s
s
s
l	(0,9355_dólares, los)
r	(en, 0,9355_dolares)
r	(fijo, en)
s
r	(fijo, .)
r	(None, fijo)
s
```

Writeup?  Extra Credit?
-------------------------

Nope.  This is an easy homework.  Focus on projects!
