yaffel.py
=========

Parser and interpreter for the functional programming language yaffel.

Installation
------------

You can install `yaffel-py` from sources using `distutils`. First download the sources:

	# git clone https://github.com/kyouko-taiga/yaffel-py.git

Then navigate to the repository directory and run `setup.py`:

	# cd yaffel-py
	# python setup.py install
	
Usage
-----

The quickest way to test `yaffel-py` is to use the command line tool. Type `yaffel` in your terminal to start a command-line interpreter. Then simply type yaffel-expressions and get their result as they are evaluated. Alternatively, you can type `yappel -e "some expression"` to evaluate an expression without loading the shell.

To use `yaffel-py` in your own code, simply import the parser among with your other dependencies and call `parse` to parse a yaffel expression:

```python
import yaffel.parser
...
result = yaffel.parser.parse('5 * (y + x) for x=4, y=7')
```

The `result` will be a tuple whose first element is the type of the parsed expression and second element is its value.
