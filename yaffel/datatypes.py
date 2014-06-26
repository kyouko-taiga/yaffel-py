# This source file is part of yaffel-py
# Main Developer : Dimitri Racordon (kyouko.taiga@gmail.com)
#
# Copyright 2014 Dimitri Racordon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from funcparserlib.lexer import Token
from itertools import zip_longest
from yaffel.exceptions import UnboundVariableError, InvalidExpressionError

import numbers, importlib

__all__ = ['Name', 'Expression', 'AnonymousFunction', 'Application', 'Set', 'Enumeration', 'Range']

def value_of(variable, context):
    if isinstance(variable, Expression):
        # `variable` is an instance of Expression, we simply evaluate it
        return variable(**context)
    elif isinstance(variable, Name):
        try:
            # we try to bound `variable` from the `context`
            return context[variable]
        except KeyError:
            raise UnboundVariableError("unbound variable '%s'" % variable) from None

    # `variable` is not symbolic
    return variable

class Name(str):
    """Represents a symbolic name in expressions or contexts."""
    def __new__(cls, c_str):
        return str.__new__(cls, c_str)

class Expression(object):
    """Represents an expression as an anonymous function.

    When parsed, yaffel expressions are recursively (actually the process is
    performed bottom-up) decomposed into smaller lazy-evaluated expressions, or
    anonymous functions, until no further decomposition is possible, i.e. when
    when they are reduced to atomic primitives such as numbers, strings, etc.
    """

    def __init__(self, unfolded_expr):
        # The unfolded expression E' of an expression E is a sequence
        # [t1, (f1, t2), (f2, t3), ...] starting with a term followed
        # unfolded_expr arbitrary number of tuples (operator, term),
        # such that E = f1(t1, f2(t2, ...)).
        self._unfolded_expr = unfolded_expr

    def __call__(self, **context):
        """Evaluates the expression value.

        This method evaluates the expression, using ``context`` to bind its
        free variables, if such are present.
        """
        try:
            # retrieve the first term value
            a = value_of(self._unfolded_expr[0], context)
        except TypeError:
            # `_unfolded_expr` is either [] or not iterable
            raise InvalidExpressionError("'%s' is not a valid expression" %
                                         self._unfolded_expr) from None

        # evaluate expression
        for f,b in self._unfolded_expr[1:]:
            a = f(a, value_of(b, context))
        return a

    def _unfolded_expr_str(self):
        if not self._unfolded_expr: return ''
    
        # helper function to get the string representation of a term
        sym = lambda term: term.value if isinstance(term, Token) else str(term)

        # if the expression is simple a constant
        a = sym(self._unfolded_expr[0])
        if len(self._unfolded_expr) <= 1:
            return a

        ret = ''
        for f,b in self._unfolded_expr[1:]:
            a = '%(f)s(%(a)s, %(b)s)' % {'f': f, 'a': a, 'b': sym(b)}
        return a        

    def __hash__(self):
        if self._unfolded_expr is None:
            return 0
        return hash(tuple(self._unfolded_expr))

    def __eq__(self, other):
        return all(a == b for a,b in zip_longest(self.unfolded_expr, other.unfolded_expr))

    def __str__(self):
        return self._unfolded_expr_str()

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

class AnonymousFunction(Expression):

    def __init__(self, args, expr):
        self._args = args
        super().__init__(expr._unfolded_expr)

    def __call__(self, *argv) :
        if len(argv) != len(self._args):
            raise TypeError("%s takes %i arguments but %i were given" %
                            (self, len(self._args), len(argv)))
        context = {self._args[i]: argv[i] for i in range(len(self._args))}
        return super().__call__(**context)

    def __hash__(self):
        return hash(tuple(self._args + [super().__hash__()]))

    def __eq__(self, other):
        f = lambda x,y: all(a == b for a,b in zip_longest(x,y))
        return f(self._args, other._args) and f(self._unfolded_expr, other._unfolded_expr)

    def __str__(self):
        return '%(args)s: %(expr)s' % {
            'args': 'f ' + ', '.join(self._args),
            'expr': self._unfolded_expr_str()
        }

class Application(object):

    def __init__(self, function, args):
        self._function = function
        self._args = args

    def __call__(self, **context):
        if isinstance(self._function, AnonymousFunction):
            # `_function` is an AnonymousFunction so we simply call it
            return self._function(*(value_of(a, context) for a in self._args))

        try:
            # `_function` is a symbol, we first try to bound it from the context
            fx = value_of(self._function, context)
        except UnboundVariableError:
            # if `function` can't be bound from the context, try to use a built-in
            fx = None
            for mod in ('builtins', 'math',):
                fx = getattr(importlib.import_module(mod), self._function, None)
                if fx: break            

        # raise an evaluation error if `_function` couldn't be bound
        if not fx:
            raise UnboundVariableError("unbound function name '%s'" % self._function)
        elif not hasattr(fx, '__call__'):
            raise TypeError("invalid type '%s' for a function application" %
                            type(self.function).__name__)

        # apply fx
        # TODO instanciate yaffel sets as python iterable so we can call python
        # built-in functions than run on tierables, such as `sum`
        return fx(*(value_of(a, context) for a in self._args))

    def __hash__(self):
        return hash(tuple([hash(self.function)] + self.args))

    def __eq__(self, other):
        f = lambda x,y: all(a == b for a,b in zip_longest(x,y))
        return (self._function == other._function) and f(self._args, other._args)

    def __str__(self):
        return '%s(%s)' % (self._function, ', '.join(map(str, self._args)))

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

class Set(object):
    """Symbolic representation of a set.

    Sets are represented symbolically as a tuple (f,u) where f is a function
    and u another set. Let a set S be defined by (f,u), then elements of S are
    given by {f(x) | x \in u}.
    """

    def __init__(self, function, context):
        self.function = function
        self.context = context

    def __call__(self, **context):
        return Set(self.function, {k: v(**context) for k,v in self.context.items()})

    def __eq__(self, other):
        return (self.function, other.function) and (self.context == other.context)

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

    def __str__(self):
        f = lambda c: '%s in %s' % (c[0], str(c[1]))
        return '{%s for %s}' % (self.function, ', '.join(f(c) for c in self.context.items()))

class Enumeration(Set):
    """Kind of set that simply enumerates values."""

    def __init__(self, elements):
        self.elements = frozenset(elements)

    def __call__(self, **context):
        return Enumeration(e(**context) for e in self.elements)

    def __hash__(self):
        return hash(self.elements)

    def __eq__(self, other):
        return self.elements == other.elements

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

    def __str__(self):
        return '{%s}' % ', '.join(str(e) for e in self.elements)

class Range(Set):
    """Numeric set that contains values from its lower to its upper bound."""

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, **context):
        # evaluate lower and upper bounds
        lower = self.lower_bound(**context)
        upper = self.upper_bound(**context)

        # check type consistency
        if not isinstance(lower, numbers.Real) or not isinstance(upper, numbers.Real):
            raise TypeError('range defined for non-numeric lower or upper bounds')
        if not lower < upper:
            raise TypeError('range defined with unordered bounds')

        return Range(lower, upper)

    def __eq__(self, other):
        return (self.lower_bound == other.lower_bound) and (self.upper_bound == other.upper_bound)

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

    def __str__(self):
        return '{%s:%s}' % (repr(self.lower_bound), self.upper_bound)
