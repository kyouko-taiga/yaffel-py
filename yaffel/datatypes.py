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
from yaffel.exceptions import UnboundVariableError, InvalidExpressionError

import numbers, importlib

__all__ = ['Expression', 'AnonymousFunction', 'Application', 'Set', 'Enumeration', 'Range']


class Expression(object):
    """Represents an expression as an anonymous function.

    When parsed, yaffel expressions are recursively (actually the process is
    performed bottom-up) decomposed into smaller lazy-evaluated expressions, or
    anonymous functions, until no further decomposition is possible, i.e. when
    when they are reduced to atomic primitives such as numbers, strings, etc.
    """

    def __init__(self, head, expr=None):
        self.head = head
        self.expr = expr

    def __call__(self, **context):
        """Evaluates the expression value.

        This method evaluates the expression, using ``context`` to bind its
        free variables, if such are present.
        """
        # retrieve the first term value
        a = self._value(self.head, context)

        # evaluate expression
        for f,b in self.expr:
            a = f(a, self._value(b, context))
        return a

    def _value(self, term, context):
        if isinstance(term, Token):
            try:
                return context[term.value]
            except KeyError:
                raise EvaluationError("unbound variable '%s'" % term.value) from None
        elif hasattr(term, '__call__'):
            return term(**context)
        else:
            return term

    def __hash__(self):
        return hash((self.head, tuple(e for e in self.expr)))

    def __eq__(self, other):
        try:
            compare_seq = lambda x,y: all(x[i] == y[i] for i in range(max(len(x), len(y))))
            return (self.head == other.head) and compare_seq(self.expr, other.expr)
        except:
            return False

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

    def __str__(self):
        # helper function to get the string representation of a term
        sym = lambda term: term.value if isinstance(term, Token) else str(term)

        # if the expression is simple a constant
        if not self.expr:
            return sym(self.head)

        ret = ''
        a = sym(self.head)
        for f,b in self.expr:
            a = '%(f)s(%(a)s, %(b)s)' % {'f': f, 'a': a, 'b': sym(b)}
        return a

class AnonymousFunction(Expression):

    def __init__(self, args, expr):
        self.args = args
        self.expr = expr

    def __call__(self, *argv) :
        if len(argv) != len(self.args):
            raise TypeError("%s takes %i arguments but %i were given" %
                            (self, len(self.args), len(argv)))
        context = {self.args[i]: argv[i] for i in range(len(self.args))}
        return self.expr(**context)

    def __hash__(self):
        return hash((tuple(self.args), tuple(self.expr)))

    def __eq__(self, other):
        try:
            compare_seq = lambda x,y: all(x[i] == y[i] for i in range(max(len(x), len(y))))
            return compare_seq(self.args, other.args) and compare_seq(self.expr, other.expr)
        except:
            return False

    def __str__(self):
        return '%(args)s: %(expr)s' % {
            'args': 'f ' + ', '.join(self.args),
            'expr': str(self.expr)
        }

class Application(Expression):

    def __init__(self, function, args):
        self.function = function
        self.args = args

    def __call__(self, **context):
        if isinstance(self.function, Token):
            # `function` is symbol, we need to bound it to an actual function
            fx_name = self.function.value
            try:
                # first try to get `function` from the context
                fx = context[fx_name]
            except KeyError:
                # if `function` can't be bound from the context, try to use a built-in
                fx = None
                for mod in ('builtins', 'math',):
                    fx = getattr(importlib.import_module(mod), fx_name, None)
                    if fx: break

            # raise an evaluation error if `name` couldn't be bound
            if not fx:
                raise EvaluationError("unbound function name '%s'" % fx_name)

        elif isinstance(self.function, AnonymousFunction):
            # `function` is an anonymous function
            fx = self.function

        else:
            raise TypeError("invalid type '%s' for a function application" %
                            type(self.function).__name__)

        # apply fx
        # TODO instanciate yaffel sets as python iterable so we can call python
        #  built-in functions than run on tierables, such as `sum`
        return fx(*(self._value(a, context) for a in self.args))

    def __hash__(self):
        return hash(self.function, tuple(self.args))

    def __eq__(self, other):
        try:
            compare_seq = lambda x,y: all(x[i] == y[i] for i in range(max(len(x), len(y))))
            return (self.function, other.function) and compare_seq(self.args, other.args)
        except:
            return False

    def __str__(self):
        return '%s(%s)' % (self.function, ', '.join(map(str, self.args)))

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

    def __init__(self, *elements):
        self.elements = set(elements)

    def __call__(self, **context):
        return Enumeration(*{e(**context) for e in self.elements})

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