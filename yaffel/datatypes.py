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
from yaffel.exceptions import UnboundValueError, InvalidExpressionError

import numbers, importlib

__all__ = ['Name', 'Expression', 'ConditionalExpression', 'AnonymousFunction', 'Application',
           'Set', 'Enumeration', 'Range']

def value_of(variable, context):
    #if hasattr(variable, '__call__'):
    if any(isinstance(variable, t) for t in (Expression, Application, Set)):
        # `variable` is either an instance of Expression or Application, we simply evaluate it
        return variable(**context)
    elif isinstance(variable, Name):
        try:
            # we try to bound `variable` from the `context`
            binding = context[variable]
            return binding if isinstance(binding, AnonymousFunction) else binding(**context)
        except TypeError:
            return context[variable]
        except KeyError:
            raise UnboundValueError("unbound variable '%s'" % variable) from None

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

    def rename_variable(self, context):
        a = self._unfolded_expr[0]
        if isinstance(a, Name):
            # `a` is a name and we want to rename it if it exists in the `context`
            self._unfolded_expr[0] = context.get(a,a)
        elif isinstance(a, Expression):
            # `a` is an expression so we delegate the renaming to it
            a.rename_variable(context)

        for i in range(1, len(self._unfolded_expr)):
            f,b = self._unfolded_expr[i]
            if isinstance(b, Name):
                # `b` is a name and we want to rename it if it exists in the `context`
                self._unfolded_expr[i] = (f, context.get(b,b))
            elif isinstance(b, Expression):
                # `b` is an expression so we delegate the renaming to it
                b.rename_variable(context)

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

class ConditionalExpression(Expression):
    """Represents a conditional expression.

    Conditional expressions are of the form "x if p else y" whose evaluation
    will be that of x if p is true, else that of y. x,y and p are expressions
    themselves, even possibly conditional expressions; p must evaluate to a
    boolean.
    """

    def __init__(self, expr, condition=None, else_expr=None):
        self._condition = condition or Expression([True])
        self._else_expr = else_expr
        if isinstance(expr, Expression):
            super().__init__(expr._unfolded_expr)
        else:
            super().__init__([expr])

    def __call__(self, **context):
        if bool(self._condition(**context)):
            return super().__call__(**context)
        elif self._else_expr is not None:
            return self._else_expr(**context)
        raise UnboundValueError("conditional expression '%s' has no else expression" % str(self))

    def __str__(self):
        return '%(expr)s if %(cond)s else %(else)s' % {
            'expr': self._unfolded_expr_str(),
            'cond': self._condition._unfolded_expr_str(),
            'else': self._else_expr._unfolded_expr_str() if self._else_expr else 'None',
        }

class AnonymousFunction(object):
    """Represents an anonymous function.

    An anonymous function allows an expression to be seen as a first-class
    citizen, and thus can be bound to a name to define recursive expressions.
    """

    def __init__(self, args, expr):
        self._args = args
        self._expr = expr

    def __call__(self, *argv, **context):
        if len(argv) != len(self._args):
            raise TypeError("%s takes %i arguments but %i were given" %
                            (self, len(self._args), len(argv)))
        context.update({self._args[i]: argv[i] for i in range(len(self._args))})
        return self._expr(**context)

    def rename_variable(self, context):
        # don't rename variables that needs to be bound in function arguments
        super().__init__({n:v for n,v in context.items() if n not in self._args})

    def __hash__(self):
        return hash(tuple(self._args + [super().__hash__()]))

    def __eq__(self, other):
        f = lambda x,y: all(a == b for a,b in zip_longest(x,y))
        return f(self._args, other._args) and f(self._unfolded_expr, other._unfolded_expr)

    def __str__(self):
        return '[%(args)s: %(expr)s]' % {
            'args': ', '.join(self._args),
            'expr': str(self._expr)
        }

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

class Application(object):
    """Represents the application of a function.

    A function application is a tuple <f,A> with f a function and A a list of
    arguments. When the application is evaluated, the function f is bound to an
    expression or a built-in function, representing its semantic. If f is bound
    to a yaffel expression, arguments may remain unbound until their evaluation
    is actually required; however, if f is bound to a built-in function, its
    arguments might be evaluated before f is actually called. This might happen
    because some built-in functions are pure python and doesn't support late
    binding.
    """

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
        except UnboundValueError:
            # if `function` can't be bound from the context, try to use a built-in
            fx = None
            for mod in ('builtins', 'math',):
                fx = getattr(importlib.import_module(mod), self._function, None)
                if fx: break

        # raise an evaluation error if `_function` couldn't be bound
        if not fx:
            raise UnboundValueError("unbound function name '%s'" % self._function)
        elif not hasattr(fx, '__call__'):
            raise TypeError("invalid type '%s' for a function application" %
                            type(fx).__name__)

        # apply fx
        # TODO instanciate yaffel sets as python iterable so we can call python
        # built-in functions than run on tierables, such as `sum`
        if isinstance(fx, AnonymousFunction):
            return fx(*(value_of(a, context) for a in self._args), **context)
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
        if not isinstance(other, Set):
            return False
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
        if not isinstance(other, Enumeration):
            return False
        return self.elements == other.elements

    def __contains__(self, item):
        return item in self.elements

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
        if not isinstance(other, Range):
            return False
        return (self.lower_bound == other.lower_bound) and (self.upper_bound == other.upper_bound)

    def __contains__(self, item):
        if not isinstance(item, numbers.Real):
            return False
        return (item >= self.lower_bound) and (item <= self.upper_bound)

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

    def __str__(self):
        return '{%s:%s}' % (repr(self.lower_bound), self.upper_bound)
