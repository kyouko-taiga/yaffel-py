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

from collections import namedtuple
from funcparserlib.lexer import make_tokenizer, Token, LexerError
from funcparserlib.parser import some, a, many, maybe, finished, skip, with_forward_decls
from functools import reduce

import numbers, importlib, operator, sys

keywords = ['for', 'in', 'not in']

class EvaluationError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

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

class LazyFunction(Expression):

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __call__(self, **context):
        try:
            # first try to get `name` from the context
            fx = context[self.name]
        except KeyError:
            # if `name` can't be bound from the context, try to use a built-in
            fx = None
            for mod in ('builtins', 'math',):
                fx = getattr(importlib.import_module(mod), self.name, None)
                if fx: break

        # raise an evaluation error if `name` couldn't be bound
        if not fx:
            raise EvaluationError("unbound variable '%s'" % self.name)

        # apply fx
        # TODO instanciate yaffel sets as python iterable so we can call python
        #  built-in functions than run on tierables, such as `sum`
        return fx(*(self._value(a, context) for a in self.args))


    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

    def __str__(self):
        return '%s(%s)' % (self.name, ', '.join(map(str, self.args)))

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

    def __repr__(self):
        return '%s(%s)' % (self.__class__, str(self))

    def __str__(self):
        return '{%s:%s}' % (repr(self.lower_bound), self.upper_bound)

def tokenize(s):
    regexps = {
        'escaped':   r'\\(?P<standard>["\\/bfnrt])',
        'unescaped': r'[^"\\]' }
    grammar_specifications = [
        ('space',    (r'[ \t\r\n]+',)),
        ('number',   (r'-?(0|([1-9][0-9]*))(\.[0-9]+)?([Ee][+-][0-9]+)?',)),
        ('string',   (r'"[^"]*"',)),                                # unsupported escaped quotes
        ('operator', (r'(\*\*)|(and)|(or)|(in)|[{}\[\]\(\)\-\+\*/=><\.,:]',)),
        ('name',     (r'[A-Za-z_][A-Za-z_0-9]*',)),
    ]

    t = make_tokenizer(grammar_specifications)
    return [x for x in t(s) if x.type not in ['space']]

def parse(seq):
    # auxiliary helper functions
    const       = lambda x: lambda _: x
    uncurry     = lambda f: lambda x: f(*x)
    token_value = lambda t: t.value
    token_type  = lambda t: some(lambda x: x.type == t)

    # semantic actions
    def make_number(t):
        try:
            return int(t)
        except ValueError:
            return float(t)

    def eval_cst_expr(head, tail):
        return reduce(lambda s, p: p[0](s, p[1]), tail, head)

    def eval_expr(x):
        if hasattr(x[0], '__call__'):
            # Whenever an expression is parsed, an instance of Expression is
            # created. Then, when we want to evaluate the result of the
            # expression for a given binding, this function will be called,
            # using the context bindings as the function parameters.
            context = x[1] or {}
            return x[0](**context)

        # If the expression is constant, we don't need to evaluate it.
        return x[0]

    def make_expression(head, tail):
        # try to evaluate as a constant expression, if possible
        # terms = [head] + [t for _,t in tail]
        # if not any(isinstance(t, Token) or hasattr(t, '__call__') for t in terms):
        #     return eval_cst_expr(head, tail)

        # don't create an additional function if 'head' is the only term and is already callable
        if not tail and hasattr(head, '__call__'):
            return head

        # return a function that will take unbound variables as parameters
        return Expression(head, tail)

    def make_binding(t):
        return (token_value(t[0]), t[1])

    def make_context(head, tail):
        context = {head[0]: head[1]}
        for k,v in tail:
            if k in context: raise EvaluationError("'%s' is already bound" % k)
            context[k] = v
        return context

    def make_enum(x):
        # check that the enumeration is not the empty set
        if x is not None:
            e = {x[0]} | {e for e in x[1]}
            return Enumeration(*e)

        # return the empty set
        return Enumeration()

    def make_range(x):
        return Range(x[0],x[1])

    def make_set(x):
        return Set(*x)

    def make_tuple(x):
        if x is not None:
            return tuple([x[0]] + [e for e in x[1]])
        return tuple()

    def make_lazy_function(x):
        return LazyFunction(token_value(x[0]), x[1])

    # primitives
    op          = lambda s: a(Token('operator', s))
    op_         = lambda s: skip(op(s))

    kw          = lambda s: a(Token('name', s))
    kw_         = lambda s: skip(kw(s))

    add         = op('+') >> const(operator.add)
    sub         = op('-') >> const(operator.sub)
    mul         = op('*') >> const(operator.mul)
    div         = op('/') >> const(operator.truediv)
    power       = op('**') >> const(operator.pow)
    and_        = op('and') >> const(operator.and_)
    or_         = op('or') >> const(operator.or_)

    name        = token_type('name')
    number      = token_type('number') >> token_value >> make_number
    string      = token_type('string') >> token_value

    # grammar rules
    mul_op      = mul | div
    add_op      = add | sub
    bin_op      = or_ | and_

    atom        = with_forward_decls(lambda:
                    fx_ins | number | name | set_expr | (op_('(') + expr + op_(')')))
    factor      = atom + many(power + atom) >> uncurry(make_expression)
    term        = factor + many(mul_op + factor) >> uncurry(make_expression)
    expr        = term + many((add_op | bin_op) + term) >> uncurry(make_expression)

    binding     = with_forward_decls(lambda: name + op_('=') + evaluation >> (make_binding))
    context     = binding + many(op_(',') + binding) >> uncurry(make_context)

    evaluable   = expr + maybe(kw_('for') + context) >> eval_expr
    evaluation  = evaluable | (op_('(') + evaluable + op_(')'))

    enumeration = op_('{') + maybe(expr + many(op_(',') + expr)) + op_('}') >> make_enum
    range_      = op_('{') + expr + op_(':') + expr  + op_('}') >> make_range
    set_        = with_forward_decls(lambda:
                    op_('{') + expr + maybe(kw_('for') + set_context) + op_('}') >> make_set)
    set_expr    = (enumeration | range_ | set_)

    set_binding = name + op_('in') + set_expr >> make_binding
    set_context = set_binding + many(op_(',') + set_binding) >> uncurry(make_context)

    tuple_      = op_('(') + maybe(expr + many(op_(',') + expr)) + op_(')') >> make_tuple
    fx_ins      = name + tuple_ >> make_lazy_function

    yaffel      = evaluable + skip(finished)
    #yaffel      = fx_ins

    # tokenize and parse the given sequence
    parsed = yaffel.parse(tokenize(seq))
    return (type(parsed), parsed)

if __name__ == '__main__':
    #print(tokenize(sys.argv[1]))
    print( '%s %s' % parse(sys.argv[1]) )
    #print( parse(sys.argv[1])[1](**{'x':24}) )
