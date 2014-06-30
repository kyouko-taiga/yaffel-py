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
from funcparserlib.lexer import make_tokenizer, Token
from funcparserlib.parser import some, a, many, maybe, finished, skip, forward_decl, NoParseError
from functools import reduce

from yaffel.datatypes import *

import operator, sys

keywords = ['for', 'in', 'not in']

def tokenize(s):
    regexps = {
        'escaped':   r'\\(?P<standard>["\\/bfnrt])',
        'unescaped': r'[^"\\]' }
    grammar_specifications = [
        ('space',    (r'[ \t\r\n]+',)),
        ('number',   (r'-?(0|([1-9][0-9]*))(\.[0-9]+)?([Ee][+-][0-9]+)?',)),
        ('string',   (r'"[^"]*"',)),                                # unsupported escaped quotes
        ('operator', (r'(\*\*)|([><=!]=)|(and)|(or)|(not)|(in)|[{}\[\]\(\)\-\+\*/=><\.,:]',)),
        ('name',     (r'[A-Za-z_][A-Za-z_0-9]*',)),
    ]

    t = make_tokenizer(grammar_specifications)
    return [x for x in t(s) if x.type not in ['space']]

# auxiliary helper functions
const       = lambda x: lambda _: x
u           = lambda f: lambda x: f(*x)
token_value = lambda t: t.value
token_type  = lambda t: some(lambda x: x.type == t)

# semantic actions
def make_number(t):
    try:
        return int(t)
    except ValueError:
        return float(t)

def make_name(t):
    return Name(t)

def make_bool(t):
    return t == 'True'

def logical_and(op):
    return lambda x,y: bool(x) and bool(y)
def logical_or(op):
    return lambda x,y: bool(x) or bool(y)

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

    # don't create an additional function if 'head' is the only term and is
    # already callable
    if not tail and hasattr(head, '__call__'):
        return head

    # return a function that will take unbound variables as parameters
    return Expression([head] + tail)

def make_renaming(expr, context):
    if context:
        expr.rename_variable(context)
    return expr

def make_boolean(x):
    e = Application(x[0], (x[1],)) if x[0] else x[1]
    return make_expression(e, [])

def make_conditional(expr, branch):
    if branch is None:
        return expr

    condition = branch[0]
    else_expr = branch[1]
    return ConditionalExpression(expr, condition, else_expr)

def make_binding(name, value):
    return (name, value)

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
        return Enumeration(e)

    # return the empty set
    return Enumeration([])

def make_range(lower, upper):
    return Range(lower, upper)

def make_set(function, context):
    return Set(function, context)

def make_tuple(x):
    if x is not None:
        return tuple([x[0]] + [e for e in x[1]])
    return tuple()

def make_application(function, args):
    return Application(function, args)

def make_lambda(x):
    if x[0] is not None:
        args = [x[0]] + x[1]
        expr = x[2]
    else:
        args = []
        expr = x[1]

    return AnonymousFunction(args, expr)

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

and_        = op('and') >> logical_and
or_         = op('or') >> logical_or
not_        = op('not') >> const(operator.not_)

lt          = op('<') >> const(operator.lt)
le          = op('<=') >> const(operator.le)
eq          = op('==') >> const(operator.eq)
ne          = op('!=') >> const(operator.ne)
ge          = op('>=') >> const(operator.ge)
gt          = op('>') >> const(operator.gt)

true        = kw('True') >> token_value >> make_bool
false       = kw('False') >> token_value >> make_bool

name        = token_type('name') >> token_value >> make_name
number      = token_type('number') >> token_value >> make_number
string      = token_type('string') >> token_value

# grammar rules
mul_op      = mul | div
add_op      = add | sub
cmp_op      = lt | le | eq | ne | ge | gt

# forward declatations
nexpr       = forward_decl()
bexpr       = forward_decl()
sexpr       = forward_decl()
expr        = forward_decl()

application = forward_decl()
renaming    = forward_decl()
set_context = forward_decl()

# numerical expression
numeric     = application | number | name | (op_('(') + nexpr + op_(')'))
factor      = numeric + many(power + numeric) >> u(make_expression)
term        = factor + many(mul_op + factor) >> u(make_expression)
nexpr.define( term + many(add_op + term) >> u(make_expression) )

# boolean expression
pred        = nexpr + cmp_op + nexpr >> u(make_expression)
formula     = true | false | pred | nexpr | (op_('(') + bexpr + op_(')'))
conjuction  = formula + many(and_ + formula) >> u(make_expression)
disjunction = conjuction + many(or_ + conjuction) >> u(make_expression)
bexpr.define( maybe(not_) + disjunction >> make_boolean )

# set expression
enumeration = op_('{') + maybe(expr + many(op_(',') + expr)) + op_('}') >> make_enum
range_      = op_('{') + nexpr + op_(':') + nexpr  + op_('}') >> u(make_range)
set_        = op_('{') + expr + maybe(kw_('for') + set_context) + op_('}') >> u(make_set)
sexpr.define( enumeration | range_ | set_ )

# anonymous function
lambda_     = op_('[') + maybe(name + many(op_(',') + name)) + op_(':') + expr + op_(']') \
              >> make_lambda

# function application
tuple_      = op_('(') + maybe(expr + many(op_(',') + expr)) + op_(')') >> make_tuple
application.define( (lambda_ | name) + tuple_ >> u(make_application) )

# conditional expression
uexpr       = sexpr | bexpr | nexpr
cexpr       = uexpr + kw_('if') + bexpr + maybe(kw_('else') + uexpr) >> u(make_conditional)

# expression context
binding     = name + op_('=') + (renaming | op_('(') + renaming + op_(')')) >> u(make_binding)
context     = binding + many(op_(',') + binding) >> u(make_context)
renaming.define( expr + maybe(kw_('for') + context) >> u(make_renaming) )

# set expression context
set_binding = name + op_('in') + sexpr >> u(make_binding)
set_context.define( set_binding + many(op_(',') + set_binding) >> u(make_context) )

# any expression
expr.define( cexpr | uexpr )
yaffel = expr + maybe(kw_('for') + context) + skip(finished) >> eval_expr
#yaffel = cexpr

def parse(seq):
    try:
        # tokenize and parse the given sequence
        parsed = yaffel.parse(tokenize(seq))
    except NoParseError as e:
        raise SyntaxError(e.msg)

    return (type(parsed), parsed)

if __name__ == '__main__':
    #print(tokenize(sys.argv[1]))
    print( '%s %s' % parse(sys.argv[1]) )
    #print( parse(sys.argv[1])[1](**{'x':24}) )
