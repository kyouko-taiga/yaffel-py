from collections import namedtuple
from funcparserlib.lexer import make_tokenizer, Token, LexerError
from funcparserlib.parser import some, a, many, maybe, finished, skip, with_forward_decls
from functools import reduce

import sys, operator

keywords = ['for', 'in', 'not in']

class EvaluationError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class Function(object):

    def __init__(self, fst, expr=None):
        # print(hex(id(self)))
        self.fst = fst
        self.expr = expr

    def __call__(self, **context):
        # retrieve the first term value
        a = self._value(self.fst, context)

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
        elif isinstance(term, Function):
            return term(**context)
        else:
            return term

def tokenize(s):
    regexps = {
        'escaped':   r'\\(?P<standard>["\\/bfnrt])',
        'unescaped': r'[^"\\]' }
    grammar_specifications = [
        ('space',    (r'[ \t\r\n]+',)),
        ('number',   (r'-?(0|([1-9][0-9]*))(\.[0-9]+)?([Ee][+-][0-9]+)?',)),
        ('string',   (r'"[^"]*"',)),                                # unsupported escaped quotes
        ('operator', (r'(\*\*)|[{}\[\]\(\)\-\+\*/=><\.,:]',)),
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
        try:
            # Whenever an expression is parsed, an instance of Function is
            # created. Then, when we want to evaluate the result of the
            # expression for a given binding, this function will be called,
            # using the bindings as the function parameters.
            return x[0](**x[1])
        except TypeError:
            # If the expression is constant, then we don't need to provide
            # any bindings.
            return x[0]()
            # if not hasattr(x[0], '__call__'):
            #     return x[0]

    def make_function(head, tail):
        # return a function that will take unbound variables as parameters
        return Function(head, tail)

    def make_binding(t):
        return (token_value(t[0]), t[1])

    def make_context(head, tail):
        context = {head[0]: head[1]}
        for k,v in tail:
            if k in context: raise EvaluationError("'%s' is already bound" % k)
            context[k] = v
        return context

    # primitives
    op          = lambda s: a(Token('operator', s))
    op_         = lambda s: skip(op(s))

    kw          = lambda s: a(Token('name', s))
    kw_         = lambda s: skip(kw(s))

    add         = op('+') >> const(operator.add)
    sub         = op('-') >> const(operator.sub)
    mul         = op('*') >> const(operator.mul)
    div         = op('/') >> const(operator.truediv)
    pow         = op('**') >> const(operator.pow)

    name        = token_type('name')
    number      = token_type('number') >> token_value >> make_number
    string      = token_type('string') >> token_value

    # grammar rules
    mul_op      = mul | div
    add_op      = add | sub

    atom        = with_forward_decls(lambda: number | name | (op_('(') + expression + op_(')')))
    factor      = atom + many(pow + atom) >> uncurry(make_function)
    term        = factor + many(mul_op + factor) >> uncurry(make_function)
    expression  = term + many(add_op + term) >> uncurry(make_function)

    binding     = with_forward_decls(lambda: name + op_('=') + evaluable >> (make_binding))
    context     = binding + many(op_(',') + binding) >> uncurry(make_context)

    evaluable   = expression + maybe(kw_('for') + context) >> eval_expr
    yaffel      = evaluable + skip(finished)
    #yaffel      = expression

    #print(tokenize(seq))

    # tokenize and parse the given sequence
    parsed = yaffel.parse(tokenize(seq))
    return (type(parsed), parsed)

if __name__ == '__main__':
    #print(tokenize(sys.argv[1]))
    print( '%s %s' % parse(sys.argv[1]) )
    #print( parse(sys.argv[1])[1](**{'x':24}) )
