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

class Set(object): pass

class Enumeration(Set):

    def __init__(self, *elements):
        self.elements = set(elements)

    def __str__(self):
        return str(self.elements)

class Range(Set):

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return '{%s:%s}' % (self.lower_bound, self.upper_bound)

def tokenize(s):
    regexps = {
        'escaped':   r'\\(?P<standard>["\\/bfnrt])',
        'unescaped': r'[^"\\]' }
    grammar_specifications = [
        ('space',    (r'[ \t\r\n]+',)),
        ('number',   (r'-?(0|([1-9][0-9]*))(\.[0-9]+)?([Ee][+-][0-9]+)?',)),
        ('string',   (r'"[^"]*"',)),                                # unsupported escaped quotes
        ('operator', (r'(\*\*)|(and)|(or)|[{}\[\]\(\)\-\+\*/=><\.,:]',)),
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

    def make_enum(x):
        # check that the enumeration is not the empty set
        if x[0] is not None:
            e = {eval_expr((x[0], x[2]))} | {eval_expr((e, x[2])) for e in x[1]}
            return Enumeration(*e)

        # return the empty set
        return Enumeration()

    def make_range(x):
        return Range(eval_expr((x[0], x[2])), eval_expr((x[1], x[2])))

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

    atom        = with_forward_decls(lambda: number | name | (op_('(') + expression + op_(')')))
    factor      = atom + many(power + atom) >> uncurry(make_function)
    term        = factor + many(mul_op + factor) >> uncurry(make_function)
    expression  = term + many((add_op | bin_op) + term) >> uncurry(make_function)

    binding     = with_forward_decls(lambda: name + op_('=') + evaluation >> (make_binding))
    context     = binding + many(op_(',') + binding) >> uncurry(make_context)

    enumeration = op_('{') + maybe(expression + many(op_(',') + expression)) \
                    + maybe(kw_('for') + context) + op_('}') >> make_enum
    range_      = op_('{') + expression + op_(':') + expression \
                    + maybe(kw_('for') + context) + op_('}') >> make_range

    evaluable   = expression + maybe(kw_('for') + context) >> eval_expr
    evaluation  = evaluable | (op_('(') + evaluable + op_(')'))
    yaffel      = (evaluable | range_ | enumeration) + skip(finished)
    #yaffel      = expression

    #print(tokenize(seq))

    # tokenize and parse the given sequence
    parsed = yaffel.parse(tokenize(seq))
    return (type(parsed), parsed)

if __name__ == '__main__':
    #print(tokenize(sys.argv[1]))
    print( '%s %s' % parse(sys.argv[1]) )
    #print( parse(sys.argv[1])[1](**{'x':24}) )
