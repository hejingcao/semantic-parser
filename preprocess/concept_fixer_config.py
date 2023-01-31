# -*- coding: utf-8 -*-

PUNCTUATIONS = '.,"?!;() '

IS_NUMBER = ['card', 'yofc', 'fraction', 'ord', 'numbered_hour']
IS_NAMED = ['named', 'mofy', 'dofw', 'season', 'named_n', 'holiday']
IS_MUCH = 'much-many_a'
IS_NEG = 'neg'
IS_COMP = ['comp', 'comp_less', 'superl']

SPLIT_STRINGS = [
    '-/',
    ('everytime', '', ['every', 'time']),
    # ('the like', ' ', ['the', 'like']),
    # (re.compile('(not) +(many)', re.IGNORECASE), ' ', [1, 2]),
    # (re.compile('(one) +(more)', re.IGNORECASE), ' ', [1, 2]),
    (re.compile('^y-mp'), '/', None),
    # 匹配 SKR200 这类
    (re.compile('^([^\w]*skr)([0-9\.]+[^\w]*)$'), '', [1, 2]),
    # 匹配 'b.a.t' 'U.S.' 这类
    (re.compile('^[a-z](\.[a-z])+\.?$'), '.', None)
]

NUMBERS = [['1', 'one', 'a', 'first'],
           ['2', 'two', 'second'],
           ['3', 'three', 'third'],
           ['4', 'four', 'fourth'],
           ['5', 'five', 'fifth'],
           ['6', 'six', 'sixth'],
           ['7', 'seven', 'seventh'],
           ['8', 'eight', 'eighth'],
           ['9', 'nine', 'ninth'],
           ['10', 'ten', 'tenth'],
           ['11', 'eleven', 'eleventh'],
           ['12', 'twelve', 'twelfth'],
           ['13', 'thirteen', 'thirteenth'],
           ['14', 'fourteen', 'fourteenth'],
           ['15', 'fifteen', 'fifteenth'],
           ['16', 'sixteen', 'sixteenth'],
           ['17', 'seventeen', 'seventeenth'],
           ['18', 'eighteen', 'eighteenth'],
           ['19', 'nineteenth', 'nineteen'],
           ['20', 'twenty', 'twentieth'],
           ['30', 'thirty', 'thirtieth'],
           ['40', 'forty', 'fortieth'],
           ['50', 'fifty', 'fiftieth'],
           ['60', 'sixty', 'sixtieth'],
           ['70', 'seventy', 'seventieth'],
           ['80', 'eighty', 'eightieth'],
           ['90', 'ninety', 'ninetieth'],
           ['100', 'hundred', 'hundredth'],
           ['1000', 'thousand', 'thousandth'],
           ['1000000', 'million', 'millionth'],
           ['1000000000', 'billion', 'billionth'],
           ['1000000000000', 'trillion', 'trillionth']]

CARD_TRANSFORM = {x[0]: re.compile('|'.join(x)) for x in NUMBERS}
CARD_REGEX = re.compile(r'[1-9][0-9]*|' + '|'.join(_ for x in NUMBERS for _ in x[1:]))
MUCH_REGEX = re.compile('more|much|many')
COMP_REGEX = re.compile('more|less|most|least')
PREFIX_REGEX = re.compile('_([a-z]+-|mid)_')
NEG_REGEX = re.compile('non|not')

REGEX_MAP = {
    'bad': re.compile('bad|worse|worst'),
    'good': re.compile('good|better|best')
}

NAME_ENTITY = {
    'un': re.compile('un|u\.n\.?'),
    'us': re.compile('us|u\.s\.?'),
    'att': re.compile('att|at&t')
}

# 特殊的情况, 可以视为相等
TOKEN_LEMMA_SPECIAL_CASE = set([('ft', 'foot'),
                                ('gray', 'grey'),
                                ('offshore', 'off-shore'),
                                ('n.m', 'new mexico'),
                                ('hi', 'hawaii'),
                                ('vice', 'co')])

# # 可以作为正常的结点
SPECIAL_LABELS_AS_NORMAL_NODE = ['named', 'card', 'ord', 'neg',
                                 'dofw',  # day of week
                                 'yofc',  # year of century
                                 'mofy',  # month of year
                                 'numbered_hour',
                                 'holiday', 'fraction', 'season', 'named_n',
                                 'year_range', 'much-many_a']

IGNORE_LABEL_SETS = [{'_both_q', 'card'}, {'_if_x_then', '_should_v_modal'}]
