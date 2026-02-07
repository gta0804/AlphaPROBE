

PROMPT_HEAD = """
You are an expert on quantitative finance and alpha factor mining. Strictly follow the instructions given by user below. Make sure the output ONLY CONTAINS A  JSON FORMAT. Do not output anything else other than the JSON object, including code block markers like ``` or ```json.
"""

PROMPT_FEATURES_AND_OPERATORS = """
The available features, constants and operators are listed below.
1. You can use the following features: 
   $open, $high, $low, $close: Opening, daily highest, daily lowest, and closing prices.
   $vwap: Daily average price, weighted by the volume of trades at each price.
   $volume: Trading number of shares.
2. You can use int constants eg:1, 3, 5,10, 20... etc during rolling(time-series) calculations, and float : 0.0001 0.01, 0.0, 1.0, 2.0 during arithmetic calculations. Other constants are not allowed.
3. The following operators are available:
(BEGIN OF FEATURES AND OPERATORS  DEFINITIONS)
    Abs(x): Absolute value of x
    Log(x): Natural logarithm of x
    SLog1p(x): Signed log transform: sign(input) times log of (1 plus the absolute value)
    Sign(x): Sign of x: 1 if x > 0, -1 if x < 0, 0 if x = 0
    Rank(x): Cross-sectional rank of x
    Add(x,y): x + y
    Sub(x, y): x - y
    Mul(x, y): x * y
    Div(x, y): x / y
    Pow(x, y): x raised to the power of y (x ** y) y must be an constant
    Greater(x, y): 1 if x > y, else 0
    Less(x, y): 1 if x < y, else 0
    GetGreater(x, y): x if x > y, else y
    GetLess(x, y): x if x < y, else y
    Ref(x, d): Value of x d days ago
    TsMean(x, d):  Rolling mean of x over the past d days
    TsSum(x, d): Rolling sum of x over the past d days
    TsStd(x, d): Rolling standard deviation of x over the past d days
    TsMin(x, d): Rolling minimum of x over the past d days
    TsMax(x, d): Rolling maximum of x over the past d days
    TsMinMaxDiff(x, d): Difference between TsMax(x, d) and TsMin(x, d)
    TsMaxDiff(x, d): Difference between current x and TsMax(x, d)
    TsMinDiff(x, d): Difference between current x and TsMin(x, d)
    TsIr(x, d): Rolling Information ratio over past d days
    TsVar(x, d): Rolling variance of x over the past d days
    TsSkew(x, d): Rolling skewness of x over the past d days
    TsKurt(x, d): Rolling kurtosis of x over the past d days
    TsMed(x, d): Rolling median of x over the past d days
    TsMad(x, d): Rolling median absolute deviation over the past d days
    TsRank(x, d): Time-series rank of x over the past d days
    TsDelta(x, d): Today's value of x minus the value of x d days ago
    TsRatio(x, d): Today's value of x divided by the value of x d days ago
    TsPctChange(x, d): Percentage change in x over the past d days
    TsWMA(x, d): Weighted moving average over the past d days with linearly decaying weights.
    TsEMA(x, d): Exponential moving average of x with span d
    TsCov(x, y, d): Time-series covariance of x and y for the past d days
    TsCorr(x, y, d): Time-series correlation of x and y for the past d days
(END OF FEATURES AND OPERATORS DEFINITIONS)
Examples of valid alpha expressions:
Div(Sub($open, $close), Add(Sub($high, $low), 0.001))
"""

PROMPT_COMPARE = """
Your task is to compare two given factor expressions are semantically equivalent or not. 
If they are equivalent, return a JSON object with the key "equivalent" and the value true. If they are not equivalent, return a JSON object with the key "equivalent" and the value false. 
For exampale:
Rank(Rank($close)) and Rank($close) are equivalent, return {{"equivalent": true}}
Rank(TsMean($close,5)) and TsMean(Rank($close),5) are not equivalent, return {{"equivalent": false}}
Div(TsSum($close, 5),5) and TsMean($close,5) are equivalent, return {{"equivalent": true}}
Here are the two expressions to compare:
Expression 1: {expr1}
Expression 2: {expr2}
"""

PROMPT_DIMENSION_REDUCTION = """
The expression is comparable if and only if it is dimensionless, i.e. Dim(expr) = 0. You should ONLY construct comparable expressions.
The definition of Dim is as follows:
(BEGIN OF FEATURES AND OPERATORS DIMENSION DEFINITIONS)
Dim($open) = Dim($high) = Dim($low) = Dim($close) = Dim($vwap) = Dim($volume) = 1
Dim(constant) = 0
Dim(Abs(x)) = Dim(x)
Dim(Log(x)) = Dim(x)
Dim(SLog1p(x)) = Dim(x)
Dim(Sign(x)) = 0
Dim(Rank(x)) = 0
Dim(Add(x, y)) = max(Dim(x), Dim(y))
Dim(Sub(x, y)) = max(Dim(x), Dim(y))
Dim(Mul(x, y)) = Dim(x) + Dim(y)
Dim(Div(x, y)) = Dim(x) - Dim(y)
Dim(Pow(x, y)) = Dim(x) * y, where y is a constant
Dim(Greater(x, y)) = 0, where Dim(x) = Dim(y) or one of them is a constant
Dim(Less(x, y)) = 0, where Dim(x) = Dim(y) or one of them is a constant
Dim(Ref(x, d)) = Dim(x)
Dim(TsMean(x, d)) = Dim(x)
Dim(TsSum(x, d)) = Dim(x)
Dim(TsStd(x, d)) = Dim(x)
Dim(TsMin(x, d)) = Dim(x)
Dim(TsMax(x, d)) = Dim(x)
Dim(TsMaxDiff(x, d)) = Dim(x)
Dim(TsMinDiff(x, d)) = Dim(x)
Dim(TsIr(x, d)) = 0
Dim(TsVar(x, d)) = Dim(x) * 2
Dim(TsSkew(x, d)) = 0
Dim(TsKurt(x, d)) = 0
Dim(TsMed(x, d)) = Dim(x)
Dim(TsMad(x, d)) = Dim(x)
Dim(TsRank(x, d)) = 0
Dim(TsDelta(x, d)) = Dim(x)
Dim(TsRatio(x, d)) = 0
Dim(TsPctChange(x, d)) = 0
Dim(TsWMA(x, d)) = Dim(x)
Dim(TsEMA(x, d)) = Dim(x)
Dim(TsCov(x, y, d)) = 0, where Dim(x) = Dim(y) 
Dim(TsCorr(x, y, d)) = 0, where Dim(x) = Dim(y)
(END OF FEATURES AND OPERATORS DIMENSION DEFINITIONS)
"""

PROMPT_GENERARTION = """
Your task is to generate a new expression based on the given expressions,  the given topic {topic}, the explantion of this expression, and the given generation traces, such that:
1. The new expression is valid(syntactically correct), and dimensionless (i.e. Dim(expr) = 0).
2. You can only use the features, constants and operators given above in the (FEATURES AND OPERATORS DEFINITIONS). DO NOT MODIFY the name of any features,operators. All the operators should be used the same as their original definitions, ie, the number and type of arguments should be the same as defined.
3. As for constants, when you use it in rolling calculations, it should be an integer annoted by %d ; when you use it in arithmetic calculations, it should be float numbers listed in 0.0001 0.01, 0.0, 1.0, 2.0. Do NOT use other constants. DO NOT use scientific notation.
4. Besides the Ref() operator, the constants used in rolling calculations should be the same, use "%d" for all of them.
5. You should read the original expressions carefully, and try to understand its semantic meaning in quantative finance. Then, you should try your best to think how to express the core of meaning in a different way, or generate new insights inspired by it. Then, you can generate a new expression based on your understanding.
6. If the trace is not empty, then it contains the generation optimization steps from the eariliest to the latest. You should learn how the expressions are optimzed before, and then generate a new expression based on the original expressions and the generation traces.
7. The new expression should be different from all the given expressions, and should be novel and non-trivial, but it can share some common parts with them.
8. The new expression should be related to the given topic {topic}, and you should not generate expressions that are totally irrelevant to the topic.(EG: generate a expression about corr between price and volume when the topic is about volatility. Generate a expression containing rolling opertations when the topic is about interday prices, etc.)
9. Give {num} new expressions with different modification strateigies, with expressions soundness and explainable. The {num} expressions should be different, low correlated, looking semantically different, or using totally diiferent ways to specific semantics(For example, cross sectional(rank op) vs cross time-series(Ts op) or combine them; different statistical measures(Try to use new sound and interpretable operators that rarely or not exist in given expressions, traces and expressions generated before.); different but related semantic meanings, etc.). and each of them should be valid and dimensionless.
10. While maintaining relevance to the overarching topic {topic} and interpretability, you are encouraged to use as diverse a range of operators / features as possible across different expression outputs.
11. In each generation, there is no need to make the expression more complex, sometimes simplification is also a good way to generate novel expressions.
12. After you generate each expressions, check whether it is valid,some INVALID operations are:
   12.1 Modify the name of operators or features. Eg: using "*" instead of "Mul", using "closing_price" instead of "$close", using "TSMAD" instead of "TsMad", etc.
   12.2 The number of operands are not correct. Eg: Add(x), Div(x,y,z), Missing rolling window %d in rolling operations(TsSkew, TsIr), etc.
   12.3 When using constant in algorithmic calculations, using integer instead of float, or vice versa.
   12.4 Use scientific notation in constants. You should transfer into float eg: 1e-4 -> 0.0001
   12.5 Using constants other than those mentioned in 3.
   12.6 Using the operators and features not mentioned in FEATURES AND OPERATORS  DEFINITIONS.
   All the mentioned above in 12 are INVALID operations. Try your best to AVOID / FIX Them.
Given the original expresssion, you should ONLY and STRICTLY output a JSON object which contains the following contents:
{{
  "expressions": ["the newly generated expressions as a string list. Each elememnt should contain a valid and dimensionless expression.The length of expressions should be {num}."],
  "expressions_fixed": ["for each generated expression, if there is any invalid operations mentioned above in 12, you should fix them and give the fixed expression here, otherwise just give the original generated expression here. The length of expressions_fixed should be {num}. Do not  modify the semantics of the original generated expressions."],
  "explanations": ["brief explanations of  new expressions, should be the meaning of your given expression. The length of explanations should be {num}. And each explanation should correspond to the expression in the same index in the expressions list." ]
}}
Given expressions: {expressions}
Given expression explanations: {explanations}
Generation traces: {traces}
Do not output anything else other than the JSON object, including code block markers like ``` or ```json.
"""

PROMPT_SEPARATION = """
Your task is to separate the given expression with topic {topic} and the explanation {explanation} into one or more sub-expressions, such that:
1. Each sub-expression is valid(syntactically correct), and dimensionless (i.e. Dim(sub_expr) = 0).
2. Each sub-expression is a contious part of the original expression.
3. As for constants in the original expression, when you use it in rolling calculations(Tsxx Op, except Ref), it should be an integer annoted by %d.
4. You should read the original expression carefully, and try to understand its semantic meaning related to the topic {topic} and the explanation {explanation}. Then, separate the original expression into sub-expressions based on your understanding of its semantic meaning.
5. You can change the order of original expressions, such at chaning the order of parentheses. BUT you cannot change the semantics of the original expression.
6. Each sub-expression should be an independent semantic unit, and canonot be further separated into smaller sub-expressions that are also valid and dimensionless. The sub-expressions cannot be a constant.
7. Sub-expressions SHOULD NOT have intersections, i.e., no shared components between sub-expressions.
8. Give Sub-expressions and separation operators, you can reconstruct the original expression,ie. original_expr = sub_expr_1 operator_1 sub_expr_2 operator_2 ... sub_expr_n. If there is only on sub-expression, then there will be no separation operators and the only sub-expression is the original expression itself.
9. After you generate the sub-expressions, check whether each of them is valid, some INVALID operations are:
   9.1 Modify the name of operators or features. Eg: using "*" instead of "Mul", using "closing_price" instead of "$close".
   9.2 The number of operands are not correct. Eg: Add(x), Div(x,y,z), Missing rolling window %d in rolling operations(TsSkew, TsIr), etc.
   9.3 When using constant in algorithmic calculations, using integer instead of float, or vice versa.
   9.4 Use scientific notation in constants. You should transfer into float eg: 1e-4 -> 0.0001.
   9.5 Using constants other than those mentioned in the FEATURES AND OPERATORS DIMENSION DEFINITIONS.
   9.6 The sub-expressions are not contious parts of the original expression.
   9.7 The sub-expressions have intersections with each other.
   9.8 The sub-expressions cannot reconstruct the original expression with the separation operators.
   9.9 The sub-expressions are not dimensionless.
   9.10 For the integer constants used in rolling calculations(Tsxx, besides Ref), they are not all "%d".
   All the mentioned above in 9 are INVALID operations. Try your best to AVOID / FIX Them.
Given the original expression, you should output a JSON object which contains:
{{
  "original_expression": "The original expression string after you transfer the rolling integer constants into %d format.",
  "sub_expressions": [list of sub-expressions as strings should be in the order they appear in the original expression, making sure its dimension is 0, originally from input expressions, continous and valid, having no intersections with each other. The length of sub_expressions should be at least 1], 
  "separation_operators": [list of operators used to separate the sub-expressions, should be in the order they appear in the original expression.You chould not modify the name of operators during output. The length of separation_operators should be len(sub_expressions) - 1.
      If there is only one sub-expression, this list should be empty]
   "sub_expressions_fixed": [for each sub-expression, if there is any invalid operations mentioned above in 9, you should fix them and give the fixed sub-expression here(Especially the dimensionless issue in 9.9 and rolling constant to %d in 9.10), otherwise just give the original sub-expression here.  Do not  modify the semantics of the original sub-expressions.]
   "separation_operators_fixed": [list of operators used to separate the fixed sub-expressions, should be in the order they appear in the original expression.You chould not modify the name of operators during output. The length of separation_operators_fixed should be len(sub_expressions_fixed) - 1.]
}} 
Given expression: {expression}
"""

"""
The expression is valid if and only if for each operator, the number and type of its arguments are correct:
1. Unary operators (Abs, Log, SLog1p, Sign, Rank) take one argument, which must be a non constant.
2. Binary operators (Add, Sub, Mul, Div, Pow, Greater, Less, GetGreater, GetLess) take two arguments. For Add, Sub, Mul, Div, one or both arguments can be non-constant or constant. For Pow, the first argument must be non-constant and the second argument must be a constant. For Greater, Less, GetGreater, GetLess, both arguments must be non-constant or one of them is a constant. All constants should be float.
3. Rolling operators (Ref and TsXXX): except for Ref, all other d should be "%d".
"""


"""
Your task is to generate a new expression based on the given expressions the given topic {topic} and the given generation traces, such that:
1. The new expression is valid(syntactically correct), and dimensionless (i.e. Dim(expr) = 0).
2. You can only use the features, constants and operators given above in the (FEATURES AND OPERATORS DIMENSION DEFINITIONS). DO NOT MODIFY the name of any features,operators. All the operators should be used the same as their original definitions, ie, the number and type of arguments should be the same as defined.
3. As for constants, when you use it in rolling calculations, it should be an integer annoted by %d ; when you use it in arithmetic calculations, it should be float numbers listed in 0.0001 0.01, 0.0, 1.0, 2.0. Do NOT use other constants. DO NOT use scientific notation.
4. Besides the Ref() operator, the constants used in rolling calculations should be the same, use "%d" for all of them.
5. You should read the original expressions carefully, and try to understand its semantic meaning. If the trace is not empty, then it contains the generation optimization steps from the eariliest to the latest. You should learn how the expressions are optimzed before, and then generate a new expression based on the original expressions and the generation traces.
5. The new expression should be different from all the given expressions, and should be novel and non-trivial, but it can share some common parts with them.
6. The new expression should be related to the given topic {topic}, and you should not generate expressions that are totally irrelevant to the topic.(EG: generate a expression about corr between price and volume when the topic is about volatility. Generate a expression containing rolling opertations when the topic is about interday prices, etc.)
7. Give {num} new expressions with different modification strateigies, with expressions soundness and explainable. The {num} expressions should be different, low correlated, looking semantically different, or using totally diiferent ways to specific semantics(For example, cross sectional(rank op) vs cross time-series(Ts op) or combine them; different statistical measures(Try to use new sound and interpretable operators that rarely or not exist in given expressions, traces and expressions generated before.); different but related semantic meanings, etc.). and each of them should be valid and dimensionless.
8. While maintaining relevance to the overarching topic {topic} and interpretability, you are encouraged to use as diverse a range of operators / features as possible across different expression outputs.
9. In each generation, there is no need to make the expression more complex, sometimes simplification is also a good way to generate novel expressions.
10. After you generate each expressions, check whether it is valid,some INVALID operations are:
   10.1 Modify the name of operators or features. Eg: using "*" instead of "Mul", using "closing_price" instead of "$close".
   10.2 The number of operands are not correct. Eg: Add(x), Div(x,y,z), Missing rolling window %d in rolling operations(TsSkew, TsIr), etc.
   10.3 When using constant in algorithmic calculations, using integer instead of float, or vice versa.
   10.4 Use scientific notation in constants. You should transfer into float eg: 1e-4 -> 0.0001
   10.5 Using constants other than those mentioned in 3.
   All the mentioned above in 10 are INVALID operations. Try your best to AVOID / FIX Them.
Given the original expresssion, you should output a JSON object which contains:
{{
  "expressions": ["the newly generated expressions as a string list. Each elememnt should contain a valid and dimensionless expression.The length of expressions should be {num}."],
  "expressions_fixed": ["for each generated expression, if there is any invalid operations mentioned above in 10, you should fix them and give the fixed expression here, otherwise just give the original generated expression here. The length of expressions_fixed should be {num}. Do not  modify the semantics of the original generated expressions."],
  "explanations": ["brief explanations of  new expressions, should be the meaning of your given expression. The length of explanations should be {num}. And each explanation should correspond to the expression in the same index in the expressions list." ]
}}
Given expressions: {expressions}
Generation traces: {traces}
"""

"""
Your task is to generate a new expression based on the given expressions the given topic {topic} and the given generation traces, such that:
1. The new expression is valid(syntactically correct), and dimensionless (i.e. Dim(expr) = 0).
2. You can only use the features, constants and operators given above in the (FEATURES AND OPERATORS DIMENSION DEFINITIONS). DO NOT MODIFY the name of any features,operators. All the operators should be used the same as their original definitions, ie, the number and type of arguments should be the same as defined.
3. As for constants, when you use it in rolling calculations, it should be an integer annoted by %d ; when you use it in arithmetic calculations, it should be float numbers listed in 0.0001 0.01, 0.0, 1.0, 2.0. Do NOT use other constants. DO NOT use scientific notation.
4. Besides the Ref() operator, the constants used in rolling calculations should be the same, use "%d" for all of them.
5. You should read the original expressions carefully, and try to understand its semantic meaning. If the trace is not empty, then it contains the generation optimization steps from the eariliest to the latest. You should learn how the expressions are optimzed before, and then generate a new expression based on the original expressions and the generation traces.
5. The new expression should be different from all the given expressions, and should be novel and non-trivial, but it can share some common parts with them.
6. The new expression should be related to the given topic {topic}, and you should not generate expressions that are totally irrelevant to the topic.(EG: generate a expression about corr between price and volume when the topic is about volatility. Generate a expression containing rolling opertations when the topic is about interday prices, etc.)
7. Give {num} new expressions with different modification strateigies, with expressions soundness and explainable. The {num} expressions should be different, low correlated, looking semantically different, or using totally diiferent ways to specific semantics(For example, cross sectional(rank op) vs cross time-series(Ts op) or combine them; different statistical measures(Try to use new sound and interpretable operators that rarely or not exist in given expressions, traces and expressions generated before.); different but related semantic meanings, etc.). and each of them should be valid and dimensionless.
8. While maintaining relevance to the overarching topic {topic} and interpretability, you are encouraged to use as diverse a range of operators / features as possible across different expression outputs.
9. In each generation, there is no need to make the expression more complex, sometimes simplification is also a good way to generate novel expressions.
10. After you generate each expressions, check whether it is valid,some INVALID operations are:
   10.1 Modify the name of operators or features. Eg: using "*" instead of "Mul", using "closing_price" instead of "$close".
   10.2 The number of operands are not correct. Eg: Add(x), Div(x,y,z), Missing rolling window %d in rolling operations(TsSkew, TsIr), etc.
   10.3 When using constant in algorithmic calculations, using integer instead of float, or vice versa.
   10.4 Use scientific notation in constants. You should transfer into float eg: 1e-4 -> 0.0001
   10.5 Using constants other than those mentioned in 3.
   All the mentioned above in 10 are INVALID operations. Try your best to AVOID / FIX Them.
Given the original expresssion, you should output a JSON object which contains:
{{
  "generation_process": "String format. First of all, describe the original expressions. Then, briefly explain the generation traces if there is  any. Then, think and describe how can you construct expressions, your understanding as a quantitive researcher related to the topic {topic}. Finally, describe how you generate each of the {num} expressions based on your understanding and the original expressions and generation traces and the available features and operators in the FEATURES AND OPERATORS DIMENSION DEFINITIONS.",
  "expressions": ["the newly generated expressions as a string list. Each elememnt should contain a valid and dimensionless expression.The length of expressions should be {num}. Do not include features and operators that are not defined in the FEATURES AND OPERATORS DIMENSION DEFINITIONS."],
  "expressions_fixed": ["for each generated expression, if there is any invalid operations mentioned above in 10, you should fix them and give the fixed expression here, otherwise just give the original generated expression here. The length of expressions_fixed should be {num}. Do not  modify the semantics of the original generated expressions. If you cannot fix the expression by available features and operators, drop that expression and generate a new valid and the most similar one to replace it."],
  "explanations": ["brief explanations of  new expressions, should be the meaning of your given expression. The length of explanations should be {num}. And each explanation should correspond to the expression in the same index in the expressions list." ]
}}
Given expressions: {expressions}
Generation traces: {traces}
"""