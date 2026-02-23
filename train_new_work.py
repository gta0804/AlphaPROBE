from alpha_knowledge.alpha_pool import AlphaKnowledgePool
from alphagen.data.expression_bst import ExpressionBST
from openai import OpenAI
from utils.llm import OpenAIModel
from tqdm import tqdm
from alphagen.data.expression import *
from alphagen_generic.features import *
from alphagen_qlib.stock_data import StockData
from alphagen.data.tree import ExpressionParser
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from alpha_knowledge.alpha_knowledge_trainer import AlphaKnowledgeTrainer

INITIAL_EXPR = [
    ('Div($high, $close)', 'Interday price movements', "This expression measures the daily high-to-close ratio by dividing the day's highest price by its closing price, which is useful for identifying intraday strength or weakness in price momentum and potential reversal signals."),
    ('Div(Sub($high, $low), $open)', 'Interday price movements', 'This expression measures the daily range volatility relative to the opening price by dividing the difference between the high and low prices by the open price, which is useful for identifying intraday price movement intensity and potential breakout or reversal signals.'),
    ('Div(Sub(Less($open, $close), $low), $open)', 'Interday price movements', 'This expression measures the normalized daily price reversal signal by calculating the difference between a binary indicator of positive close-open return (1 if close > open, else 0) and the low price, then scaling it by the opening price, which is useful for identifying potential mean-reversion opportunities based on intraday price behavior relative to the open.'),
    ('Div(Sub(Less($open, $close), $low), Add(Sub($high, $low), 0.000001))', '', 'This expression measures the normalized gap between the binary indicator of a bullish day (open < close) and the low price, scaled by the daily range plus a small constant, which is useful for capturing the relative strength of bullish signals while accounting for daily volatility and avoiding division by zero.'),
    ('Div(Sub($close, $open), Add(Sub($high, $low), 0.000001))', 'Interday price movements', 'This expression measures the normalized daily price change by dividing the difference between closing and opening prices by the daily price range (high minus low) with a small constant to avoid division by zero, which is useful for capturing intraday momentum and volatility-adjusted price movements.'),
    ('Div(Sub(Sub(Mul(2.0, $close), $high), $low), $open)', 'Interday price movements', 'This expression measures the normalized price momentum by calculating (2 * close - high - low) / open, which captures the deviation of the closing price from the daily range relative to the opening price, useful for identifying intraday strength and potential reversal signals.'),
    ('Div(Sub(Sub(Mul(2.0, $close), $high), $low), Add(Sub($high, $low), 0.000001))', 'Interday price movements', 'This expression measures the normalized price momentum by calculating the ratio of (twice the close minus high) to (high minus low plus a small constant), which is useful for identifying overbought or oversold conditions while avoiding division by zero.'),
    ('Div(Sub($high, GetGreater($open, $close)), $open)', 'Interday price movements', "This expression measures intraday price momentum and potential reversal signals by calculating the ratio of the day's high minus the greater of open or close to the opening price, which is useful for identifying stocks that have shown strong upward movement during the day but may be overextended relative to their opening levels."),
    ('Div(Sub($high, GetGreater($open, $close)), Add(Sub($high, $low), 0.000001))', 'Interday price movements', "This expression measures the normalized intraday upside volatility by computing the ratio of the day's high-to-max(open,close) gap to the daily high-low range (with a small constant to avoid division by zero), which is useful for identifying days with strong upward price momentum relative to the day's total volatility."),
    ('Div(TsStd($close, %d), $close)', 'Price volatility', 'This expression measures the relative volatility of a security by calculating the ratio of the rolling standard deviation of closing prices over the past %d days to the current closing price, which is useful for identifying periods of high normalized price fluctuation and assessing risk-adjusted price behavior.'),
    ('Div(TsMean(GetGreater(Sub($high, $low), GetGreater(Abs(Sub($high, Ref($close, 1))),Abs(Sub($low, Ref($close, 1))) )), %d), $close)', 'Price volatility', "This expression measures the average daily volatility relative to price by calculating the rolling mean of the greater of the day's high-low range and the gap from the previous close, then normalizing by the current close, which is useful for assessing normalized volatility persistence and identifying periods of elevated price instability."),
    ('Sub(TsMean(Greater($close, Ref($close, 1)), %d), TsMean(Less($close, Ref($close, 1)), %d))', 'Intraday price movements', 'This expression measures the difference between the average number of days with positive returns and the average number of days with negative returns over a rolling window, which is useful for identifying momentum or mean-reversion signals by quantifying the asymmetry in recent price movements.'),
    ('TsMean(Less($close, Ref($close, 1)), %d)', 'Intraday price movements', "This expression measures the proportion of days over the past %d days where the closing price decreased compared to the previous day by computing a rolling mean of a binary indicator (1 if today's close is less than yesterday's close, else 0), which is useful for identifying short-term downward price momentum or bearish sentiment trends."),
    ('TsMean(Greater($close, Ref($close, 1)), %d)', 'Intraday price movements', 'This expression measures the proportion of days in the past %d days where the closing price increased from the previous day by calculating the rolling mean of a binary indicator (1 for price increase, 0 otherwise), which is useful for identifying momentum or trend strength in price movements.'),
    ('Div(TsMean($close, %d), $close)', 'Intraday price movements', 'This expression measures the relative deviation of the current closing price from its recent average by dividing the rolling mean of the closing price over the past %d days by the current closing price, which is useful for identifying overbought or oversold conditions when the ratio deviates significantly from 1.'),
    ('Div(TsMax($high, %d), $close)', 'Intraday price movements', 'This expression measures the ratio of the highest price over the past %d days to the current closing price, which is useful for identifying potential overbought conditions or resistance levels, as values significantly above 1.0 may indicate a price near recent highs and possible reversal points.'),
    ('TsRank($close, %d)', 'Intraday price movements', 'This expression measures the relative position of the current closing price within its recent historical distribution by computing the time-series rank of the closing price over the past %d days, which is useful for identifying momentum or mean-reversion signals based on whether the price is near recent highs or lows.'),
    ('Div(Sub($close, TsMin($low, %d)), Add(Sub(TsMax($high, %d), TsMin($low, %d)), 0.000001))', 'Intraday price movements', 'This expression measures the normalized position of the current close price within the recent price range by calculating (current close - minimum low over d days) / (maximum high over d days - minimum low over d days + a small constant to avoid division by zero), which is useful for identifying overbought or oversold conditions and potential reversal points in price action.'),
    ('Div(Sub(TsSum(GetGreater(Sub($close, Ref($close, 1)), 0.0), %d), TsSum(GetGreater(Sub(Ref($close, 1), $close), 0.0), %d)), Add(TsSum(Abs(Sub($close, Ref($close, 1))), %d), 0.000001))', 'Intraday price movements', 'This expression measures the modified RSI (Relative Strength Index) by calculating the ratio of the sum of upward price changes over a period to the sum of downward price changes over the same period, divided by the total absolute price movement plus a small constant to avoid division by zero, which is useful for identifying overbought or oversold conditions in a stock.'),
    ('Div(TsSum(GetGreater(Sub(Ref($close, 1), $close), 0.0), %d), Add(TsSum(Abs(Sub($close, Ref($close, 1))), %d), 0.000001))', 'Intraday price movements', 'This expression measures the relative strength of upward price movements over a specified period by calculating the ratio of the sum of positive daily returns to the sum of absolute daily returns, which is useful for identifying momentum and trend strength in price action.'),
    ('Div(TsSum(GetGreater(Sub($close, Ref($close, 1)), 0.0), %d), Add(TsSum(Abs(Sub($close, Ref($close, 1))), %d), 0.000001))', 'Intraday price movements', 'This expression measures the rolling momentum-to-volatility ratio by calculating the ratio of the sum of positive daily returns to the sum of absolute daily returns over a specified window, which is useful for identifying periods of sustained directional price movement relative to total price fluctuation, a common momentum indicator in quantitative strategies.'),
    ('Div(TsStd(Mul(Abs(Sub(Div($close, Ref($close, 1)), 1.0)), $volume), %d), Add(TsMean(Mul(Abs(Sub(Div($close, Ref($close, 1)), 1.0)), $volume), %d), 0.000001))', 'Intraday price movements', 'This expression measures the volatility-adjusted volume by computing the ratio of the rolling standard deviation to the rolling mean of the product of absolute daily returns and volume over a specified window, which is useful for identifying periods of abnormal trading activity relative to recent norms, potentially signaling momentum or reversal opportunities.'),
    ('Div(TsMean($volume, %d), Add($volume, 0.000001))', 'Intraday volume movements', "This expression measures the ratio of the average trading volume over the past %d days to the current day's volume, which is useful for identifying volume anomalies or deviations from recent norms, potentially signaling shifts in market sentiment or liquidity conditions."),
    ('Div(TsStd($volume, %d), Add($volume, 0.000001))', 'Intraday volume movements', 'This expression measures the relative volatility of trading volume by dividing the rolling standard deviation of volume over the past %d days by the current volume (with a small constant to avoid division by zero), which is useful for identifying periods of abnormal volume stability or instability that may signal shifts in market sentiment or liquidity conditions.'),
    ('Div(Sub(TsSum(GetGreater(Sub($volume, Ref($volume, 1)), 0.0), %d), TsSum(GetGreater(Sub(Ref($volume, 1), $volume), 0.0), %d)), Add(TsSum(Abs(Sub($volume, Ref($volume, 1))), %d), 0.000001))', 'Intraday volume movements', 'This expression measures the ratio of net positive volume flow to total absolute volume changes over a rolling window, which is useful for identifying periods of sustained buying or selling pressure and potential price momentum shifts.'),
    ('Div(TsSum(GetGreater(Sub($volume, Ref($volume, 1)), 0.0), %d), Add(TsSum(Abs(Sub($volume, Ref($volume, 1))), %d), 0.000001))', 'Intraday volume movements', 'This expression measures the proportion of positive volume changes relative to the total absolute volume changes over a rolling window, which is useful for identifying periods of sustained buying pressure or accumulation by comparing the net positive volume flow to the total volume volatility.'),
    ('Div(TsSum(GetGreater(Sub(Ref($volume, 1), $volume), 0.0), %d), Add(TsSum(Abs(Sub($volume, Ref($volume, 1))), %d), 0.000001))', 'Intraday volume movements', 'This expression measures the volume-based persistence by calculating the ratio of the sum of positive volume changes over the past %d days to the total absolute volume changes over the same period, which is useful for identifying stocks with consistent positive volume momentum and filtering out noise from erratic volume movements.'),
    ('TsCorr(Div($close, Ref($close, 1)), Log(Add(Div($volume, Ref($volume, 1)), 1.0)), %d)', 'Relationship between volume and prices', 'This expression measures the correlation between daily price returns and log-transformed daily volume growth over a specified period, which is useful for identifying momentum or liquidity-driven price movements by assessing how closely price changes align with changes in trading volume.'),
    ('TsCorr($close, Log(Add($volume, 1.0)), %d)', 'Relationship between volume and prices', 'This expression measures the time-series correlation between daily closing prices and the natural logarithm of trading volume over the past %d days, which is useful for identifying momentum or divergence patterns where price movements align with or deviate from volume trends.')
]

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if args.instruments == 'sp500':
        QLIB_PATH = 'PATH/TO/data/qlib_data/us_data_qlib'
    else:    
        QLIB_PATH = 'PATH/TO/.qlib/qlib_data/cn_data'
    # Initialize StockData and target expression
    data = StockData(instrument=args.instruments, start_time='2014-01-01', end_time='2020-12-31', qlib_path=QLIB_PATH)
    data_test = StockData(instrument=args.instruments, start_time='2022-07-01', end_time='2025-06-30', qlib_path=QLIB_PATH)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    # expression_parser = ExpressionParser()
    pool = AlphaKnowledgePool(
        capacity=args.pool_capacity,
        stock_data=data,
        target=target,
        ic_mut_threshold=args.ic_mut_threshold,
        top_k = args.top_k,
        depth_decay= args.depth_decay,
        embedding_model_name=args.embedding_model_name,
        use_res_correlation=args.use_res_correlation,
        use_semantic_similarity=args.use_semantic_similarity,
        use_edit_distance=args.use_edit_distance,
        separate_leaf_non_leaf=args.separate_leaf_non_leaf,
        device = "cuda:0"
    )
    trainer = AlphaKnowledgeTrainer(pool = pool, initial_expressions= INITIAL_EXPR, test_data= data_test, target= target, args=args)
    trainer.train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--pool_capacity', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=50)
    parser.add_argument('--depth_limit', type=int, default=7)
    parser.add_argument('--generate_num', type=int, default=5)
    parser.add_argument('--top_k', type=int, default= 15)
    parser.add_argument("--depth_decay", type=float, default=0.05)
    parser.add_argument("--embedding_model_name", type=str, default="xxx")
    parser.add_argument("--use_res_correlation", type=bool, default=True)
    parser.add_argument("--use_semantic_similarity", type=bool, default=True)
    parser.add_argument("--use_edit_distance", type=bool, default=False)
    parser.add_argument("--separate_leaf_non_leaf", type=bool, default=True)
    parser.add_argument('--search_time', type=int, default=20)
    parser.add_argument("--times_decay", type=float, default=0.10)
    parser.add_argument('--start_times', type=int, default=2)
    parser.add_argument('--ic_threshold', type=float, default=0.006)
    parser.add_argument('--ic_mut_threshold', type=float, default=0.9)
    parser.add_argument('--ic_new_threshold', type=float, default=0.45)
    parser.add_argument("--icir_decay_threshold", type=float, default=0.7)
    parser.add_argument('--threshold_ric', type=float, default=0.015, help="Threshold ic during testing")
    parser.add_argument('--threshold_ricir', type=float, default=0.15)
    parser.add_argument('--n_factors', type=int, default=15,
                        help='Maximum number of factors to select at each step.')
    parser.add_argument('--chunk_size', type=int, default=180,
                        help='Chunk size for calculating Spearman correlation.')
    parser.add_argument('--window', type=str, default='inf',
                        help="Rolling window size for factor evaluation. 'inf' for expanding window.")
    parser.add_argument('--label_days', type=int, default=20,
                        help="Number of days to label the target.")
    parser.add_argument('--corr_threshold', type=float, default=0.95,
                        help="Correlation threshold for multicollinearity detection.")
    parser.add_argument('--ridge_alpha', type=float, default=1e-6,
                        help="Ridge regression regularization parameter.")
    parser.add_argument('--use_vif', type=bool, default=False,
                        help="Whether to use VIF for multicollinearity detection.")
    parser.add_argument('--linear_dep_tol', type=float, default=1e-10,
                        help="Tolerance for linear dependence detection.")
    args = parser.parse_args()
    print(args)
    train(args)