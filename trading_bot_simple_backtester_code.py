import polars as pl
import pandas as pd
from datetime import datetime
import numpy as np

def load_trade_data(csv_path):
    """
    Loads trade data from a CSV file using polars.
    Returns a polars DataFrame with a datetime 'timestamp' column.
    """
    trades = pl.read_csv(csv_path, try_parse_dates=True)
    if trades['timestamp'].dtype != pl.Datetime:
        trades = trades.with_columns(
            pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        )
    return trades

def filter_trades_by_date(trades, entry_dt, exit_dt):
    """
    Filters trades DataFrame between entry and exit datetimes.
    """
    return trades.filter(
        (pl.col('timestamp') >= entry_dt) & (pl.col('timestamp') <= exit_dt)
    )

def process_trade_row(row, state, slippage_pct, trading_fee_usd, initial_capital):
    """
    Processes a single trade row and updates the trading state.
    For buy-and-hold strategy: buy once at entry, hold until exit.
    Returns updated state and a record of the new state.
    """
    timestamp = row['timestamp']
    side = row['side'].upper()
    price = float(row['price'])
    volume = float(row['volume'])

    capital = state['capital']
    position_size = state['position_size']
    entry_price = state['entry_price']

    # For buy-and-hold strategy, we only buy once at the beginning
    # and hold until the exit datetime
    if side == 'BUY' and position_size == 0:
        # First buy - enter position
        effective_buy_price = price * (1 + slippage_pct)
        available_capital = capital - trading_fee_usd
        if available_capital > 0:
            shares_to_buy = available_capital / effective_buy_price
            position_size = shares_to_buy
            entry_price = effective_buy_price
            capital = 0
            print(f"BUY: {shares_to_buy:.6f} shares at ${effective_buy_price:.4f} (timestamp: {timestamp})")
    elif side == 'SELL' and position_size > 0:
        # For buy-and-hold, we only sell at the exit datetime
        # This will be handled by close_position_at_exit function
        pass

    new_state = {
        'capital': capital,
        'position_size': position_size,
        'entry_price': entry_price
    }
    record = {
        'timestamp': timestamp,
        'capital': capital,
        'position': position_size
    }
    return new_state, record

def close_position_at_exit(state, exit_dt, exit_price, slippage_pct, trading_fee_usd, initial_capital):
    """
    Closes any open position at the exit date using the exit price.
    This is where we sell in the buy-and-hold strategy.
    Returns updated state and a record of the new state.
    """
    capital = state['capital']
    position_size = state['position_size']
    entry_price = state['entry_price']

    if position_size > 0:
        if entry_price is not None:
            effective_exit_price = exit_price * (1 - slippage_pct)
            gross_pnl = (effective_exit_price - entry_price) * position_size
            net_pnl = gross_pnl - trading_fee_usd
            capital = initial_capital + net_pnl
            print(f"SELL: {position_size:.6f} shares at ${effective_exit_price:.4f}")
            print(f"Entry price: ${entry_price:.4f}, Exit price: ${effective_exit_price:.4f}")
            print(f"Gross PnL: ${gross_pnl:.2f}, Net PnL (after fees): ${net_pnl:.2f}")
        else:
            capital = initial_capital - trading_fee_usd
            print(f"Warning: No entry price found, applying trading fee only")
        position_size = 0
        entry_price = None
    else:
        print(f"No position to close at exit")

    record = {
        'timestamp': exit_dt,
        'capital': capital,
        'position': 0
    }
    new_state = {
        'capital': capital,
        'position_size': position_size,
        'entry_price': entry_price
    }
    return new_state, record

def kraken_backtest_simple_entry_exit(entry_date, exit_date, initial_capital, csv_path, 
                                     slippage_pct=0.0075, trading_fee_usd=0.50):
    """
    Simple backtester for Kraken trades implementing buy-and-hold strategy.
    Buys at entry datetime and holds until exit datetime.
    Returns a pandas DataFrame with timestamp, capital, and position columns.
    """
    trades = load_trade_data(csv_path)
    entry_dt = pd.to_datetime(entry_date)
    exit_dt = pd.to_datetime(exit_date)
    filtered_trades = filter_trades_by_date(trades, entry_dt, exit_dt)

    print(f"\nRunning BUY-AND-HOLD backtest from {entry_date} to {exit_date}\n")

    if filtered_trades.height == 0:
        result_data = {
            'timestamp': [entry_dt, exit_dt],
            'capital': [initial_capital, initial_capital],
            'position': [0, 0]
        }
        
        return pd.DataFrame(result_data)

    filtered_trades = filtered_trades.sort('timestamp')

    # Initial state
    state = {
        'capital': initial_capital,
        'position_size': 0,
        'entry_price': None
    }
    records = [{
        'timestamp': entry_dt,
        'capital': initial_capital,
        'position': 0
    }]

    # Process trades to implement buy-and-hold strategy
    # We scan through trades to find the first BUY opportunity to enter position
    # Once we buy, we hold until the exit datetime
    for row in filtered_trades.iter_rows(named=True):
        state, record = process_trade_row(
            row, state, slippage_pct, trading_fee_usd, initial_capital
        )
        records.append(record)
    
    # Close position at exit datetime (this is where we sell in buy-and-hold)
    # Find the price at or closest to the exit datetime
    exit_price = None
    if filtered_trades.height > 0:
        # Find the trade closest to the exit datetime
        exit_trades = filtered_trades.filter(pl.col('timestamp') <= exit_dt)
        if exit_trades.height > 0:
            # Use the last trade before or at exit datetime
            exit_price = exit_trades['price'][-1]
        else:
            # If no trades before exit, use the first available trade price
            exit_price = filtered_trades['price'][0]
    else:
        exit_price = state['entry_price'] if state['entry_price'] else 0
    
    if exit_price is None:
        exit_price = state['entry_price'] if state['entry_price'] else 0
    
    print(f"EXIT: Closing position at ${exit_price:.4f} (timestamp: {exit_dt})")
    state, record = close_position_at_exit(
        state, exit_dt, exit_price, slippage_pct, trading_fee_usd, initial_capital
    )
    if records[-1]['timestamp'] != exit_dt:
        records.append(record)

    return pd.DataFrame(records)

def print_backtest_summary(results, initial_capital):
    """
    Prints a summary of the backtest results.
    """
    print("\nBacktest Results:")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${results['capital'].iloc[-1]:,.2f}")
    print(f"Total Return: ${results['capital'].iloc[-1] - initial_capital:,.2f}")
    print(f"Return %: {((results['capital'].iloc[-1] / initial_capital) - 1) * 100:.2f}%")
    print("\n")

if __name__ == "__main__":
    # Example parameters for SOL trading
    entry_date = "2025-05-16 04:03:09"
    exit_date = "2025-05-17 00:27:20.726655960"
    initial_capital = 1000  # $1000 starting capital
    csv_path = "kraken_trades_fartcoin_usd_0626_0516_0523.csv"

    # Run backtest with default slippage (0.75%) and trading fee ($0.50)
    results = kraken_backtest_simple_entry_exit(
        entry_date=entry_date,
        exit_date=exit_date,
        initial_capital=initial_capital,
        csv_path=csv_path
    )
    print_backtest_summary(results, initial_capital)

    # Example of custom slippage and fees
    print("Custom Parameters Example:")
    custom_results = kraken_backtest_simple_entry_exit(
        entry_date=entry_date,
        exit_date=exit_date,
        initial_capital=initial_capital,
        csv_path=csv_path,
        slippage_pct=0.005,  # 1% slippage
        trading_fee_usd=1.00  # $1.00 trading fee
    )
    print(f"With 1% slippage and $1.00 fee:")
    print_backtest_summary(custom_results, initial_capital)

    # TODO: Output gross PnL per trade to other code for reusability
