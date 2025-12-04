import pandas as pd
from typing import Dict, List
from datetime import datetime

class PaperTrader:
    """
    Paper trading simulator for binary options.
    Tracks hypothetical trades and calculates P&L.
    """
    
    def __init__(self, initial_balance: float = 10000.0, payout_ratio: float = 0.85, trade_amount: float = 10.0):
        """
        Initialize paper trader.
        
        Args:
            initial_balance (float): Starting balance in USD
            payout_ratio (float): Payout ratio (e.g., 0.85 means 85% profit on winning trades)
            trade_amount (float): Fixed amount per trade
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.payout_ratio = payout_ratio
        self.trade_amount = trade_amount
        self.trades = []
        self.trade_history = []
    
    def execute_trade(self, signal: Dict, actual_outcome: int, timestamp: datetime = None):
        """
        Execute a paper trade.
        
        Args:
            signal (Dict): Signal from SignalGenerator with 'action' and 'probability'
            actual_outcome (int): Actual outcome (1 = UP, 0 = DOWN)
            timestamp (datetime): Trade timestamp
        """
        if signal['action'] == 'NO_TRADE':
            return
        
        # Determine if trade won
        if signal['action'] == 'CALL':
            won = (actual_outcome == 1)
        elif signal['action'] == 'PUT':
            won = (actual_outcome == 0)
        else:
            return
        
        # Calculate P&L
        if won:
            profit = self.trade_amount * self.payout_ratio
        else:
            profit = -self.trade_amount
        
        self.balance += profit
        
        # Record trade
        trade_record = {
            'timestamp': timestamp or datetime.now(),
            'action': signal['action'],
            'probability': signal['probability'],
            'outcome': 'WIN' if won else 'LOSS',
            'profit': profit,
            'balance': self.balance
        }
        
        self.trade_history.append(trade_record)
    
    def get_statistics(self) -> Dict:
        """
        Calculate trading statistics.
        
        Returns:
            Dict: Statistics including win rate, total profit, etc.
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'net_return': 0.0,
                'final_balance': self.balance
            }
        
        df = pd.DataFrame(self.trade_history)
        
        total_trades = len(df)
        wins = (df['outcome'] == 'WIN').sum()
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        total_profit = df['profit'].sum()
        net_return = (self.balance - self.initial_balance) / self.initial_balance
        
        stats = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'net_return': net_return,
            'final_balance': self.balance,
            'initial_balance': self.initial_balance
        }
        
        return stats
    
    def print_summary(self):
        """Print trading summary."""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"PAPER TRADING SUMMARY")
        print(f"{'='*60}")
        print(f"Initial Balance:  ${stats['initial_balance']:,.2f}")
        print(f"Final Balance:    ${stats['final_balance']:,.2f}")
        print(f"Total Profit:     ${stats['total_profit']:,.2f}")
        print(f"Net Return:       {stats['net_return']:.2%}")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades:   {stats['total_trades']}")
        print(f"  Wins:           {stats['wins']}")
        print(f"  Losses:         {stats['losses']}")
        print(f"  Win Rate:       {stats['win_rate']:.2%}")
        print(f"{'='*60}\n")
    
    def save_trades(self, filepath: str):
        """Save trade history to CSV."""
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            df.to_csv(filepath, index=False)
            print(f"Trade history saved to: {filepath}")
