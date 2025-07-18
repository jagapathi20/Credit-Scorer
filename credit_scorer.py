import json
import pandas as pd
import numpy as np
import math
from collections import Counter
from typing import Dict, Any

class CreditScorer:
    def __init__(self):
        self.feature_weights = {
            'behavioral_score': 0.35,
            'risk_score': 0.25,
            'consistency_score': 0.2,
            'health_score':0.2
        }
        self.liquidation_penality = 200
        self.bot_threshold = 0.8
        self.min_transactions = 3

    def load_data(self, json_file:str) -> pd.DataFrame:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        
            df = pd.DataFrame(data)

            if 'actionData' in df.columns:
                action_data = pd.json_normalize(df['actionData'])
                df = pd.concat([df.drop('actionData', axis=1), action_data], axis=1)

            column_mapping = {
                'userWallet':'wallet_address',
                'amount':'amount',
                'assetSymbol':'asset',
                'timestamp':'timestamp',
                'action':'action'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]

            required_cols = ['wallet_address', 'action', 'timestamp', 'asset']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required colum: {col}")
                
            if df['timestamp'].dtype in ['int64', 'float64']:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

        
            def normalize_amount(row):
                if pd.isna(row['amount']):
                    return row['amount']
                
                decimals = {
                    'USDC': 6,
                    'USDT': 6,
                    'DAI' : 18,
                    'WETH': 18,
                    'ETH': 18,
                    'WMATIC': 18,
                    'MATIC': 18,
                    'WBTC': 8,
                    'WPOL': 18,
                    'AAVE': 18
                }

                token_decimals = decimals.get(row['asset'], 18)
                return row['amount'] / (10 ** token_decimals)
            
            df['amount'] = df.apply(normalize_amount, axis=1)
            df = df.dropna(subset=['amount'])
            df = df[df['amount'] > 0]

            df = df.sort_values(['wallet_address', 'timestamp'])

            return df
        
        except Exception as e:
            print(f"error loading data: {e}")
            return pd.DataFrame()
        
    
    def engineer_features(self, df:pd.DataFrame) -> pd.DataFrame:

        wallet_features = []

        for wallet in df['wallet_address'].unique():
            wallet_df = df[df['wallet_address'] == wallet].copy()

            total_txns = len(wallet_df)
            unique_actions = wallet_df['action'].nunique()
            unique_assets = wallet_df['asset'].nunique()

            time_span = (wallet_df['timestamp'].max() - wallet_df['timestamp'].min()).days
            avg_time_between_txns = time_span / max(total_txns - 1, 1)

            total_volume = wallet_df['amount'].sum()
            avg_txn_size = wallet_df['amount'].mean()
            median_txn_size = wallet_df['amount'].median()
            volume_std = wallet_df['amount'].std() if total_txns > 1 else 0

            action_counts = wallet_df['action'].value_counts()
            deposit_count = action_counts.get('deposit', 0)
            borrow_count = action_counts.get('borrow', 0)
            repay_count = action_counts.get('repay', 0)
            redeem_count = action_counts.get('redeemunderlying', 0)
            liquidation_count = action_counts.get('liquidationcall', 0)

            deposit_ratio = deposit_count / total_txns
            borrow_ratio = borrow_count / total_txns
            repay_ratio = repay_count / total_txns
            liquidation_ratio = liquidation_count / total_txns

            repay_to_borrow_ratio = repay_count / max(borrow_count, 1)
            activity_consistency = 1 / (1 + volume_std/ max(avg_txn_size, 1))

            def calculate_shannon_diversity(asset_series):
    
                if len(asset_series) == 0:
                    return 0.0
                
                # Count occurrences of each asset
                asset_counts = Counter(asset_series)
                total_transactions = len(asset_series)
                
                # Calculate Shannon diversity index
                shannon_index = 0.0
                for count in asset_counts.values():
                    if count > 0:
                        proportion = count / total_transactions
                        shannon_index -= proportion * math.log(proportion)
                
                return shannon_index

            def calculate_normalized_shannon_diversity(asset_series):
                
                if len(asset_series) == 0:
                        return 0.0
                    
                unique_assets = len(set(asset_series))
                if unique_assets <= 1:
                        return 0.0
                    
                shannon_index = calculate_shannon_diversity(asset_series)
                max_possible_diversity = math.log(unique_assets)
                    
                return shannon_index / max_possible_diversity

            asset_diversification = calculate_normalized_shannon_diversity(wallet_df['asset'])

            hour_variance = wallet_df['timestamp'].dt.hour.var() if total_txns > 1 else 20
            day_variance = wallet_df['timestamp'].dt.day.var() if total_txns > 1 else 60
            
            large_txn_ratio = (wallet_df['amount'] > wallet_df['amount'].quantile(0.9)).sum() / total_txns

            features = {
                    'wallet_address': wallet,
                    'total_transactions': total_txns,
                    'unique_actions': unique_actions,
                    'unique_assets': unique_assets,
                    'time_span_days': time_span,
                    'avg_time_between_txns': avg_time_between_txns,
                    'total_volume': total_volume,
                    'avg_txn_size': avg_txn_size,
                    'median_txn_size': median_txn_size,
                    'volume_std': volume_std,
                    'deposit_count': deposit_count,
                    'borrow_count': borrow_count,
                    'repay_count': repay_count,
                    'redeem_count': redeem_count,
                    'liquidation_count': liquidation_count,
                    'deposit_ratio': deposit_ratio,
                    'borrow_ratio': borrow_ratio,
                    'repay_ratio': repay_ratio,
                    'liquidation_ratio': liquidation_ratio,
                    'repay_to_borrow_ratio': repay_to_borrow_ratio,
                    'activity_consistency': activity_consistency,
                    'asset_diversification': asset_diversification,
                    'hour_variance': hour_variance,
                    'day_variance': day_variance,
                    'large_txn_ratio': large_txn_ratio
                }
                
            wallet_features.append(features)# use a single dictionary

        return pd.DataFrame(wallet_features)
    
    def calculate_behavioral_score(self, features: pd.DataFrame) -> pd.Series:

        behavioral_score = np.zeros(len(features))

        action_diversity = features['unique_actions'] / 5.0
        behavioral_score += action_diversity * 25

        consistency_bonus = features['activity_consistency'] * 20
        behavioral_score += consistency_bonus

        diversification_bonus = np.minimum(features['asset_diversification'] * 100, 15)
        behavioral_score += diversification_bonus

        time_bonus = np.minimum(features['time_span_days'] / 365 * 20, 20)
        behavioral_score += time_bonus

        freq_score = np.minimum(features['total_transactions'] / 50 * 20, 20) # dividing by 50
        behavioral_score += freq_score

        return np.clip(behavioral_score, 0, 100)

    def calculate_risk_score(self, features: pd.DataFrame) -> pd.Series:

        risk_score = np.full(len(features), 100.0)

        liquidation_penality = features['liquidation_count'] * 30
        risk_score -= liquidation_penality

        liquidation_ratio_penalty = features['liquidation_ratio'] * 50
        risk_score -= liquidation_ratio_penalty

        repay_penalty = np.maximum(0, (1 - features['repay_to_borrow_ratio']) * 20)
        risk_score -= repay_penalty

        large_txn_penalty = features['large_txn_ratio'] * 15
        risk_score -= large_txn_penalty

        volume_volatility = features['volume_std'] / features['avg_txn_size']
        volatility_penality = np.minimum(volume_volatility * 10, 25)
        risk_score -= volatility_penality

        return np.clip(risk_score, 0, 100)
    
    def calculate_consistency_score(self, features:pd.DataFrame) -> pd.Series:

        consistency_score = np.zeros(len(features))

        time_consistency = 100 - np.minimum(features['hour_variance'], 50)
        consistency_score += time_consistency * 0.3

        interval_consistency = 100 / (1 + features['avg_time_between_txns'] / 30)
        consistency_score += interval_consistency * 0.4

        balance_score = 100 - abs(features['deposit_ratio'] - features['borrow_ratio']) * 100
        consistency_score += balance_score * 0.3

        return np.clip(consistency_score, 0, 100)
    
    def calculate_health_score(self, features: pd.DataFrame) -> pd.Series:

        health_score = np.zeros(len(features))

        deposit_bonus = features['deposit_ratio'] * 30
        health_score += deposit_bonus

        repay_discipline = np.minimum(features['repay_to_borrow_ratio'], 2) * 25
        health_score += repay_discipline
        
        volume_score = np.minimum(np.log1p(features['total_volume']) * 5, 25)
        health_score += volume_score
        
       
        activity_bonus = np.minimum(features['total_transactions'] / 20 * 20, 20)
        health_score += activity_bonus
        
        return np.clip(health_score, 0, 100)

    def detect_bot_behaviour(self, features: pd.DataFrame) -> pd.Series:

        bot_indicators = np.zeros(len(features))

        bot_indicators += (features['hour_variance'] < 1).astype(int) * 0.3
        bot_indicators += (features['avg_time_between_txns'] < 0.1).astype(int) * 0.2
        bot_indicators += (features['total_transactions'] > 100).astype(int) * 0.2 # maybe change this

        size_uniformity = features['volume_std'] / features['avg_txn_size']
        bot_indicators += (size_uniformity < 0.1).astype(int) * 0.3

        return bot_indicators
    
    def calculate_credit_score(self, features:pd.DataFrame) -> pd.DataFrame:
        results = features[['wallet_address']].copy()

        behavioral = self.calculate_behavioral_score(features)
        risk = self.calculate_risk_score(features)
        consistency = self.calculate_consistency_score(features)
        health = self.calculate_health_score(features)

        bot_probability = self.detect_bot_behaviour(features)

        final_score = (
            behavioral * self.feature_weights['behavioral_score']+
            risk * self.feature_weights['risk_score'] +
            consistency * self.feature_weights['consistency_score'] +
            health * self.feature_weights['health_score']
        )

        bot_penalty = bot_probability * 300
        final_score -= bot_penalty

        min_txn_penalty = np.where(
            features['total_transactions'] < self.min_transactions,
            200.0,
            0.0
        )
        final_score -= min_txn_penalty

        final_score = np.clip(final_score * 10, 0, 1000)

        results['behavioral_score'] = behavioral
        results['risk_score'] = risk
        results['consistency_score'] = consistency
        results['health_score'] = health
        results['bot_probability'] = bot_probability
        results['credit_score'] = final_score.round().astype(int)
        results['total_transactions'] = features['total_transactions']
        results['liquidation_count'] = features['liquidation_count']
        
        return results
    
    def generate_score_explanation(self, wallet_address: str, results: pd.DataFrame) -> Dict[str, Any]:

        wallet_data = results[results['wallet_address'] == wallet_address]

        if wallet_data.empty:
            return {"error" : "wallet not found"}

        row = wallet_data.iloc[0]
        
        explanation = {
            "wallet_address": wallet_address,
            "credit_score": int(row['credit_score']), 
            "risk_level": self._get_risk_level(row['credit_score']),
            "component_scores": {
                "behavioral": f"{row['behavioral_score']:.1f}/100",
                "risk": f"{row['risk_score']:.1f}/100",
                "consistency": f"{row['consistency_score']:.1f}/100",
                "health": f"{row['health_score']:.1f}/100"
            },
            "key_metrics": {
                "total_transactions": int(row['total_transactions']),
                "liquidation_count": int(row['liquidation_count']),
                "bot_probability": f"{row['bot_probability']:.2f}"
            },
            "interpretation": self._interpret_score(row['credit_score'])
        }
        
        return explanation
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 800:
            return "Very Low Risk"
        elif score >= 600:
            return "Low Risk"
        elif score >= 400:
            return "Medium Risk"
        elif score >= 200:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _interpret_score(self, score: float) -> str:
        if score >= 800:
            return "Excellent: Highly reliable user with consistent, responsible behavior"
        elif score >= 600:
            return "Good: Reliable user with generally positive behavior patterns"
        elif score >= 400:
            return "Fair: Average user with mixed behavior patterns"
        elif score >= 200:
            return "Poor: Risky user with concerning behavior patterns"
        else:
            return "Very Poor: High-risk user with problematic or bot-like behavior"
        
    def run_scoring(self, json_file: str, output_file: str = None) -> pd.DataFrame:

        print("Loading transaction data..")
        df = self.load_data(json_file)

        if df.empty:
            print("no data loaded. please check your json file")
            return pd.DataFrame()
        
        print(f"Loaded {len(df)} transactions for {df['wallet_address'].nunique()} wallets")

        print("Engineering features...")
        features = self.engineer_features(df)
        
        print("Calculating credit scores...")
        results = self.calculate_credit_score(features)
        
        results = results.sort_values('credit_score', ascending=False)
        
        print(f"Generated scores for {len(results)} wallets")
        print(f"Score distribution:")
        print(f"  Mean: {results['credit_score'].mean():.1f}")
        print(f"  Median: {results['credit_score'].median():.1f}")
        print(f"  Min: {results['credit_score'].min()}")
        print(f"  Max: {results['credit_score'].max()}")
        
        if output_file:
            results.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results
    

def main():

    scorer = CreditScorer()

    from pathlib import Path
    dir = Path(__file__).parent
    file_path = dir / 'data'/ 'user-wallet-transactions.json'   
    
    json_file = file_path
    results = scorer.run_scoring(json_file, "scores.csv")
    
    if not results.empty:
        print("\nTop 10 wallets by credit score:")
        print(results[['wallet_address', 'credit_score', 'total_transactions', 'liquidation_count']].head(10))
        
        
        if len(results) > 0:
            sample_wallet = results.iloc[0]['wallet_address']
            explanation = scorer.generate_score_explanation(sample_wallet, results)
            print(f"\nExample explanation for {sample_wallet}:")
            print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    main()
