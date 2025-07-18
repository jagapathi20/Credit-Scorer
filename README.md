# Credit Scoring Model

## Overview

This machine learning model assigns credit scores (0-1000) to DeFi protocol wallets based on historical transaction behavior. Higher scores indicate reliable, responsible usage while lower scores reflect risky, exploitative, or bot-like behavior.

## Scoring Methodology

### Core Components (Weighted Average)

1. **Behavioral Score (35% weight)** - Measures user engagement patterns
2. **Risk Score (25% weight)** - Evaluates financial risk indicators
3. **Consistency Score (20% weight)** - Assesses temporal behavior patterns
4. **Health Score (20% weight)** - Overall protocol participation health

### Feature Engineering

#### Transaction Metrics:
- Volume statistics (total, average, median, volatility)
- Action diversity (deposit, borrow, repay, redeem, liquidation)
- Asset diversification across interactions
- Temporal patterns (time spans, intervals, consistency)

#### Behavioral Indicators:
- Repay-to-borrow ratio (financial discipline)
- Activity consistency (regular vs erratic patterns)
- Transaction size uniformity (bot detection)
- Time-based variance (hour/day patterns)

## Scoring Logic

### Behavioral Score (0-100)
- **Action Diversity**: Rewards users engaging in multiple action types (deposit, borrow, repay)
- **Consistency Bonus**: Values stable transaction patterns over erratic behavior
- **Asset Diversification**: Rewards interaction with multiple assets
- **Long-term Engagement**: Bonuses for extended protocol participation
- **Activity Frequency**: Rewards regular transaction frequency

### Risk Score (0-100)
- **Liquidation Penalty**: Heavy penalties for liquidation events (-30 per liquidation)
- **Repayment Discipline**: Penalties for poor repay-to-borrow ratios
- **Volume Volatility**: Penalties for extremely erratic transaction sizes
- **Large Transaction Ratio**: Penalties for suspicious large transaction patterns

### Consistency Score (0-100)
- **Temporal Regularity**: Rewards consistent transaction timing
- **Activity Balance**: Values balanced deposit/borrow behavior
- **Interval Consistency**: Rewards regular transaction intervals

### Health Score (0-100)
- **Deposit Activity**: Rewards deposit behavior (protocol liquidity provision)
- **Repayment Discipline**: Bonuses for strong repayment patterns
- **Volume Participation**: Rewards meaningful transaction volumes
- **Active Engagement**: Bonuses for sustained protocol usage

## Risk Indicators & Penalties

### Bot Detection:
- Extremely regular timing patterns
- Uniform transaction sizes
- High-frequency activity with low variance
- **Penalty**: Up to 300 points reduction

### Minimum Activity Threshold:
- Users with <3 transactions receive 200-point penalty
- Ensures sufficient data for reliable scoring

### Liquidation Impact:
- Each liquidation event: -30 points base penalty
- High liquidation ratio: additional -50 points
- Reflects poor risk management

## Score Interpretation

| Score Range | Risk Level | Interpretation |
|-------------|------------|----------------|
| 800-1000 | Very Low Risk | Excellent: Highly reliable, responsible behavior |
| 600-799 | Low Risk | Good: Generally positive behavior patterns |
| 400-599 | Medium Risk | Fair: Mixed behavior patterns |
| 200-399 | High Risk | Poor: Concerning behavior patterns |
| 0-199 | Very High Risk | Very Poor: Problematic/bot-like behavior |

## Usage

### Required JSON Format

```json
[
  {
    "_id": {
      "$oid": "681d38fed63812d4655f571a"
    },
    "userWallet": "0x00000000001accfa9cef68cf5371a23025b6d4b6",
    "network": "polygon",
    "protocol": "aave_v2",
    "txHash": "0x695c69acf608fbf5d38e48ca5535e118cc213a89e3d6d2e66e6b0e3b2e8d4190",
    "logId": "0x695c69acf608fbf5d38e48ca5535e118cc213a89e3d6d2e66e6b0e3b2e8d4190_Deposit",
    "timestamp": 1629178166,
    "blockNumber": 1629178166,
    "action": "deposit",
    "actionData": {
      "type": "Deposit",
      "amount": "2000000000",
      "assetSymbol": "USDC",
      "assetPriceUSD": "0.9938318274296357543568636362026045",
      "poolId": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
      "userId": "0x00000000001accfa9cef68cf5371a23025b6d4b6"
    },
    "__v": 0,
    "createdAt": {
      "$date": "2025-05-08T23:06:39.465Z"
    },
    "updatedAt": {
      "$date": "2025-05-08T23:06:39.465Z"
    }
  }
]
```

### Python Usage

```python
from credit_scorer import CreditScorer

# Initialize scorer
scorer = CreditScorer()

# Generate scores from JSON transaction data
results = scorer.run_scoring("transactions.json", "scores.csv")

# Get detailed explanation for specific wallet
explanation = scorer.generate_score_explanation("0x123...", results)
```

## Model Validation

The model incorporates multiple validation mechanisms:

1. **Feature Normalization**: All component scores normalized to 0-100 range
2. **Outlier Handling**: Extreme values clipped to prevent score manipulation
3. **Multi-factor Analysis**: Weighted combination prevents single-factor bias
4. **Temporal Validation**: Time-based patterns validate genuine vs artificial behavior
5. **Safe Variance Calculation**: Handles edge cases for wallets with limited transaction history

## Extensibility

### Adding New Features:
- Modify `engineer_features()` method to include additional metrics
- Update component score calculations to incorporate new features
- Adjust feature weights in `__init__()` based on validation results

### Customizing Penalties:
- Modify `liquidation_penalty`, `bot_threshold` parameters
- Adjust component weights based on business requirements
- Add new risk indicators to `detect_bot_behavior()` method

### Integration:
- Model designed for batch processing of transaction data
- Supports real-time scoring with incremental updates
- Output format compatible with risk management systems

## Technical Notes

- **Performance**: Optimized for datasets up to 100K transactions
- **Memory**: Efficient pandas operations for large-scale processing
- **Scalability**: Component-based architecture supports distributed processing
- **Interpretability**: Detailed score breakdowns enable audit trails
- **Robustness**: Safe variance calculations handle edge cases (single transactions, missing data)

## Key Improvements

### Data Processing:
- Handles nested actionData structure from MongoDB documents
- Converts Unix timestamps to proper datetime objects
- Extracts amount and asset information from nested JSON structure
- Maps userWallet to wallet_address for internal consistency

### Variance Calculations:
- Safe variance calculation prevents NaN values
- Sensible defaults for edge cases:
  - `hour_variance`: 20 (moderate inconsistency)
  - `day_variance`: 60 (moderate inconsistency)
  - `volume_std`: 0 (no volatility penalty for insufficient data)

The model balances sophisticated behavioral analysis with practical interpretability, providing actionable credit risk insights for DeFi protocol management.