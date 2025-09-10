"""
Simple Automation Script for ChatGPT Micro-Cap Trading

This script integrates with the existing trading_script.py to provide
automated LLM-based trading decisions.

Usage:
    python simple_automation.py --api-key YOUR_KEY
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import yfinance as yf

# Import existing trading functions
from trading_script import (
    process_portfolio, daily_results, load_latest_portfolio_state,
    set_data_dir, PORTFOLIO_CSV, TRADE_LOG_CSV, last_trading_date
)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def generate_trading_prompt(portfolio_df: pd.DataFrame, cash: float, total_equity: float) -> str:
    """Generate a trading prompt with current portfolio data"""
    
    # Format holdings
    if portfolio_df.empty:
        holdings_text = "No current holdings"
    else:
        holdings_text = portfolio_df.to_string(index=False)
    
    # Get current date
    today = last_trading_date().date().isoformat()
    
    prompt = f"""You are a professional portfolio analyst. Here is your current portfolio state as of {today}:

[ Holdings ]
{holdings_text}

[ Snapshot ]
Cash Balance: ${cash:,.2f}
Total Equity: ${total_equity:,.2f}

Rules:
- You have ${cash:,.2f} in cash available for new positions
- Prefer U.S. micro-cap stocks (<$300M market cap)
- Full shares only, no options or derivatives
- Use stop-losses for risk management
- Be conservative with position sizing

STRICT CONSTRAINTS (must follow):
- Only output real, US-listed stock/ETF tickers. Do NOT invent placeholders (e.g., ABCD, EFGH).
- Ticker format: 1–5 uppercase letters; class shares like BRK.B are allowed.
- If you are not confident a ticker is real and currently listed, output no trade for it.
- If there are no high-conviction ideas that meet constraints, return an empty trades array.

Respond with ONLY a JSON object in this exact format:
{{
    "analysis": "Brief market analysis",
    "trades": [
        {{
            "action": "buy",
            "ticker": "SYMBOL",
            "shares": 100,
            "price": 25.50,
            "stop_loss": 20.00,
            "reason": "Brief rationale"
        }}
    ],
    "confidence": 0.8
}}

Only recommend trades you are confident about. If no trades are recommended, use an empty trades array."""
    
    return prompt


def call_openai_api(prompt: str, api_key: str, model: str = "gpt-4") -> str:
    """Call OpenAI API and return response"""
    if not HAS_OPENAI:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional portfolio analyst. Always respond with valid JSON in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "API call failed: {e}"}}'


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response and extract trading decisions"""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response: {e}")
        print(f"Raw response: {response}")
        return {"error": "Failed to parse response", "raw_response": response}


def execute_automated_trades(trades: List[Dict[str, Any]], portfolio_df: pd.DataFrame, cash: float, cost_rate: float = 0.0) -> tuple[pd.DataFrame, float]:
    """Execute trades recommended by LLM

    Applies a per-trade cost/slippage rate to effective prices:
      - BUY effective price: price * (1 + cost_rate)
      - SELL effective price: price * (1 - cost_rate)
    """
    
    print(f"\n=== Executing {len(trades)} LLM-recommended trades ===")
    total_buy_cost = 0.0
    total_sell_proceeds = 0.0
    starting_cash = cash
    
    for trade in trades:
        action = trade.get('action', '').lower()
        ticker = trade.get('ticker', '').upper()
        shares = float(trade.get('shares', 0))
        price = float(trade.get('price', 0))
        stop_loss = float(trade.get('stop_loss', 0))
        reason = trade.get('reason', 'LLM recommendation')
        market_price = float(trade.get('market_price', price) or price)
        orig_price = trade.get('_llm_price', None)
        adjusted = orig_price is not None and abs(float(orig_price) - market_price) > 0.005
        suffix = f" (mkt ${market_price:.2f}, LLM ${float(orig_price):.2f})" if adjusted else ""
        header = f"{action.upper()}: {shares:g} shares of {ticker} at ${price:.2f}{suffix}"
        
        if action == 'buy':
            if shares > 0 and price > 0 and ticker:
                eff_price = price * (1.0 + float(cost_rate))
                cost = shares * eff_price
                if cost <= cash:
                    print(f"BUY: {shares:g} shares of {ticker} at ${price:.2f} (stop: ${stop_loss:.2f}) - {reason}")
                    print(f"  -> {header} | cost ${cost:,.2f} (eff px ${eff_price:.2f}, rate {float(cost_rate):.2%})")
                    total_buy_cost += cost
                    # Here you would call the actual buy function from trading_script
                    # For now, just simulate the trade
                    cash -= cost
                    print(f"  Simulated: Cash reduced by ${cost:.2f}, new balance: ${cash:.2f}")
                else:
                    print(f"BUY REJECTED: {ticker} - Insufficient cash (need ${cost:.2f}, have ${cash:.2f})")
            else:
                print(f"INVALID BUY ORDER: {trade}")
        
        elif action == 'sell':
            if shares > 0 and price > 0 and ticker:
                eff_price = price * (1.0 - float(cost_rate))
                proceeds = shares * eff_price
                print(f"SELL: {shares:g} shares of {ticker} at ${price:.2f} - {reason}")
                print(f"  -> {header} | proceeds ${proceeds:,.2f} (eff px ${eff_price:.2f}, rate {float(cost_rate):.2%})")
                total_sell_proceeds += proceeds
                # Here you would call the actual sell function from trading_script
                # For now, just simulate the trade
                cash += proceeds
                print(f"  Simulated: Cash increased by ${proceeds:.2f}, new balance: ${cash:.2f}")
            else:
                print(f"INVALID SELL ORDER: {trade}")
        
        elif action == 'hold':
            print(f"HOLD: {ticker} - {reason}")
        
        else:
            print(f"UNKNOWN ACTION: {action} for {ticker}")
    
    # Summary totals for executed trades
    if total_buy_cost > 0:
        print(f"Total buy cost (incl. {float(cost_rate):.2%}): ${total_buy_cost:,.2f}")
    if total_sell_proceeds > 0:
        print(f"Total sell proceeds (net of {float(cost_rate):.2%}): ${total_sell_proceeds:,.2f}")
    if (total_buy_cost + total_sell_proceeds) > 0:
        net = total_buy_cost - total_sell_proceeds
        impact_sign = '-' if net >= 0 else '+'
        print(f"Net cash impact: {impact_sign}${abs(net):,.2f}")
        print(f"Cash after execution: ${starting_cash:,.2f} -> ${cash:,.2f}")
    
    return portfolio_df, cash


def run_automated_trading(api_key: str, model: str = "gpt-4", data_dir: str = "Start Your Own", dry_run: bool = False, cost_rate: float = 0.001):
    """Run the automated trading process

    cost_rate: combined fees + slippage fraction (e.g., 0.001 = 0.1%)
    """
    
    print("=== Automated Trading System ===")
    
    # Set up data directory
    data_path = Path(data_dir)
    set_data_dir(data_path)
    
    # Load current portfolio
    portfolio_file = data_path / "chatgpt_portfolio_update.csv"
    if portfolio_file.exists():
        portfolio_df, cash = load_latest_portfolio_state(str(portfolio_file))
    else:
        portfolio_df = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        cash = 10000.0  # Default starting cash

    # Normalize portfolio to DataFrame shape (load_latest_portfolio_state may return list[dict])
    if isinstance(portfolio_df, list):
        portfolio_df = pd.DataFrame(portfolio_df)
    if portfolio_df is None or not isinstance(portfolio_df, pd.DataFrame):
        portfolio_df = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
    expected_cols = ["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
    for col in expected_cols:
        if col not in portfolio_df.columns:
            portfolio_df[col] = pd.Series(dtype="float" if col != "ticker" else "string")

    # Calculate total equity (simplified) safely
    cost_basis_series = pd.to_numeric(portfolio_df.get("cost_basis", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    total_value = float(cost_basis_series.sum())
    total_equity = cash + total_value
    
    print(f"Portfolio loaded: ${cash:,.2f} cash, ${total_equity:,.2f} total equity")
    
    # Generate prompt
    prompt = generate_trading_prompt(portfolio_df, cash, total_equity)
    print(f"\nGenerated prompt ({len(prompt)} characters)")

    # Save prompt before sending
    requests_file = data_path / "llm_requests.jsonl"
    with open(requests_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": pd.Timestamp.now().isoformat(),
            "model": model,
            "prompt": prompt
        }) + "\n")
    
    # Call LLM
    print("Calling LLM for trading recommendations...")
    response = call_openai_api(prompt, api_key, model)
    print(f"Received response ({len(response)} characters)")
    
    # Parse response
    parsed_response = parse_llm_response(response)
    
    if "error" in parsed_response:
        print(f"Error: {parsed_response['error']}")
        return
    
    # Display analysis
    analysis = parsed_response.get('analysis', 'No analysis provided')
    confidence = parsed_response.get('confidence', 0.0)
    trades_raw = parsed_response.get('trades', [])

    # Validate trades
    valid_trades, rejected_msgs, adjusted_msgs = validate_trades(trades_raw)
    
    print(f"\n=== LLM Analysis ===")
    print(f"Analysis: {analysis}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Recommended trades (valid): {len(valid_trades)}  |  adjusted: {len(adjusted_msgs)}  |  rejected: {len(rejected_msgs)}")
    for msg in adjusted_msgs:
        print(f"  * {msg}")
    for msg in rejected_msgs:
        print(f"  - {msg}")
    
    # Execute trades
    if valid_trades and not dry_run:
        portfolio_df, cash = execute_automated_trades(valid_trades, portfolio_df, cash, cost_rate=cost_rate)
    elif valid_trades and dry_run:
        print(f"\n=== DRY RUN - Would execute {len(valid_trades)} trades ===")
        total_buy_cost = 0.0
        total_sell_proceeds = 0.0
        for trade in valid_trades:
            action = str(trade.get('action', 'unknown')).lower()
            ticker = trade.get('ticker', 'unknown')
            shares = float(trade.get('shares', 0) or 0)
            used = float(trade.get('price', 0) or 0)
            mkt = float(trade.get('market_price', used) or used)
            orig = trade.get('_llm_price', None)
            adjusted = orig is not None and abs(float(orig) - mkt) > 0.005
            suffix = f" (mkt ${mkt:.2f}, LLM ${float(orig):.2f})" if adjusted else ""
            header = f"  {action.upper()}: {shares:g} shares of {ticker} at ${used:.2f}{suffix}"
            
            if action == 'buy':
                eff_px = used * (1.0 + float(cost_rate))
                line_total = shares * eff_px
                total_buy_cost += line_total
                print(f"{header} | cost ${line_total:,.2f} (eff px ${eff_px:.2f}, rate {float(cost_rate):.2%})")
            elif action == 'sell':
                eff_px = used * (1.0 - float(cost_rate))
                line_total = shares * eff_px
                total_sell_proceeds += line_total
                print(f"{header} | proceeds ${line_total:,.2f} (eff px ${eff_px:.2f}, rate {float(cost_rate):.2%})")
            else:
                print(header)
        
        # Summary totals
        if total_buy_cost > 0:
            print(f"Total buy cost (incl. {float(cost_rate):.2%}): ${total_buy_cost:,.2f}")
        if total_sell_proceeds > 0:
            print(f"Total sell proceeds (net of {float(cost_rate):.2%}): ${total_sell_proceeds:,.2f}")
        if (total_buy_cost + total_sell_proceeds) > 0:
            net = total_buy_cost - total_sell_proceeds
            impact_sign = '-' if net >= 0 else '+'
            print(f"Net cash impact: {impact_sign}${abs(net):,.2f}")
            try:
                cash_after = cash - net
                print(f"Cash after hypothetical: ${cash:,.2f} -> ${cash_after:,.2f}")
            except Exception:
                pass
    else:
        print("No valid trades recommended")
    
    # Save the LLM response for review
    response_file = data_path / "llm_responses.jsonl"
    with open(response_file, "a") as f:
        f.write(json.dumps({
            "timestamp": pd.Timestamp.now().isoformat(),
            "response": parsed_response,
            "raw_response": response
        }) + "\n")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Response saved to: {response_file}")


_TICKER_CACHE: Dict[str, bool] = {}

def is_valid_ticker(ticker: str) -> bool:
    """Return True if ticker looks like a real US-listed symbol with recent price history."""
    t = (ticker or "").upper().strip()
    if not t:
        return False
    # Allow 1-5 letters, optional "." or "-" class suffix (e.g., BRK.B)
    if not re.fullmatch(r"^[A-Z]{1,5}([.-][A-Z])?$", t):
        return False
    if t in _TICKER_CACHE:
        return _TICKER_CACHE[t]
    try:
        df = yf.download(t, period="3mo", progress=False, auto_adjust=True)
        ok = isinstance(df, pd.DataFrame) and not df.empty
        _TICKER_CACHE[t] = bool(ok)
        return bool(ok)
    except Exception:
        _TICKER_CACHE[t] = False
        return False

# --- Market price helper ---

def get_market_price(ticker: str) -> float | None:
    """Return most recent close/last price for ticker, or None if unavailable."""
    try:
        # Try a small recent window to improve reliability
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def validate_trades(trades: List[Dict[str, Any]], price_tolerance: float = 0.10) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Filter out invalid trades and adjust unreasonable prices.
    Returns (valid_trades, rejected_messages, adjusted_messages).
    """
    valid: List[Dict[str, Any]] = []
    rejected: List[str] = []
    adjusted: List[str] = []
    for tr in trades or []:
        action = str(tr.get("action", "")).lower().strip()
        ticker = str(tr.get("ticker", "")).upper().strip()
        if action not in {"buy", "sell", "hold"}:
            rejected.append(f"Rejected trade with unknown action: {tr}")
            continue
        if action in {"buy", "sell"}:
            if not ticker or not is_valid_ticker(ticker):
                rejected.append(f"Rejected invalid ticker: '{ticker}'")
                continue
            # Parse numbers
            try:
                shares = float(tr.get("shares", 0))
                llm_price = float(tr.get("price", 0))
            except (TypeError, ValueError):
                rejected.append(f"Rejected {ticker}: invalid shares/price")
                continue
            if shares <= 0:
                rejected.append(f"Rejected {ticker}: non-positive shares")
                continue
            # Fetch market price and compare
            mkt = get_market_price(ticker)
            if mkt is None or mkt <= 0:
                rejected.append(f"Rejected {ticker}: no market price available")
                continue
            tr["market_price"] = mkt
            # If LLM price missing or far from market, override to market
            if llm_price <= 0:
                tr["_llm_price"] = llm_price
                tr["price"] = mkt
                adjusted.append(f"Filled missing price for {ticker}: using market ${mkt:.2f}")
            else:
                diff = abs(llm_price - mkt) / mkt
                if diff > price_tolerance:
                    tr["_llm_price"] = llm_price
                    tr["price"] = mkt
                    adjusted.append(f"Adjusted price for {ticker}: LLM ${llm_price:.2f} → market ${mkt:.2f}")
                else:
                    tr["price"] = llm_price
        # Normalize ticker back into trade
        tr["ticker"] = ticker
        valid.append(tr)
    return valid, rejected, adjusted


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple Automated Trading")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument("--data-dir", default="Start Your Own", help="Data directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute trades, just show recommendations")
    parser.add_argument("--cost-rate", type=float, default=0.001, help="Fees+slippage rate as a fraction (e.g., 0.001 = 0.1%)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        return
    
    # Run automated trading
    run_automated_trading(
        api_key=api_key,
        model=args.model,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        cost_rate=args.cost_rate
    )


if __name__ == "__main__":
    main()