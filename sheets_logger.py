"""
Google Sheets Logger - Schreibt tägliche Trade-Zusammenfassung ins Google Sheet
"""

import json
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_sheet():
    with open("config.json") as f:
        config = json.load(f)

    creds = Credentials.from_service_account_file(
        config["google_sheets"]["credentials_file"],
        scopes=SCOPES,
    )
    client = gspread.authorize(creds)
    spreadsheet = client.open(config["google_sheets"]["spreadsheet_name"])
    return spreadsheet


def ensure_headers(sheet):
    headers = [
        "Datum", "Uhrzeit", "Symbol", "Signal", "Einstieg", "Menge",
        "Stop Loss", "Take Profit", "Risiko %", "Risiko Betrag ($)",
        "RR Ratio", "Konto vor Trade ($)", "Order ID", "Status"
    ]
    if sheet.row_values(1) != headers:
        sheet.insert_row(headers, 1)


def log_trade(trade_data: dict):
    """
    Schreibt einen Trade ins Google Sheet.

    trade_data erwartet:
        symbol, signal, entry_price, quantity, sl_price, tp_price,
        risk_percent, risk_amount, rr_ratio, account_value, order_id, status
    """
    spreadsheet = get_sheet()

    # Heutiges Sheet (ein Tab pro Monat)
    month = datetime.now().strftime("%Y-%m")
    try:
        sheet = spreadsheet.worksheet(month)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=month, rows=1000, cols=20)
        ensure_headers(sheet)

    ensure_headers(sheet)

    now = datetime.now()
    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        trade_data.get("symbol", ""),
        trade_data.get("signal", "").upper(),
        trade_data.get("entry_price", ""),
        trade_data.get("quantity", ""),
        trade_data.get("sl_price", ""),
        trade_data.get("tp_price", ""),
        f"{trade_data.get('risk_percent', 1.0)}%",
        trade_data.get("risk_amount", ""),
        f"1:{trade_data.get('rr_ratio', 2.0)}",
        trade_data.get("account_value", ""),
        trade_data.get("order_id", ""),
        trade_data.get("status", ""),
    ]

    sheet.append_row(row)
    print(f"Trade ins Google Sheet geloggt: {trade_data.get('symbol')} {trade_data.get('signal')}")


def log_eod_summary(account_value: float, daily_pnl: float, trades_today: int):
    """
    Schreibt eine Tagesabschluss-Zeile ins Sheet.
    """
    spreadsheet = get_sheet()
    month = datetime.now().strftime("%Y-%m")
    try:
        sheet = spreadsheet.worksheet(month)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=month, rows=1000, cols=20)
        ensure_headers(sheet)

    now = datetime.now()
    summary_row = [
        now.strftime("%Y-%m-%d"),
        "TAGESABSCHLUSS",
        f"Trades heute: {trades_today}",
        "",
        "",
        "",
        "",
        "",
        "",
        f"PnL: {daily_pnl:+.2f}$",
        "",
        f"Kontostand: {account_value:.2f}$",
        "",
        "",
    ]
    sheet.append_row(summary_row)
    print(f"Tagesabschluss geloggt: Kontostand {account_value:.2f}$ | PnL {daily_pnl:+.2f}$")
