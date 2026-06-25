name: Background Data Fetch

on:
  schedule:
    # Every 5 minutes, Mon-Fri, 03:00-10:55 UTC (08:30-16:25 IST) — wraps
    # NSE's 09:15-15:30 IST session with buffer; GitHub's cron scheduler is
    # best-effort and can run a few minutes late.
    - cron: "*/5 3-10 * * 1-5"
  workflow_dispatch: {}

jobs:
  fetch:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run background fetcher
        env:
          ANGEL_API_KEY: ${{ secrets.ANGEL_API_KEY }}
          ANGEL_CLIENT_ID: ${{ secrets.ANGEL_CLIENT_ID }}
          ANGEL_MPIN: ${{ secrets.ANGEL_MPIN }}
          ANGEL_TOTP_SECRET: ${{ secrets.ANGEL_TOTP_SECRET }}
          FIREBASE_SERVICE_ACCOUNT_JSON: ${{ secrets.FIREBASE_SERVICE_ACCOUNT_JSON }}
        run: python background_fetcher.py
