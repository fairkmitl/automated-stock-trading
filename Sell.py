from settrade_v2 import Investor
from settrade_v2.errors import SettradeError

investor = Investor(
    app_id="gOhKkBtFBpue3MTd",
    app_secret="OO7zhuLqMR+bmnoiSwlXPO0WFZmG74Nm/+NOGHqNNFk=",
    broker_id="SANDBOX",
    app_code="SANDBOX",
    is_auto_queue=False)

equity = investor.Equity(account_no="fair_wee-E")

try:
    place_order = equity.place_order(
        side="Sell",
        symbol="PTT",
        trustee_id_type="Local",
        volume=100,
        qty_open=0,
        price=30,
        price_type="Limit",
        validity_type="Day",
        bypass_warning=False,
        # valid_till_date="2022-09-01",
        pin="000000"
    )
    print(place_order)
except SettradeError as e:
    print("---- error message  ----")
    print(e)
    print("---- error code ----")
    print(e.code)
    print("---- status code ----")
    print(e.status_code)
