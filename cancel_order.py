from settrade_v2 import Investor

investor = Investor(
    app_id="gOhKkBtFBpue3MTd",
    app_secret="OO7zhuLqMR+bmnoiSwlXPO0WFZmG74Nm/+NOGHqNNFk=",
    broker_id="SANDBOX",
    app_code="SANDBOX",
    is_auto_queue=False)

equity = investor.Equity(account_no="fair_wee-E")
cancel_order = equity.cancel_order(order_no="32JRWINQC9", pin="000000")

print(cancel_order)
