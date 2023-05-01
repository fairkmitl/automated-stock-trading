from settrade_v2 import Investor
from settrade_v2.errors import SettradeError

investor = Investor(
    app_id="gOhKkBtFBpue3MTd",
    app_secret="OO7zhuLqMR+bmnoiSwlXPO0WFZmG74Nm/+NOGHqNNFk=",
    broker_id="SANDBOX",
    app_code="SANDBOX",
    is_auto_queue=False)

# deri = investor.Derivatives(account_no="fair_wee-D")
equity = investor.Equity(account_no="fair_wee-E")

try:
    account_info = equity.get_account_info()
    print(account_info)
except SettradeError as e:
    print("---- error message  ----")
    print(e)
    print("---- error code ----")
    print(e.code)
    print("---- status code ----")
    print(e.status_code)
