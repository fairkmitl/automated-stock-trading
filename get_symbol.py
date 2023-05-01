from settrade_v2 import Investor
from settrade_v2.errors import SettradeError

investor = Investor(
    app_id="gOhKkBtFBpue3MTd",
    app_secret="OO7zhuLqMR+bmnoiSwlXPO0WFZmG74Nm/+NOGHqNNFk=",
    broker_id="SANDBOX",
    app_code="SANDBOX",
    is_auto_queue=False)

mkt_data = investor.MarketData()
res = mkt_data.get_quote_symbol("AOT")
print(res)