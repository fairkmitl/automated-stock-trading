from settrade_v2 import Investor

investor = Investor(
    app_id="gOhKkBtFBpue3MTd",
    app_secret="OO7zhuLqMR+bmnoiSwlXPO0WFZmG74Nm/+NOGHqNNFk=",
    broker_id="SANDBOX",
    app_code="SANDBOX",
    is_auto_queue=False)

equity = investor.Equity(account_no="fair_wee-E")

def getPort():
    portfolio = equity.get_portfolios()
    return portfolio
    # print(portfolio)