import pickle
from langchain_community.document_loaders import WebBaseLoader



#fetch data
urls = [
     "https://deriv.com/"
    "https://deriv.com/trade-types/cfds/",
    "https://deriv.com/trade-types/options/digital-options/up-and-down/",
    "https://deriv.com/trade-types/options/digital-options/digits/",
    "https://deriv.com/trade-types/options/digital-options/in-out/",
    "https://deriv.com/trade-types/options/digital-options/reset-call-reset-put/",
    "https://deriv.com/trade-types/options/digital-options/high-low-ticks/",
    "https://deriv.com/trade-types/options/digital-options/touch-no-touch/",
    "https://deriv.com/trade-types/options/digital-options/asians/",
    "https://deriv.com/trade-types/options/digital-options/only-ups-only-downs/",
    "https://deriv.com/trade-types/options/digital-options/lookbacks/",
    "https://deriv.com/trade-types/options/accumulator-options/",
    "https://deriv.com/trade-types/options/vanilla-options/",
    "https://deriv.com/trade-types/options/turbo-options/",
    "https://deriv.com/trade-types/multiplier/",
    "https://deriv.com/dmt5/",
    "https://deriv.com/derivx/",
    "https://deriv.com/deriv-ctrader/",
    "https://deriv.com/dtrader/",
    "https://deriv.com/deriv-go/",
    "https://deriv.com/dbot/",
    "https://deriv.com/markets/forex/",
    "https://deriv.com/markets/synthetic/",
    "https://deriv.com/markets/stock/",
    "https://deriv.com/markets/exchange-traded-funds/",
    "https://deriv.com/markets/cryptocurrencies/",
    "https://deriv.com/markets/commodities/",
    "https://deriv.com/who-we-are/",
    "https://deriv.com/why-choose-us/",
    "https://deriv.com/partners/",
]


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

with open('data.pkl', 'wb') as file:
    pickle.dump(docs_list, file)
