import requests
r = requests.get("http://localhost:6543/sentiment?text="+ "@VirginAmerica why can't you supp the biz traveler like @SouthwestAir  and have customer service like @JetBlue #neverflyvirginforbusiness #HowRu")
print(r._content)